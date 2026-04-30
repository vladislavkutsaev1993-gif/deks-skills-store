"""
DEKS Viver Runtime Server.

Запускать как: python deks_viver/server.py
(или через subprocess из skills/deks_viver.py)

Что делает:
- Запускает Playwright с персистентным профилем Chrome
- Поднимает HTTP API на localhost:7547
- Главный цикл: cmd_queue → dispatch → result
- Playwright ТОЛЬКО в этом потоке (thread-safe через очередь)

TODO (будущее): заменить HTTP polling на WebSocket для realtime-событий
"""

import os
import sys
import json
import time
import queue
import uuid
import threading

# Добавляем корень проекта в path (чтобы импорты работали как из проекта)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from http.server import HTTPServer, BaseHTTPRequestHandler

VIVER_PORT = 7547

_cmd_q      = queue.Queue()
_result_map: dict = {}
_result_lck = threading.Lock()


# ── HTTP API ──────────────────────────────────────────────────────────────────

def _enqueue(action: str, body: dict, timeout: int = 15) -> object:
    """Поставить команду в очередь и ждать результата."""
    cid = str(uuid.uuid4())
    _cmd_q.put((cid, action, body))
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _result_lck:
            if cid in _result_map:
                return _result_map.pop(cid)
        time.sleep(0.05)
    return {"error": "timeout"}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass   # тишина в консоли

    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        elif self.path == "/state":
            res = _enqueue("get_state", {})
            self._reply(200, json.dumps(res or {}).encode())
        elif self.path == "/snapshot":
            res = _enqueue("get_snapshot", {})
            self._reply(200, json.dumps(res or {}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        try:
            n    = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n)) if n else {}
            res  = _enqueue(body.get("action", ""), body)
            self._reply(200, json.dumps({"ok": True, "result": res or "ok"}).encode())
        except Exception as ex:
            self._reply(500, json.dumps({"ok": False, "error": str(ex)}).encode())

    def _reply(self, code: int, data: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _start_http():
    HTTPServer(("127.0.0.1", VIVER_PORT), _Handler).serve_forever()


# ── Command dispatch ──────────────────────────────────────────────────────────

def _dispatch(action: str, body: dict, ctrl, session, planner) -> object:
    from deks_viver.permissions import check as perm_check

    allowed, reason = perm_check(action)
    if allowed is False:
        return {"error": f"blocked: {reason}"}

    # ── Навигация ───────────────────────────────────────────────────────
    if action == "open_url":
        url = body.get("url", "")
        ctrl.open_url(url)
        session.update_from_dom(ctrl.get_dom_state())
        session.save()
        return session.to_dict()

    if action == "execute_goal":
        return planner.execute(
            body.get("goal", ""),
            body.get("url", ""),
            ctrl, session,
        )

    if action == "back":
        ctrl.back()
        session.update_from_dom(ctrl.get_dom_state())
        return session.to_dict()

    if action == "forward":
        ctrl.forward()
        session.update_from_dom(ctrl.get_dom_state())
        return session.to_dict()

    if action == "reload":
        ctrl.reload()
        session.update_from_dom(ctrl.get_dom_state())
        return session.to_dict()

    # ── Медиа ───────────────────────────────────────────────────────────
    if action in ("play", "force_play"):
        ctrl.force_play()
        return "ok"

    if action == "pause":
        ctrl.pause()
        return "ok"

    if action == "fullscreen":
        ctrl.fullscreen()
        return "ok"

    # ── UI ──────────────────────────────────────────────────────────────
    if action == "click":
        ctrl.click(body.get("selector"), body.get("x"), body.get("y"))
        return "ok"

    if action == "type":
        ctrl.type_text(body.get("text", ""), body.get("selector"))
        return "ok"

    if action == "key":
        ctrl.key(body.get("key", ""))
        return "ok"

    if action == "scroll":
        ctrl.scroll(body.get("direction", "down"), body.get("amount", 3))
        return "ok"

    if action == "dismiss_popups":
        return ctrl.dismiss_popups()

    if action == "execute_js":
        return ctrl.execute_js(body.get("code", ""))

    # ── Наблюдение ──────────────────────────────────────────────────────
    if action == "get_state":
        session.update_from_dom(ctrl.get_dom_state())
        return session.to_dict()

    if action == "get_snapshot":
        snap = ctrl.get_dom_snapshot()
        session.update_from_dom(snap)
        return snap

    if action == "capture":
        return {"path": ctrl.capture()}

    # ── Lifecycle ────────────────────────────────────────────────────────
    if action == "close":
        ctrl.close()
        return "closing"

    return "ok"


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from playwright.sync_api import sync_playwright
    from deks_viver.browser  import PlaywrightBackend
    from deks_viver.session  import ViewerSession
    from deks_viver.goals    import GoalPlanner

    # Поднимаем HTTP в фоне
    threading.Thread(target=_start_http, daemon=True).start()
    print(f"[VIVER] HTTP API listening on localhost:{VIVER_PORT}")

    # Загружаем сохранённую сессию
    session = ViewerSession.load()
    planner = GoalPlanner()

    with sync_playwright() as pw:
        ctrl, ctx = PlaywrightBackend.launch(pw)
        print("[VIVER] Browser ready")

        # ── Основной цикл Playwright (единственный поток) ───────────────
        running = True
        while running:
            try:
                cid, action, body = _cmd_q.get(timeout=0.15)
            except queue.Empty:
                # Проверяем жив ли браузер
                try:
                    if not ctx.browser.is_connected():
                        break
                except Exception:
                    break
                continue

            try:
                result = _dispatch(action, body, ctrl, session, planner)
            except Exception as ex:
                print(f"[VIVER] Error in '{action}': {ex}")
                result = {"error": str(ex)}

            with _result_lck:
                _result_map[cid] = result

            if result == "closing":
                running = False

        # Сохраняем сессию при закрытии
        session.save()
        print("[VIVER] Session saved. Closing.")
        try:
            ctx.close()
        except Exception:
            pass
