"""
DEKS Viver — интерактивное медиа-окно DEKS.
Устанавливается через магазин навыков.

Два режима:
  - Скилл (импортируется в DEKS): управление виверами, команды пользователя
  - Runtime (subprocess с --runtime): само окно на pywebview + HTTP API
"""
import os
import sys
import json
import time
import threading
import subprocess
import urllib.request

VIVER_PORT = 7547
VIVER_URL  = f"http://localhost:{VIVER_PORT}"


# ══════════════════════════════════════════════════════════════════════════════
#  RUNTIME — запускается как отдельный subprocess: python deks_viver.py --runtime
# ══════════════════════════════════════════════════════════════════════════════

if "--runtime" in sys.argv:
    import webview
    from http.server import HTTPServer, BaseHTTPRequestHandler

    _window = None
    _state  = {"url": "", "title": "", "fullscreen": False, "history": []}

    def _update_state():
        global _state
        try:
            url   = _window.evaluate_js("location.href")
            title = _window.evaluate_js("document.title")
            if url and url != "about:blank":
                _state["url"]   = url
            if title:
                _state["title"] = title
        except Exception:
            pass

    def _handle_command(action: str, body: dict):
        global _state, _window
        if _window is None:
            return "no_window"

        if action == "open_url":
            url = body.get("url", "")
            if url:
                _window.load_url(url)
                _state["url"]   = url
                _state["title"] = body.get("title", url)
                if url not in _state["history"]:
                    _state["history"].append(url)

        elif action == "play":
            _window.evaluate_js(
                "var v=document.querySelector('video');"
                "if(v&&v.paused){v.play();}"
                "else{document.dispatchEvent(new KeyboardEvent('keydown',{key:'k',bubbles:true,keyCode:75}));}"
            )

        elif action == "pause":
            _window.evaluate_js(
                "var v=document.querySelector('video');"
                "if(v&&!v.paused){v.pause();}"
            )

        elif action == "fullscreen":
            _state["fullscreen"] = not _state.get("fullscreen", False)
            _window.toggle_fullscreen()

        elif action == "back":
            _window.evaluate_js("history.back()")

        elif action == "forward":
            _window.evaluate_js("history.forward()")

        elif action == "reload":
            _window.evaluate_js("location.reload()")

        elif action == "close":
            _window.destroy()

        elif action == "get_state":
            _update_state()
            return _state

        return "ok"

    class _ViverHandler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass  # тишина

        def do_GET(self):
            if self.path == "/ping":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            elif self.path == "/state":
                _update_state()
                body = json.dumps(_state).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                body   = json.loads(self.rfile.read(length)) if length else {}
                action = body.get("action", "")
                result = _handle_command(action, body)
                resp   = json.dumps({"ok": True, "result": result or "ok"}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            except Exception as ex:
                err = json.dumps({"ok": False, "error": str(ex)}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)

    def _on_loaded(window):
        _update_state()

    def _start_http():
        server = HTTPServer(("127.0.0.1", VIVER_PORT), _ViverHandler)
        server.serve_forever()

    threading.Thread(target=_start_http, daemon=True).start()

    _window = webview.create_window(
        "DEKS Viver",
        url="about:blank",
        width=1280,
        height=780,
        resizable=True,
        background_color="#0a0a0a",
    )
    _window.events.loaded += _on_loaded

    webview.start(debug=False)
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
#  SKILL MODE — импортируется как скилл DEKS
# ══════════════════════════════════════════════════════════════════════════════

from skills.base_skill import BaseSkill

_TRIGGERS_OPEN = [
    "открой вивер", "открой deks viver", "открой интерактивное окно",
    "покажи интерактивное", "запусти вивер", "включи вивер",
]
_TRIGGERS_CLOSE = [
    "закрой вивер", "закрой интерактивное", "убери вивер", "выключи вивер",
]
_TRIGGERS_PLAY = [
    "плей", "нажми плей", "нажми play", "запусти видео", "включи видео",
    "воспроизвести",
]
_TRIGGERS_PAUSE = [
    "пауза", "поставь на паузу", "останови видео",
]
_TRIGGERS_FULLSCREEN = [
    "полный экран", "на весь экран", "разверни на весь", "fullscreen",
    "развернуть",
]
_TRIGGERS_BACK = [
    "назад в вивере", "вернись в вивере", "предыдущая страница",
]
_TRIGGERS_WRONG = [
    "ты открыл не то", "открыл не то", "не тот фильм", "не та страница",
    "не то видео", "неправильно открыл", "что сейчас открыто",
    "что в вивере", "что ты сейчас показываешь",
]

_ALL_TRIGGERS = (
    _TRIGGERS_OPEN + _TRIGGERS_CLOSE + _TRIGGERS_PLAY +
    _TRIGGERS_PAUSE + _TRIGGERS_FULLSCREEN + _TRIGGERS_BACK + _TRIGGERS_WRONG
)


class DeksViverSkill(BaseSkill):

    def __init__(self, app, name="deks_viver"):
        super().__init__(app, name)
        self._proc = None

    # ── Основной обработчик ───────────────────────────────────────────────

    def handle(self, command: str) -> str | None:
        cmd = command.lower().strip()

        if not any(t in cmd for t in _ALL_TRIGGERS):
            return None

        if any(t in cmd for t in _TRIGGERS_CLOSE):
            self._send("close")
            return "Закрываю Viver."

        if any(t in cmd for t in _TRIGGERS_OPEN):
            self._launch()
            return "Открываю DEKS Viver."

        if any(t in cmd for t in _TRIGGERS_FULLSCREEN):
            if not self.is_alive():
                return "Viver не открыт."
            self._send("fullscreen")
            return "Разворачиваю на весь экран."

        if any(t in cmd for t in _TRIGGERS_PLAY):
            if not self.is_alive():
                return "Viver не открыт."
            self._send("play")
            return "Запускаю."

        if any(t in cmd for t in _TRIGGERS_PAUSE):
            if not self.is_alive():
                return "Viver не открыт."
            self._send("pause")
            return "Пауза."

        if any(t in cmd for t in _TRIGGERS_BACK):
            if not self.is_alive():
                return "Viver не открыт."
            self._send("back")
            return "Назад."

        if any(t in cmd for t in _TRIGGERS_WRONG):
            threading.Thread(target=self._handle_wrong, daemon=True).start()
            return ""

        return None

    # ── Запуск процесса ───────────────────────────────────────────────────

    def _launch(self):
        if self.is_alive():
            return True
        script = os.path.abspath(__file__)
        self._proc = subprocess.Popen(
            [sys.executable, script, "--runtime"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(30):          # ждём до 6 сек пока HTTP поднимется
            if self.is_alive():
                return True
            time.sleep(0.2)
        return False

    # ── HTTP-команды ──────────────────────────────────────────────────────

    def _send(self, action: str, **kwargs) -> dict | None:
        try:
            body = json.dumps({"action": action, **kwargs}).encode()
            req  = urllib.request.Request(
                f"{VIVER_URL}/command",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3) as r:
                return json.loads(r.read().decode())
        except Exception:
            return None

    def get_state(self) -> dict:
        """Вернуть текущее состояние вивера (URL, title, и т.д.)"""
        try:
            self._send("get_state")   # обновляем через JS
            req = urllib.request.Request(f"{VIVER_URL}/state", method="GET")
            with urllib.request.urlopen(req, timeout=2) as r:
                return json.loads(r.read().decode())
        except Exception:
            return {}

    def is_alive(self) -> bool:
        """Проверить: работает ли вивер прямо сейчас."""
        try:
            req = urllib.request.Request(f"{VIVER_URL}/ping", method="GET")
            with urllib.request.urlopen(req, timeout=1) as r:
                return r.read() == b"ok"
        except Exception:
            return False

    # ── Открыть URL (вызывается извне, из ollama_mixin) ──────────────────

    def open_url(self, url: str) -> bool:
        """
        Открыть URL в вивере.
        Возвращает True если вивер жив и команда отправлена.
        """
        if not self.is_alive():
            return False
        result = self._send("open_url", url=url)
        return result is not None

    # ── Коррекция ошибки ──────────────────────────────────────────────────

    def _handle_wrong(self):
        """
        Пользователь сказал 'ты открыл не то'.
        Смотрим что сейчас в вивере → сообщаем → просим уточнение.
        """
        if not self.is_alive():
            self.app.after(0, lambda: self.app.deks_say(
                "Viver не открыт — скажи что именно нужно."
            ))
            return

        state = self.get_state()
        title = state.get("title", "")
        url   = state.get("url", "")
        shown = title if title and title != url else url

        if shown:
            msg = f"Сейчас открыто: {shown}. Что показать вместо этого?"
        else:
            msg = "Viver пустой. Что нужно открыть?"

        self.app.after(0, lambda m=msg: self.app.deks_say(m))


# ══════════════════════════════════════════════════════════════════════════════
#  Публичный хелпер: вызывается из ollama_mixin / llm_runtime_patch
# ══════════════════════════════════════════════════════════════════════════════

def try_open_in_viver(url: str) -> bool:
    """
    Если вивер запущен — открывает URL в нём и возвращает True.
    Если вивер не доступен — возвращает False (fallback в браузер).
    """
    try:
        body = json.dumps({"action": "open_url", "url": url}).encode()
        req  = urllib.request.Request(
            f"http://localhost:{VIVER_PORT}/command",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=1) as r:
            return r.status == 200
    except Exception:
        return False
