"""
DEKS Viver — интерактивное пространство действий DEKS.

Архитектура (как советовал GPT):
  ┌─ DEKS Agent Layer ──────────────────────────────┐
  │  ├─ Viewer Runtime (Playwright, headless=False)  │
  │  │    └─ ViewerController (abstraction layer)    │
  │  └─ HTTP Bridge localhost:7547                   │
  └─────────────────────────────────────────────────┘

Если установлен — ВСЁ web-содержимое идёт через него.
Авто-запуск. Авто-плей. Авто-dismiss попапов.
Видит что открыто через DOM (не только скриншот).
"""

import os
import sys
import json
import time
import queue
import uuid
import threading
import subprocess
import urllib.request

VIVER_PORT = 7547
VIVER_URL  = f"http://localhost:{VIVER_PORT}"

_viver_skill_instance = None   # синглтон, ставится при загрузке скилла


# ══════════════════════════════════════════════════════════════════════════════
#  RUNTIME — subprocess: python deks_viver.py --runtime
# ══════════════════════════════════════════════════════════════════════════════

if "--runtime" in sys.argv:

    from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout
    from http.server import HTTPServer, BaseHTTPRequestHandler

    # Очередь команд (HTTP-поток → Playwright-поток)
    _cmd_q      = queue.Queue()
    _result_map = {}
    _result_lck = threading.Lock()

    # Глобальное состояние вивера
    _state = {
        "url":        "",
        "title":      "",
        "playing":    False,
        "fullscreen": False,
        "history":    [],
        "last_goal":  "",
    }

    # ── ViewerController — абстракция над Playwright ──────────────────────

    class ViewerController:
        """
        Abstraction layer над браузером.
        Бэкенд (Playwright) можно заменить — публичный API останется.
        """

        def __init__(self, page):
            self.page = page

        # ── Навигация ─────────────────────────────────────────────────────

        def open_url(self, url: str):
            self.page.goto(url, wait_until="domcontentloaded", timeout=20000)
            time.sleep(1.2)
            self.dismiss_popups()
            time.sleep(0.8)
            self._try_autoplay()
            self._sync_state()

        def back(self):
            self.page.go_back(timeout=8000)
            self._sync_state()

        def forward(self):
            self.page.go_forward(timeout=8000)
            self._sync_state()

        def reload(self):
            self.page.reload(wait_until="domcontentloaded", timeout=15000)
            self._sync_state()

        # ── Медиа ─────────────────────────────────────────────────────────

        def force_play(self):
            """Все методы подряд: JS → селекторы → клавиши."""
            # 1. Прямой JS на video-элемент
            try:
                self.page.evaluate(
                    "var v=document.querySelector('video');"
                    "if(v&&v.paused)v.play();"
                )
            except Exception:
                pass
            time.sleep(0.4)
            # 2. Кнопки play по распространённым селекторам
            for sel in [
                'button[aria-label*="play" i]',
                'button[title*="play" i]',
                '[data-testid*="play-button" i]',
                '[class*="PlayButton"]',
                '[class*="play-btn"]',
                '.ytp-play-button',      # YouTube
                '[aria-label="Play"]',
            ]:
                try:
                    if self.page.locator(sel).count() > 0:
                        self.page.locator(sel).first.click(timeout=1500)
                        break
                except Exception:
                    pass
            # 3. Клавиша k (YouTube/большинство видеоплееров)
            try:
                self.page.keyboard.press("k")
            except Exception:
                pass

        def pause(self):
            try:
                self.page.evaluate(
                    "var v=document.querySelector('video');if(v&&!v.paused)v.pause();"
                )
            except Exception:
                pass

        def fullscreen(self):
            """Fullscreen: сначала видео-элемент, потом F11 на весь браузер."""
            try:
                self.page.evaluate(
                    "var v=document.querySelector('video');"
                    "if(v){var fn=v.requestFullscreen||v.webkitRequestFullscreen;"
                    "if(fn)fn.call(v);"
                    "else{document.documentElement.requestFullscreen();}}"
                    "else{document.documentElement.requestFullscreen();}"
                )
            except Exception:
                pass
            time.sleep(0.3)
            try:
                self.page.keyboard.press("F11")
            except Exception:
                pass

        def _try_autoplay(self):
            """Тихо пробует включить воспроизведение после загрузки."""
            try:
                playing = self.page.evaluate(
                    "var v=document.querySelector('video');v?!v.paused:false"
                )
                if not playing:
                    self.page.evaluate(
                        "var v=document.querySelector('video');if(v)v.play();"
                    )
            except Exception:
                pass

        # ── Взаимодействие с UI ───────────────────────────────────────────

        def click(self, selector: str = None, x: int = None, y: int = None):
            if selector:
                self.page.locator(selector).first.click(timeout=5000)
            elif x is not None and y is not None:
                self.page.mouse.click(x, y)

        def type_text(self, text: str, selector: str = None):
            if selector:
                self.page.fill(selector, text)
            else:
                self.page.keyboard.type(text)

        def key(self, key: str):
            self.page.keyboard.press(key)

        def scroll(self, direction: str = "down", amount: int = 3):
            key = "PageDown" if direction == "down" else "PageUp"
            for _ in range(amount):
                self.page.keyboard.press(key)
                time.sleep(0.05)

        def dismiss_popups(self) -> bool:
            """Авто-закрыть cookie-баннеры, consent-диалоги, age-check."""
            for sel in [
                'button:has-text("Accept all")',
                'button:has-text("Accept All")',
                'button:has-text("Принять всё")',
                'button:has-text("Принять")',
                'button:has-text("Agree")',
                'button:has-text("I Agree")',
                'button:has-text("OK")',
                'button:has-text("Got it")',
                'button[id*="accept" i]',
                'button[class*="accept" i]',
                '#accept-all',
                '[data-testid*="accept" i]',
                '[aria-label*="accept" i]',
                '.cookie-accept',
                '.consent-accept',
            ]:
                try:
                    loc = self.page.locator(sel)
                    if loc.count() > 0:
                        loc.first.click(timeout=1200)
                        return True
                except Exception:
                    continue
            return False

        # ── Состояние и наблюдение ────────────────────────────────────────

        def get_dom_state(self) -> dict:
            """
            Читаем всё из DOM: URL, title, video state.
            Это основной источник правды — дешевле скриншота.
            """
            result = {}
            try:
                result["url"]     = self.page.url
                result["title"]   = self.page.title()
                result["playing"] = self.page.evaluate(
                    "var v=document.querySelector('video');v?!v.paused:false"
                )
                result["video_url"] = self.page.evaluate(
                    "var v=document.querySelector('video');v?(v.src||v.currentSrc):null"
                )
            except Exception:
                pass
            return result

        def capture(self) -> str:
            """
            Скриншот страницы (vision fallback).
            Playwright делает это нативно — не нужны внешние инструменты.
            """
            path = os.path.join(
                os.environ.get("TEMP", os.path.expanduser("~")),
                "deks_viver_capture.png"
            )
            try:
                self.page.screenshot(path=path, full_page=False)
            except Exception:
                pass
            return path

        def execute_js(self, code: str):
            return self.page.evaluate(code)

        def _sync_state(self):
            global _state
            try:
                dom = self.get_dom_state()
                if dom.get("url") and dom["url"] not in ("about:blank", ""):
                    _state["url"] = dom["url"]
                    if dom["url"] not in _state["history"]:
                        _state["history"].append(dom["url"])
                if dom.get("title"):
                    _state["title"] = dom["title"]
                if "playing" in dom:
                    _state["playing"] = dom["playing"]
            except Exception:
                pass

        def close(self):
            try:
                self.page.context.browser.close()
            except Exception:
                pass

    # ── Goal execution (plan → act → observe → retry) ─────────────────────

    def _execute_goal(ctrl: ViewerController, goal: str, url: str = "") -> dict:
        """
        Goal-based execution: агент пытается достичь цели, а не просто
        выполнить одну команду.
        """
        global _state
        _state["last_goal"] = goal
        goal_lower = goal.lower()

        if url:
            ctrl.open_url(url)

        # Наблюдаем: что реально произошло?
        dom = ctrl.get_dom_state()
        _state.update({k: v for k, v in dom.items() if k in _state})

        # Попытки достичь цели
        played = dom.get("playing", False)
        if not played:
            ctrl.force_play()
            time.sleep(1.0)
            dom = ctrl.get_dom_state()
            played = dom.get("playing", False)

        if "полный экран" in goal_lower or "весь экран" in goal_lower or "fullscreen" in goal_lower:
            ctrl.fullscreen()

        ctrl._sync_state()
        return _state

    # ── Command dispatch ───────────────────────────────────────────────────

    def _dispatch(action: str, body: dict, ctrl: ViewerController):
        global _state

        if action == "open_url":
            ctrl.open_url(body.get("url", ""))
            return _state

        elif action == "execute_goal":
            return _execute_goal(ctrl, body.get("goal", ""), body.get("url", ""))

        elif action in ("play", "force_play"):
            ctrl.force_play()

        elif action == "pause":
            ctrl.pause()

        elif action == "fullscreen":
            ctrl.fullscreen()

        elif action == "back":
            ctrl.back()

        elif action == "forward":
            ctrl.forward()

        elif action == "reload":
            ctrl.reload()

        elif action == "scroll":
            ctrl.scroll(body.get("direction", "down"), body.get("amount", 3))

        elif action == "click":
            ctrl.click(body.get("selector"), body.get("x"), body.get("y"))

        elif action == "type":
            ctrl.type_text(body.get("text", ""), body.get("selector"))

        elif action == "key":
            ctrl.key(body.get("key", ""))

        elif action == "dismiss_popups":
            return ctrl.dismiss_popups()

        elif action == "execute_js":
            return ctrl.execute_js(body.get("code", ""))

        elif action == "capture":
            return {"path": ctrl.capture()}

        elif action == "get_state":
            ctrl._sync_state()
            return _state

        elif action == "close":
            ctrl.close()
            return "closing"

        return "ok"

    # ── HTTP API ───────────────────────────────────────────────────────────

    def _enqueue(action: str, body: dict, timeout: int = 15) -> object:
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
        def log_message(self, *a): pass

        def do_GET(self):
            if self.path == "/ping":
                self.send_response(200); self.end_headers()
                self.wfile.write(b"ok")
            elif self.path == "/state":
                result = _enqueue("get_state", {})
                self._reply(200, json.dumps(result or _state).encode())
            else:
                self.send_response(404); self.end_headers()

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

    threading.Thread(target=_start_http, daemon=True).start()

    # ── Playwright main loop (ЕДИНСТВЕННЫЙ поток Playwright) ───────────────
    with sync_playwright() as pw:
        # Реальный Chrome для DRM (Netflix и т.д.)
        try:
            browser = pw.chromium.launch(
                headless=False,
                channel="chrome",
                args=["--start-maximized", "--disable-blink-features=AutomationControlled"],
            )
        except Exception:
            # Fallback: скачанный Chromium
            browser = pw.chromium.launch(
                headless=False,
                args=["--start-maximized"],
            )

        ctx  = browser.new_context(
            viewport=None,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        page = ctx.new_page()
        page.set_extra_http_headers({"Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8"})
        ctrl = ViewerController(page)

        # Основной цикл обработки команд
        running = True
        while running:
            try:
                cid, action, body = _cmd_q.get(timeout=0.15)
            except queue.Empty:
                if not browser.is_connected():
                    break
                continue

            try:
                result = _dispatch(action, body, ctrl)
            except Exception as ex:
                print(f"[VIVER RUNTIME] Error in {action}: {ex}")
                result = {"error": str(ex)}

            with _result_lck:
                _result_map[cid] = result

            if action == "close" or result == "closing":
                running = False

        try:
            browser.close()
        except Exception:
            pass

    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
#  SKILL MODE — импортируется DEKS
# ══════════════════════════════════════════════════════════════════════════════

from skills.base_skill import BaseSkill

_TRIGGERS_PLAY = [
    "плей", "нажми плей", "запусти видео", "включи видео",
    "воспроизвести", "воспроизведи", "play",
]
_TRIGGERS_PAUSE = [
    "пауза", "поставь на паузу", "останови видео",
]
_TRIGGERS_FULLSCREEN = [
    "полный экран", "на весь экран", "разверни на весь", "fullscreen",
    "развернуть", "разверни экран",
]
_TRIGGERS_BACK    = ["назад в вивере", "вернись в вивере", "предыдущая страница"]
_TRIGGERS_FORWARD = ["вперёд в вивере", "вперед в вивере"]
_TRIGGERS_WRONG   = [
    "ты открыл не то", "открыл не то", "не тот фильм", "не та страница",
    "не то видео", "неправильно открыл", "что сейчас открыто",
    "что в вивере", "что ты сейчас показываешь", "не тот сайт",
    "не та музыка", "не та песня", "что открыто",
]
_TRIGGERS_CLOSE = [
    "закрой вивер", "выключи вивер", "убери вивер",
]

_ALL_TRIGGERS = (
    _TRIGGERS_PLAY + _TRIGGERS_PAUSE + _TRIGGERS_FULLSCREEN +
    _TRIGGERS_BACK + _TRIGGERS_FORWARD + _TRIGGERS_WRONG + _TRIGGERS_CLOSE
)


class DeksViverSkill(BaseSkill):

    def __init__(self, app, name="deks_viver"):
        global _viver_skill_instance
        super().__init__(app, name)
        self._proc = None
        _viver_skill_instance = self
        print("[VIVER] Skill loaded — all web content will open in DEKS Viver")

    # ── Обработчик голосовых команд ───────────────────────────────────────

    def handle(self, command: str) -> str | None:
        cmd = command.lower().strip()
        if not any(t in cmd for t in _ALL_TRIGGERS):
            return None

        if any(t in cmd for t in _TRIGGERS_CLOSE):
            self._send("close")
            return "Закрываю Viver."

        if any(t in cmd for t in _TRIGGERS_FULLSCREEN):
            if not self.is_alive():
                return "Viver не открыт."
            self._send("fullscreen")
            return "На весь экран."

        if any(t in cmd for t in _TRIGGERS_PLAY):
            if not self.is_alive():
                return "Viver не открыт — скажи что включить."
            self._send("force_play")
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

        if any(t in cmd for t in _TRIGGERS_FORWARD):
            if not self.is_alive():
                return "Viver не открыт."
            self._send("forward")
            return "Вперёд."

        if any(t in cmd for t in _TRIGGERS_WRONG):
            threading.Thread(target=self._handle_wrong, daemon=True).start()
            return ""

        return None

    # ── Запуск процесса ───────────────────────────────────────────────────

    def _launch(self) -> bool:
        if self.is_alive():
            return True

        # Гарантируем что playwright браузеры установлены
        try:
            subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                capture_output=True, timeout=120,
            )
        except Exception:
            pass

        script = os.path.abspath(__file__)
        try:
            self._proc = subprocess.Popen(
                [sys.executable, script, "--runtime"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as ex:
            print(f"[VIVER] Launch failed: {ex}")
            return False

        for _ in range(40):   # ждём до 8 сек
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
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read().decode())
        except Exception as ex:
            print(f"[VIVER] Send error ({action}): {ex}")
            return None

    def get_state(self) -> dict:
        try:
            req = urllib.request.Request(f"{VIVER_URL}/state", method="GET")
            with urllib.request.urlopen(req, timeout=5) as r:
                return json.loads(r.read().decode())
        except Exception:
            return {}

    def is_alive(self) -> bool:
        try:
            req = urllib.request.Request(f"{VIVER_URL}/ping", method="GET")
            with urllib.request.urlopen(req, timeout=1) as r:
                return r.read() == b"ok"
        except Exception:
            return False

    def open_url(self, url: str) -> bool:
        result = self._send("open_url", url=url)
        return result is not None

    def capture(self) -> str | None:
        """Скриншот текущего состояния вивера для vision."""
        result = self._send("capture")
        if result and isinstance(result.get("result"), dict):
            return result["result"].get("path")
        return None

    # ── Коррекция ошибки ──────────────────────────────────────────────────

    def _handle_wrong(self):
        if not self.is_alive():
            self.app.after(0, lambda: self.app.deks_say("Viver не открыт. Скажи что нужно."))
            return

        state  = self.get_state()
        title  = state.get("title", "")
        url    = state.get("url",   "")
        shown  = title if (title and title != url) else url

        msg = (
            f"Сейчас открыто: {shown}. Что показать вместо этого?"
            if shown else
            "Viver пустой. Что нужно открыть?"
        )
        self.app.after(0, lambda m=msg: self.app.deks_say(m))


# ══════════════════════════════════════════════════════════════════════════════
#  Публичный хелпер — вызывается из ollama_mixin / commands.py
# ══════════════════════════════════════════════════════════════════════════════

def try_open_in_viver(url: str) -> bool:
    """
    Если скилл загружен — открывает URL в Viver (авто-запуск при необходимости).
    Возвращает True при успехе, False если скилл не установлен.
    """
    global _viver_skill_instance
    if _viver_skill_instance is None:
        return False

    if not _viver_skill_instance.is_alive():
        launched = _viver_skill_instance._launch()
        if not launched:
            print("[VIVER] Auto-launch failed — falling back to browser")
            return False

    return _viver_skill_instance.open_url(url)
