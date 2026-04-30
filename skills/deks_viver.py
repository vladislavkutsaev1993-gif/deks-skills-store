"""
DEKS Viver — интерактивное медиа-окно DEKS.
Устанавливается через магазин навыков.

Если установлен — ВСЁ интернет-содержимое открывается через него.
Авто-запускается при первом запросе. Может видеть что открыл.

Два режима одного файла:
  - Skill mode  (импортируется DEKS)
  - Runtime mode (subprocess с флагом --runtime)
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

# Синглтон — ссылка на живой экземпляр скилла (устанавливается в __init__)
_viver_skill_instance = None


# ══════════════════════════════════════════════════════════════════════════════
#  RUNTIME — отдельный subprocess: python deks_viver.py --runtime
# ══════════════════════════════════════════════════════════════════════════════

if "--runtime" in sys.argv:
    import webview
    from http.server import HTTPServer, BaseHTTPRequestHandler

    _window = None
    _state  = {
        "url":        "",
        "title":      "",
        "fullscreen": False,
        "history":    [],
        "playing":    False,
    }

    # JS-скрипт который инжектируется при каждой загрузке страницы
    _AUTO_INJECT_JS = """
    (function() {
        // Авто-плей видео на YouTube / Netflix / любом сайте
        function tryPlay() {
            var v = document.querySelector('video');
            if (v && v.paused && v.readyState >= 2) { v.play(); }
        }
        // Небольшая задержка — страница должна отрисоваться
        setTimeout(tryPlay, 1500);
        setTimeout(tryPlay, 3000);
    })();
    """

    def _update_state():
        try:
            url   = _window.evaluate_js("location.href")
            title = _window.evaluate_js("document.title")
            playing = _window.evaluate_js(
                "var v=document.querySelector('video'); v ? !v.paused : false"
            )
            if url and url not in ("about:blank", "null", None):
                _state["url"] = url
                if url not in _state["history"]:
                    _state["history"].append(url)
            if title and title not in ("", "null", None):
                _state["title"] = title
            if isinstance(playing, bool):
                _state["playing"] = playing
        except Exception:
            pass

    def _handle_command(action: str, body: dict) -> object:
        global _state
        if _window is None:
            return "no_window"

        if action == "open_url":
            url = body.get("url", "")
            if url:
                _window.load_url(url)
                _state["url"]   = url
                _state["title"] = url
                if url not in _state["history"]:
                    _state["history"].append(url)

        elif action == "play":
            _window.evaluate_js(
                "var v=document.querySelector('video');"
                "if(v){if(v.paused)v.play();"
                "else{v.pause();}}"  # toggle
            )

        elif action == "force_play":
            _window.evaluate_js(
                "var v=document.querySelector('video');if(v)v.play();"
            )

        elif action == "pause":
            _window.evaluate_js(
                "var v=document.querySelector('video');if(v)v.pause();"
            )

        elif action == "fullscreen":
            # Попробуем и через pywebview и через JS
            _state["fullscreen"] = not _state.get("fullscreen", False)
            try:
                _window.toggle_fullscreen()
            except Exception:
                pass
            _window.evaluate_js(
                "var v=document.querySelector('video');"
                "if(v){var fn=v.requestFullscreen||v.webkitRequestFullscreen||v.mozRequestFullScreen;"
                "if(fn)fn.call(v);}"
            )

        elif action == "fullscreen_page":
            # Полностраничный fullscreen (не только видео)
            _state["fullscreen"] = not _state.get("fullscreen", False)
            try:
                _window.toggle_fullscreen()
            except Exception:
                pass

        elif action == "back":
            _window.evaluate_js("history.back()")

        elif action == "forward":
            _window.evaluate_js("history.forward()")

        elif action == "reload":
            _window.evaluate_js("location.reload()")

        elif action == "scroll_down":
            _window.evaluate_js("window.scrollBy(0, window.innerHeight * 0.8)")

        elif action == "scroll_up":
            _window.evaluate_js("window.scrollBy(0, -window.innerHeight * 0.8)")

        elif action == "execute_js":
            code = body.get("code", "")
            if code:
                return _window.evaluate_js(code)

        elif action == "click_selector":
            selector = body.get("selector", "")
            if selector:
                _window.evaluate_js(
                    f"var el=document.querySelector('{selector}');if(el)el.click();"
                )

        elif action == "get_state":
            _update_state()
            return _state

        elif action == "capture":
            # Делаем скриншот содержимого через Canvas API
            try:
                b64 = _window.evaluate_js("""
                (function() {
                    try {
                        var c = document.createElement('canvas');
                        c.width = window.innerWidth;
                        c.height = window.innerHeight;
                        var ctx = c.getContext('2d');
                        // Базовый скриншот через html2canvas если доступен
                        if(window.html2canvas) {
                            html2canvas(document.body).then(function(canvas) {
                                window._viver_capture = canvas.toDataURL('image/png');
                            });
                            return 'async';
                        }
                        return 'no_html2canvas';
                    } catch(e) { return 'error:' + e.message; }
                })();
                """)
                return {"capture": b64}
            except Exception as ex:
                return {"error": str(ex)}

        elif action == "close":
            try:
                _window.destroy()
            except Exception:
                pass

        return "ok"

    def _on_loaded(window):
        _update_state()
        # Инжектируем авто-плей
        try:
            window.evaluate_js(_AUTO_INJECT_JS)
        except Exception:
            pass

    class _ViverHandler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass

        def do_GET(self):
            if self.path == "/ping":
                self.send_response(200); self.end_headers()
                self.wfile.write(b"ok")
            elif self.path == "/state":
                _update_state()
                body = json.dumps(_state).encode()
                self._reply(200, body)
            else:
                self.send_response(404); self.end_headers()

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                body   = json.loads(self.rfile.read(length)) if length else {}
                action = body.get("action", "")
                result = _handle_command(action, body)
                resp   = json.dumps({"ok": True, "result": result or "ok"}).encode()
                self._reply(200, resp)
            except Exception as ex:
                err = json.dumps({"ok": False, "error": str(ex)}).encode()
                self._reply(500, err)

        def _reply(self, code, data: bytes):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

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
#  SKILL MODE — импортируется в DEKS
# ══════════════════════════════════════════════════════════════════════════════

from skills.base_skill import BaseSkill

_TRIGGERS_PLAY = [
    "плей", "нажми плей", "нажми play", "запусти видео",
    "включи видео", "воспроизвести", "воспроизведи",
]
_TRIGGERS_PAUSE = [
    "пауза", "поставь на паузу", "останови видео", "стоп видео",
]
_TRIGGERS_FULLSCREEN = [
    "полный экран", "на весь экран", "разверни на весь", "fullscreen",
    "развернуть", "разверни экран",
]
_TRIGGERS_BACK = [
    "назад в вивере", "вернись в вивере", "предыдущая страница",
    "вернись назад",
]
_TRIGGERS_FORWARD = ["вперёд в вивере", "вперед в вивере"]
_TRIGGERS_WRONG = [
    "ты открыл не то", "открыл не то", "не тот фильм", "не та страница",
    "не то видео", "неправильно открыл", "что сейчас открыто",
    "что в вивере", "что ты сейчас показываешь", "что открыто",
    "не тот сайт", "не та музыка", "не та песня",
]
_TRIGGERS_CLOSE = [
    "закрой вивер", "закрой интерактивное", "выключи вивер", "убери вивер",
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
        _viver_skill_instance = self   # регистрируем синглтон
        print("[VIVER] Skill loaded — all web content will open in DEKS Viver")

    # ── Основной обработчик ───────────────────────────────────────────────

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
        """Запустить viver если не запущен. Возвращает True если успешно."""
        if self.is_alive():
            return True
        script = os.path.abspath(__file__)
        try:
            self._proc = subprocess.Popen(
                [sys.executable, script, "--runtime"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"[VIVER] Launch error: {e}")
            return False

        for _ in range(35):      # ждём до 7 сек
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
        """Текущее состояние вивера: URL, title, playing, fullscreen."""
        try:
            self._send("get_state")
            req = urllib.request.Request(f"{VIVER_URL}/state", method="GET")
            with urllib.request.urlopen(req, timeout=2) as r:
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
        """Открыть URL в вивере (вызывается из try_open_in_viver)."""
        result = self._send("open_url", url=url)
        return result is not None

    def capture(self) -> str | None:
        """
        Сделать скриншот вивера для анализа.
        Возвращает путь к PNG-файлу.
        """
        try:
            import mss
            import mss.tools
            # Найти окно по заголовку через ctypes (Windows)
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = user32.FindWindowW(None, "DEKS Viver")
            if hwnd:
                rect = ctypes.wintypes.RECT()
                user32.GetWindowRect(hwnd, ctypes.byref(rect))
                x, y = rect.left, rect.top
                w = rect.right  - rect.left
                h = rect.bottom - rect.top
                with mss.mss() as sct:
                    region = {"left": x, "top": y, "width": w, "height": h}
                    img    = sct.grab(region)
                    path   = os.path.join(
                        os.environ.get("TEMP", os.path.expanduser("~")),
                        "deks_viver_capture.png"
                    )
                    mss.tools.to_png(img.rgb, img.size, output=path)
                    return path
        except Exception as e:
            print(f"[VIVER] Capture error: {e}")
        return None

    # ── Коррекция ошибки ──────────────────────────────────────────────────

    def _handle_wrong(self):
        """'Ты открыл не то' — смотрим что в вивере → сообщаем → ждём уточнения."""
        if not self.is_alive():
            self.app.after(0, lambda: self.app.deks_say(
                "Viver не открыт. Скажи что нужно."
            ))
            return

        state  = self.get_state()
        title  = state.get("title", "")
        url    = state.get("url",   "")
        shown  = title if (title and title != url) else url

        if shown:
            msg = f"Сейчас открыто: {shown}. Что показать вместо этого?"
        else:
            msg = "Viver пустой. Что нужно открыть?"

        self.app.after(0, lambda m=msg: self.app.deks_say(m))


# ══════════════════════════════════════════════════════════════════════════════
#  Публичная функция для ollama_mixin / llm_runtime_patch / commands.py
# ══════════════════════════════════════════════════════════════════════════════

def try_open_in_viver(url: str) -> bool:
    """
    Если скилл установлен — открывает URL в вивере.
    Авто-запускает вивер если он не запущен.
    Возвращает True при успехе, False если скилл не активен.
    """
    global _viver_skill_instance
    if _viver_skill_instance is None:
        return False   # скилл не загружен

    # Авто-запуск если вивер не работает
    if not _viver_skill_instance.is_alive():
        launched = _viver_skill_instance._launch()
        if not launched:
            print("[VIVER] Auto-launch failed — falling back to browser")
            return False

    return _viver_skill_instance.open_url(url)
