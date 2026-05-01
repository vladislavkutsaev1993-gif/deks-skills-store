"""
DEKS Viewer — standalone window process.

Запускается как subprocess из viewer/__init__.py.
Поднимает HTTP API на localhost:7548.
pywebview (Edge WebView2) ТОЛЬКО на главном потоке — команды через очередь.

Архитектура:
- Для web/video: window.load_url() напрямую (нет iframe → нет X-Frame-Options блокировок)
- Для image/text: window.load_url(viewer.html) + evaluate_js()
- Для перехода обратно в idle: возврат на viewer.html
"""

import os
import sys
import json
import re
import time
import queue
import threading

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from http.server import HTTPServer, BaseHTTPRequestHandler

VIEWER_PORT = 7548
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_HTML_PATH = os.path.join(_THIS_DIR, "viewer.html")
_HTML_URL  = "file:///" + _HTML_PATH.replace("\\", "/")

_cmd_queue: queue.Queue = queue.Queue()
_window_ready = threading.Event()
_window = None

# Текущий режим — чтобы знать нужно ли сначала перейти на viewer.html
_current_mode = "idle"   # idle | html | web


# ── HTTP API ──────────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n)) if n else {}
            _cmd_queue.put(body)
            self._reply(200, json.dumps({"ok": True}).encode())
        except Exception as ex:
            self._reply(500, json.dumps({"ok": False, "error": str(ex)}).encode())

    def _reply(self, code: int, data: bytes):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _start_http():
    HTTPServer(("127.0.0.1", VIEWER_PORT), _Handler).serve_forever()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_youtube_embed(url: str) -> str:
    """YouTube URL → embed URL с autoplay."""
    m = re.search(
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        url
    )
    if m:
        vid_id = m.group(1)
        return f"https://www.youtube.com/embed/{vid_id}?autoplay=1&rel=0"
    return url


def _is_youtube(url: str) -> bool:
    return bool(re.search(r'youtube\.com|youtu\.be', url))


def _wait_for_load(timeout: float = 3.0):
    """Ждём пока страница загрузится (события loaded через Event)."""
    _window_ready.clear()
    _window_ready.wait(timeout=timeout)


# ── Command processing (main thread tick) ─────────────────────────────────────

def _process_commands():
    global _window, _current_mode

    if _window is None:
        return

    while not _cmd_queue.empty():
        try:
            body = _cmd_queue.get_nowait()
        except queue.Empty:
            break

        action = body.get("action", "")

        if action == "show_web":
            url = body.get("url", "")
            _window.show()
            _window.load_url(url)
            _current_mode = "web"

        elif action == "show_video":
            url = body.get("url", "")
            target = _to_youtube_embed(url) if _is_youtube(url) else url
            _window.show()
            _window.load_url(target)
            _current_mode = "web"

        elif action == "show_image":
            url = body.get("url", "")
            _window.show()
            if _current_mode == "web":
                _window.load_url(_HTML_URL)
                time.sleep(1.0)
            _window.evaluate_js(f"showImage({json.dumps(url)})")
            _current_mode = "html"

        elif action == "show_text":
            text = body.get("text", "")
            _window.show()
            if _current_mode == "web":
                _window.load_url(_HTML_URL)
                time.sleep(1.0)
            _window.evaluate_js(f"showText({json.dumps(text)})")
            _current_mode = "html"

        elif action == "idle":
            _window.load_url(_HTML_URL)
            _current_mode = "idle"

        elif action == "fullscreen":
            _window.toggle_fullscreen()

        elif action == "hide":
            _window.hide()

        elif action == "show":
            _window.show()

        elif action == "click_text":
            text = body.get("text", "").lower()
            js = f"""
(function() {{
    var needle = {json.dumps(text)};

    function norm(s) {{
        return (s || '').toLowerCase().replace(/\\s+/g, ' ').trim();
    }}

    function fireClick(el) {{
        el.dispatchEvent(new MouseEvent('mouseover', {{bubbles: true}}));
        el.dispatchEvent(new MouseEvent('mousedown', {{bubbles: true, cancelable: true}}));
        el.dispatchEvent(new MouseEvent('mouseup',   {{bubbles: true, cancelable: true}}));
        el.dispatchEvent(new MouseEvent('click',     {{bubbles: true, cancelable: true, view: window}}));
        if (el.tagName === 'A' && el.href) {{
            window.location.href = el.href;
        }}
    }}

    // Проход 1 — ссылки и кнопки (самое надёжное)
    var links = Array.from(document.querySelectorAll('a[href], button'));
    for (var el of links) {{
        if (!el.offsetParent) continue;
        if (norm(el.textContent).includes(needle)) {{
            fireClick(el);
            return true;
        }}
    }}

    // Проход 2 — любой видимый элемент с подходящим текстом
    var all = Array.from(document.querySelectorAll('*'));
    for (var el of all) {{
        if (!el.offsetParent) continue;
        var tag = el.tagName;
        if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'HTML' || tag === 'BODY' || tag === 'HEAD') continue;
        var txt = norm(el.textContent);
        if (txt.includes(needle) && txt.length < needle.length * 15) {{
            var target = el.closest('a[href]') || el;
            fireClick(target);
            return true;
        }}
    }}

    return false;
}})()
"""
            _window.evaluate_js(js)

        elif action == "scroll":
            direction = body.get("direction", "down")
            amount = int(body.get("amount", 500))
            if direction == "up":
                amount = -amount
            _window.evaluate_js(f"window.scrollBy({{top: {amount}, behavior: 'smooth'}})")

        elif action == "back":
            _window.evaluate_js("window.history.back()")

        elif action == "forward":
            _window.evaluate_js("window.history.forward()")

        elif action == "close":
            _window.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import webview

    threading.Thread(target=_start_http, daemon=True).start()
    print(f"[VIEWER] HTTP API on localhost:{VIEWER_PORT}")

    _window = webview.create_window(
        title="DEKS Viewer",
        url=_HTML_URL,
        width=900,
        height=600,
        resizable=True,
        frameless=False,
        easy_drag=False,
        background_color="#0d0d0d",
        on_top=False,
        min_size=(400, 300),
    )

    _start_hidden = "--hidden" in sys.argv
    _first_load_done = [False]  # список чтобы изменять из вложенной функции

    def _on_loaded():
        _window_ready.set()
        # Скрываем ТОЛЬКО при первой загрузке (прогрев)
        if _start_hidden and not _first_load_done[0]:
            _first_load_done[0] = True
            _window.hide()

    _window.events.loaded += _on_loaded

    def _tick():
        while True:
            if _window_ready.is_set():
                _process_commands()
            time.sleep(0.08)

    threading.Thread(target=_tick, daemon=True).start()

    # Пробуем запустить с Edge WebView2 (поддерживает YouTube, современный JS)
    try:
        webview.start(gui="edgechromium", debug=False, private_mode=False)
    except Exception:
        # Fallback на дефолтный движок
        webview.start(debug=False, private_mode=False)

    print("[VIEWER] Closed.")
