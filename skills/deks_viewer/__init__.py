"""
DEKS Viewer — клиент.

Запускает viewer/server.py как subprocess и управляет им через HTTP.

Использование:
    from viewer import viewer_client as viewer
    viewer.show_web("https://youtube.com")
    viewer.show_image("https://...")
    viewer.show_video("https://youtu.be/abc123")
    viewer.close()
"""

import os
import sys
import json
import subprocess
import time
import urllib.request
import urllib.error

VIEWER_PORT = 7548
_BASE = f"http://127.0.0.1:{VIEWER_PORT}"

_proc: subprocess.Popen | None = None


def _is_running() -> bool:
    try:
        urllib.request.urlopen(f"{_BASE}/ping", timeout=1)
        return True
    except Exception:
        return False


def _ensure_started(hidden: bool = False) -> bool:
    global _proc

    if _is_running():
        return True

    # Если старый процесс завис — убиваем
    if _proc is not None:
        try:
            _proc.terminate()
        except Exception:
            pass
        _proc = None

    server_path = os.path.join(os.path.dirname(__file__), "server.py")
    args = [sys.executable, server_path]
    if hidden:
        args.append("--hidden")

    _proc = subprocess.Popen(
        args,
        creationflags=0x00000008,  # DETACHED_PROCESS — без консоли
    )

    # Ждём до 8 секунд (Edge WebView2 стартует медленно)
    for _ in range(80):
        time.sleep(0.1)
        if _is_running():
            print(f"[VIEWER] Process started {'(hidden)' if hidden else ''}")
            return True

    print("[VIEWER] Failed to start viewer process")
    return False


def warm_up() -> None:
    """Прогрев viewer при старте DEKS — запускает скрытым в фоне.
    Первый реальный вызов show_web/show_video откроет окно мгновенно."""
    import threading as _th
    def _do():
        _ensure_started(hidden=True)
    _th.Thread(target=_do, daemon=True).start()


def _send(action: str, **kwargs) -> bool:
    if not _ensure_started():
        return False
    try:
        body = {"action": action, **kwargs}
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{_BASE}/",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        print(f"[VIEWER] Send error: {e}")
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def show_web(url: str) -> bool:
    """Открыть сайт в Viewer."""
    return _send("show_web", url=url)


def show_image(url: str) -> bool:
    """Показать картинку по URL."""
    return _send("show_image", url=url)


def show_video(url: str) -> bool:
    """Открыть видео (YouTube URL или прямая ссылка)."""
    return _send("show_video", url=url)


def show_text(text: str) -> bool:
    """Показать текст/код в Viewer."""
    return _send("show_text", text=text)


def fullscreen() -> bool:
    """Переключить полноэкранный режим."""
    return _send("fullscreen")


def hide() -> bool:
    """Скрыть окно Viewer (процесс продолжает работать)."""
    return _send("hide")


def show() -> bool:
    """Показать ранее скрытое окно."""
    return _send("show")


def idle() -> bool:
    """Вернуть Viewer в idle-состояние (пустой экран)."""
    return _send("idle")


def close() -> bool:
    """Закрыть Viewer полностью."""
    global _proc
    _send("close")
    if _proc is not None:
        try:
            _proc.terminate()
        except Exception:
            pass
        _proc = None
    return True


def is_running() -> bool:
    return _is_running()
