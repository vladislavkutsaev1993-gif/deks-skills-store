"""
DEKS Viver — тонкий skill-bridge.

Как советовал GPT: этот файл должен быть ОЧЕНЬ тонким.
Вся логика — в пакете deks_viver/.

Этот файл только:
  1. Определяет DeksViverSkill (голосовые команды)
  2. Запускает runtime (deks_viver/server.py)
  3. Предоставляет try_open_in_viver() для ollama_mixin
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

_viver_skill_instance = None   # синглтон


# ══════════════════════════════════════════════════════════════════════════════
#  SKILL MODE
# ══════════════════════════════════════════════════════════════════════════════

from skills.base_skill import BaseSkill

_TRIGGERS_PLAY = [
    "плей", "нажми плей", "запусти видео", "включи видео",
    "воспроизвести", "воспроизведи", "play",
]
_TRIGGERS_PAUSE     = ["пауза", "поставь на паузу", "останови видео"]
_TRIGGERS_FULLSCREEN = [
    "полный экран", "на весь экран", "разверни на весь", "fullscreen",
    "развернуть", "разверни экран",
]
_TRIGGERS_BACK    = ["назад в вивере", "вернись в вивере"]
_TRIGGERS_FORWARD = ["вперёд в вивере", "вперед в вивере"]
_TRIGGERS_WRONG   = [
    "ты открыл не то", "открыл не то", "не тот фильм", "не та страница",
    "не то видео", "неправильно открыл", "что сейчас открыто",
    "что в вивере", "что ты сейчас показываешь", "не тот сайт",
    "не та музыка", "не та песня", "что открыто",
]
_TRIGGERS_CLOSE   = ["закрой вивер", "выключи вивер", "убери вивер"]

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

    # ── Голосовые команды ─────────────────────────────────────────────────

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

    # ── Запуск runtime ────────────────────────────────────────────────────

    def _launch(self) -> bool:
        if self.is_alive():
            return True

        # Установить playwright-браузеры если нужно
        try:
            subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                capture_output=True, timeout=120,
            )
        except Exception:
            pass

        # Путь к server.py
        server_script = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "deks_viver", "server.py"
        )
        if not os.path.exists(server_script):
            print(f"[VIVER] server.py not found: {server_script}")
            return False

        try:
            self._proc = subprocess.Popen(
                [sys.executable, server_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as ex:
            print(f"[VIVER] Launch failed: {ex}")
            return False

        for _ in range(40):    # ждём до 8 сек
            if self.is_alive():
                return True
            time.sleep(0.2)

        return False

    # ── HTTP-клиент ───────────────────────────────────────────────────────

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
        result = self._send("capture")
        if result and isinstance(result.get("result"), dict):
            return result["result"].get("path")
        return None

    # ── Коррекция ошибки ("ты открыл не то") ─────────────────────────────

    def _handle_wrong(self):
        if not self.is_alive():
            self.app.after(0, lambda: self.app.deks_say("Viver не открыт. Скажи что нужно."))
            return

        state = self.get_state()
        title = state.get("title", "")
        url   = state.get("url",   "")
        shown = title if (title and title != url) else url

        msg = (
            f"Сейчас открыто: {shown}. Что показать вместо этого?"
            if shown else
            "Viver пустой. Что нужно открыть?"
        )
        self.app.after(0, lambda m=msg: self.app.deks_say(m))


# ══════════════════════════════════════════════════════════════════════════════
#  Публичный хелпер для ollama_mixin / commands.py
# ══════════════════════════════════════════════════════════════════════════════

def try_open_in_viver(url: str) -> bool:
    """
    Если скилл загружен — открыть URL в Viver.
    Авто-запуск если не запущен. Возвращает True при успехе.
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
