"""
eyes.py — навык зрения DEKS.
Анализирует экран через Groq Vision (llama-4-scout).
Быстро (~1-2с), бесплатно, работает без VPN.
По умолчанию использует ключ Groq из настроек DEKS.
"""

import threading
import urllib.parse
import http.client
import json
import base64
from io import BytesIO

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from skills.base_skill import BaseSkill

GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class EyesSkill(BaseSkill):

    def __init__(self, app, name):
        super().__init__(app, name)
        self._last_img = None
        self._last_n_monitors = 1
        self._context_active = False
        self._lock = threading.Lock()

    # ── Контракт маркета ─────────────────────────────────────────────────────

    def get_settings(self) -> list:
        return [
            {
                "key": "groq_key",
                "label": "Groq API ключ (необязательно)",
                "placeholder": "gsk_... (если пусто — используется ключ DEKS)",
                "secret": True,
            }
        ]

    def get_data_files(self) -> list:
        return []

    def is_configured(self) -> bool:
        return True  # Работает с ключом из DEKS или своим

    # ── handle ───────────────────────────────────────────────────────────────

    def handle(self, command: str) -> str | None:
        cmd = command.lower().strip()

        if self._context_active and self._last_img is not None:
            if self.is_hit(cmd, "eyes_clarify"):
                threading.Thread(
                    target=self._async_vision, args=(cmd, True), daemon=True
                ).start()
                return ""  # молчаливый захват

        if self.is_hit(cmd, "eyes_look"):
            threading.Thread(
                target=self._async_vision, args=(cmd, False), daemon=True
            ).start()
            return ""  # молчаливый захват

        if self._context_active:
            self._clear_context()

        return None

    def _async_vision(self, user_text: str, reuse: bool):
        result = self._ask_vision(user_text, reuse)
        if result:
            self.app.after(0, lambda r=result: self.app.deks_say(r))

    # ── Скриншот ─────────────────────────────────────────────────────────────

    def _take_screenshot(self):
        if not MSS_AVAILABLE or not PIL_AVAILABLE:
            return None, 0
        try:
            with mss.mss() as sct:
                n = len(sct.monitors) - 1
                raw = sct.grab(sct.monitors[0])
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                img.thumbnail((1920, 600), Image.LANCZOS)
                return img, n
        except Exception as e:
            self._log(f"[Eyes] Ошибка скриншота: {e}")
            return None, 0

    # ── Vision ───────────────────────────────────────────────────────────────

    def _ask_vision(self, user_text: str, reuse: bool) -> str:
        if not MSS_AVAILABLE:
            return "Установите mss: pip install mss"
        if not PIL_AVAILABLE:
            return "Установите Pillow: pip install Pillow"

        if reuse and self._last_img is not None:
            img, n = self._last_img, self._last_n_monitors
        else:
            img, n = self._take_screenshot()
            if img is None:
                return "Не удалось сделать скриншот."
            with self._lock:
                self._last_img = img
                self._last_n_monitors = n
                self._context_active = True

        self._log(f"[Eyes] Скриншот готов, мониторов: {n}")

        # Конвертируем в base64
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # Промпт с учётом числа мониторов
        if n == 1:
            vision_prompt = (
                "Опиши кратко что видишь на скриншоте: "
                "какие приложения открыты, что происходит. "
                "Только факты. На русском. Максимум 3 предложения."
            )
        else:
            vision_prompt = (
                f"На картинке {n} монитора слева направо. "
                f"Опиши кратко что на каждом: приложения, контент. "
                f"Только факты. На русском. Максимум 2 предложения на монитор."
            )

        # Получаем описание от Groq Vision
        screen_desc = self._call_groq_vision(img_b64, vision_prompt)
        if not screen_desc:
            return "Не удалось проанализировать экран."

        self._log("[Eyes] Описание получено")

        # Передаём описание основному LLM для финального ответа
        return self._ask_main_llm(user_text, screen_desc, n)

    def _call_groq_vision(self, img_b64: str, prompt: str) -> str:
        api_key = self._get_api_key()
        if not api_key:
            return ""

        payload = {
            "model": GROQ_MODEL,
            "max_tokens": 400,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}"
                }}
            ]}]
        }

        try:
            body = json.dumps(payload).encode("utf-8")
            conn = http.client.HTTPSConnection("api.groq.com", timeout=20)
            conn.request("POST", "/openai/v1/chat/completions", body=body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            })
            resp = conn.getresponse()
            data = json.loads(resp.read().decode("utf-8"))
            conn.close()
            if resp.status != 200:
                self._log(f"[Eyes] Groq Vision ERR {resp.status}: {data}")
                return ""
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            self._log(f"[Eyes] Groq Vision exception: {e}")
            return ""

    def _ask_main_llm(self, user_question: str, screen_desc: str, n: int) -> str:
        api_key = self._get_api_key()
        if not api_key:
            return screen_desc

        n_word = (
            f"{n} монитор{'а' if 2 <= n <= 4 else 'ов' if n > 4 else ''}"
        )
        prompt = (
            f"Пользователь спросил: \"{user_question}\"\n\n"
            f"На экране ({n_word}):\n{screen_desc}\n\n"
            f"Ответь кратко и по делу. "
            f"Только русский язык. Максимум 2-3 предложения."
        )

        # Берём модель из настроек DEKS (может быть не Groq)
        try:
            from config import load_api_keys
            keys = load_api_keys()
            ctrl_url = keys.get("controller_url", GROQ_API_URL)
            ctrl_model = keys.get("controller_model", "llama-3.3-70b-versatile")
            ctrl_key = keys.get("controller_key", api_key)
        except Exception:
            ctrl_url = GROQ_API_URL
            ctrl_model = "llama-3.3-70b-versatile"
            ctrl_key = api_key

        payload = {
            "model": ctrl_model,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            body = json.dumps(payload).encode("utf-8")
            parsed = urllib.parse.urlparse(ctrl_url)
            conn = http.client.HTTPSConnection(parsed.netloc, timeout=15)
            conn.request("POST", parsed.path, body=body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {ctrl_key}",
            })
            resp = conn.getresponse()
            data = json.loads(resp.read().decode("utf-8"))
            conn.close()
            if resp.status != 200:
                return screen_desc
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            self._log(f"[Eyes] LLM exception: {e}")
            return screen_desc

    # ── Утилиты ──────────────────────────────────────────────────────────────

    def _get_api_key(self) -> str:
        """Свой ключ из настроек навыка → иначе ключ DEKS (Groq)."""
        own_key = self.load_setting("groq_key", "").strip()
        if own_key:
            return own_key
        try:
            from config import load_api_keys
            return load_api_keys().get("controller_key", "")
        except Exception:
            return ""

    def _clear_context(self):
        with self._lock:
            self._last_img = None
            self._last_n_monitors = 1
            self._context_active = False

    def _log(self, msg: str):
        try:
            import logging
            logging.getLogger("DEKS").info(msg)
        except Exception:
            pass
