"""
eyes.py — навык зрения DEKS.
Анализирует экран через Vision AI (Groq / OpenAI-compatible).
Скриншот хранится только в RAM, на диск не пишется.

Конфиг: DEKS_DATA/eyes_config.json
Получить API ключ Groq бесплатно: https://console.groq.com/keys
"""

import json
import base64
import threading
import http.client
import urllib.parse
from io import BytesIO
from pathlib import Path

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


CONFIG_FILENAME = "eyes_config.json"

DEFAULT_CONFIG = {
    "api_key": "",
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    "api_url": "https://api.groq.com/openai/v1/chat/completions"
}

VISION_SYSTEM_PROMPT = (
    "Ты — зрительный модуль голосового ассистента DEKS. "
    "Тебе показывают скриншот экрана пользователя. "
    "Опиши кратко и по делу что видишь: какие приложения открыты, "
    "что происходит на экране. Говори на русском языке. "
    "Будь конкретным, не используй длинные перечисления. "
    "Максимум 2-3 предложения если не просят подробнее."
)


class EyesSkill(BaseSkill):

    def __init__(self, app, name):
        super().__init__(app, name)
        self._config = self._load_config()
        self._last_screenshot_b64: str | None = None
        self._context_active = False
        self._lock = threading.Lock()

    # ── Настройки (интерфейс BaseSkill) ──────────────────────────────────────

    def get_settings(self) -> list:
        return [
            {
                "key": "api_key",
                "label": "Groq API ключ",
                "placeholder": "gsk_...",
                "secret": True
            }
        ]

    def load_setting(self, key: str, default: str = "") -> str:
        return self._config.get(key, default)

    def save_setting(self, key: str, value: str):
        self.save_config({key: value})

    # ── Конфиг ────────────────────────────────────────────────────────────────

    def _config_path(self) -> Path:
        base = Path(__file__).parent.parent / "DEKS_DATA"
        return base / CONFIG_FILENAME

    def _load_config(self) -> dict:
        path = self._config_path()
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in DEFAULT_CONFIG.items():
                    data.setdefault(k, v)
                return data
            except Exception:
                pass
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)
        return dict(DEFAULT_CONFIG)

    def save_config(self, updates: dict):
        """Сохранить изменения конфига (вызывается из UI настроек)."""
        self._config.update(updates)
        path = self._config_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)

    def is_configured(self) -> bool:
        return bool(self._config.get("api_key", "").strip())

    # ── Контракт скилла ───────────────────────────────────────────────────────

    def handle(self, command: str) -> str | None:
        cmd = command.lower().strip()

        # Уточнение — используем тот же скриншот из памяти
        if self._context_active and self._last_screenshot_b64:
            if self.is_hit(cmd, "eyes_clarify"):
                return self._ask_vision(cmd, reuse_screenshot=True)

        # Новый взгляд на экран
        if self.is_hit(cmd, "eyes_look"):
            return self._ask_vision(cmd, reuse_screenshot=False)

        # Команда не наша — сбрасываем контекст
        if self._context_active:
            self._clear_context()

        return None

    # ── Скриншот ──────────────────────────────────────────────────────────────

    def _take_screenshot_b64(self) -> str | None:
        """Скриншот целиком в память → base64. На диск ничего не пишем."""
        if not MSS_AVAILABLE or not PIL_AVAILABLE:
            return None
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[0]
                raw = sct.grab(monitor)
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                img.thumbnail((1280, 720), Image.LANCZOS)
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=85)
                buf.seek(0)
                return base64.b64encode(buf.read()).decode("utf-8")
        except Exception as e:
            self._log(f"[Eyes] Ошибка скриншота: {e}")
            return None

    # ── Vision API ────────────────────────────────────────────────────────────

    def _ask_vision(self, user_text: str, reuse_screenshot: bool) -> str:
        if not MSS_AVAILABLE:
            return "Для навыка Eyes нужна библиотека mss. Установите: pip install mss"
        if not PIL_AVAILABLE:
            return "Для навыка Eyes нужна библиотека Pillow. Установите: pip install Pillow"
        if not self.is_configured():
            return (
                "Навык Eyes не настроен. "
                "Добавьте API ключ в настройках скилла. "
                "Получить бесплатный ключ: console.groq.com/keys"
            )

        if reuse_screenshot and self._last_screenshot_b64:
            screenshot_b64 = self._last_screenshot_b64
        else:
            screenshot_b64 = self._take_screenshot_b64()
            if not screenshot_b64:
                return "Не удалось сделать скриншот экрана."
            with self._lock:
                self._last_screenshot_b64 = screenshot_b64
                self._context_active = True

        question = (
            f"Пользователь уточняет: '{user_text}'. Посмотри внимательнее на тот же экран."
            if reuse_screenshot
            else "Что сейчас на экране? Опиши кратко."
        )

        try:
            return self._call_vision_api(screenshot_b64, question)
        except Exception as e:
            self._log(f"[Eyes] Ошибка Vision API: {e}")
            return "Не удалось получить ответ от Vision AI. Проверьте API ключ в настройках."

    def _call_vision_api(self, image_b64: str, question: str) -> str:
        api_key = self._config["api_key"].strip()
        api_url = self._config.get("api_url", DEFAULT_CONFIG["api_url"])
        model = self._config.get("model", DEFAULT_CONFIG["model"])

        payload = {
            "model": model,
            "max_tokens": 512,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": VISION_SYSTEM_PROMPT + "\n\n" + question
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                        }
                    ]
                }
            ]
        }

        body = json.dumps(payload).encode("utf-8")
        parsed = urllib.parse.urlparse(api_url)
        conn = http.client.HTTPSConnection(parsed.netloc, timeout=20)
        conn.request("POST", parsed.path, body=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })
        resp = conn.getresponse()
        data = json.loads(resp.read().decode("utf-8"))
        conn.close()

        if resp.status != 200:
            error = data.get("error", {}).get("message", str(data))
            raise Exception(f"API error {resp.status}: {error}")

        return data["choices"][0]["message"]["content"].strip()

    # ── Утилиты ───────────────────────────────────────────────────────────────

    def _clear_context(self):
        with self._lock:
            self._last_screenshot_b64 = None
            self._context_active = False

    def _log(self, msg: str):
        try:
            import logging
            logging.getLogger("DEKS").info(msg)
        except Exception:
            pass
