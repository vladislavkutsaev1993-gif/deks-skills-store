"""
eyes.py — навык зрения DEKS.
Анализирует экран через Vision AI (OpenRouter).
Скриншот хранится только в RAM, на диск не пишется.

Конфиг: DEKS_DATA/eyes_config.json
Получить API ключ бесплатно: https://openrouter.ai
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
    "model": "google/gemma-4-26b-a4b-it:free",
}


def _build_vision_prompt(n_monitors: int) -> str:
    if n_monitors == 1:
        return (
            "Опиши кратко что видишь на этом скриншоте: "
            "какие приложения открыты, что происходит на экране. "
            "Только факты, без лишних слов. На русском языке. "
            "Максимум 3 предложения."
        )
    return (
        f"На этом изображении объединены {n_monitors} монитора слева направо. "
        f"Опиши кратко что происходит на каждом мониторе по порядку: "
        f"какие приложения открыты, что делается. "
        f"Только факты, на русском языке. Максимум 2 предложения на монитор."
    )


class EyesSkill(BaseSkill):

    def __init__(self, app, name):
        super().__init__(app, name)
        self._config = self._load_config()
        self._last_screenshot: str | None = None
        self._last_n_monitors: int = 1
        self._context_active = False
        self._lock = threading.Lock()

    # ── Настройки ────────────────────────────────────────────────────────────

    def get_settings(self) -> list:
        return [
            {
                "key": "api_key",
                "label": "OpenRouter API ключ",
                "placeholder": "sk-or-v1-...",
                "secret": True
            }
        ]

    def get_data_files(self) -> list:
        return ["eyes_config.json"]

    def load_setting(self, key: str, default: str = "") -> str:
        return self._config.get(key, default)

    def save_setting(self, key: str, value: str):
        self.save_config({key: value})

    # ── Конфиг ───────────────────────────────────────────────────────────────

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
        self._config.update(updates)
        path = self._config_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)

    def is_configured(self) -> bool:
        return bool(self._config.get("api_key", "").strip())

    # ── Контракт скилла ──────────────────────────────────────────────────────

    def handle(self, command: str) -> str | None:
        cmd = command.lower().strip()

        if self._context_active and self._last_screenshot:
            if self.is_hit(cmd, "eyes_clarify"):
                return self._ask_vision(cmd, reuse_screenshot=True)

        if self.is_hit(cmd, "eyes_look"):
            return self._ask_vision(cmd, reuse_screenshot=False)

        if self._context_active:
            self._clear_context()

        return None

    # ── Скриншот ─────────────────────────────────────────────────────────────

    def _take_screenshot(self) -> tuple[str | None, int]:
        """
        Снимает объединённый скриншот всех мониторов (monitors[0]).
        Возвращает (base64_str, n_monitors).
        """
        if not MSS_AVAILABLE or not PIL_AVAILABLE:
            return None, 0
        try:
            with mss.mss() as sct:
                n_monitors = len(sct.monitors) - 1  # считаем реальные мониторы
                raw = sct.grab(sct.monitors[0])      # [0] = объединённый экран
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                img.thumbnail((1920, 800), Image.LANCZOS)
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=85)
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode("utf-8")
                return img_b64, n_monitors
        except Exception as e:
            self._log(f"[Eyes] Ошибка скриншота: {e}")
            return None, 0

    # ── Vision ───────────────────────────────────────────────────────────────

    def _ask_vision(self, user_text: str, reuse_screenshot: bool) -> str:
        if not MSS_AVAILABLE:
            return "Для навыка Eyes нужна библиотека mss. Установите: pip install mss"
        if not PIL_AVAILABLE:
            return "Для навыка Eyes нужна библиотека Pillow. Установите: pip install Pillow"
        if not self.is_configured():
            return (
                "Навык Eyes не настроен. "
                "Добавьте API ключ в настройках скилла. "
                "Получить бесплатный ключ: openrouter.ai"
            )

        if reuse_screenshot and self._last_screenshot:
            img_b64 = self._last_screenshot
            n_monitors = self._last_n_monitors
        else:
            img_b64, n_monitors = self._take_screenshot()
            if not img_b64:
                return "Не удалось сделать скриншот экрана."
            with self._lock:
                self._last_screenshot = img_b64
                self._last_n_monitors = n_monitors
                self._context_active = True

        self._log(f"[Eyes] Скриншот готов, мониторов: {n_monitors}")

        try:
            screen_desc = self._call_vision_api(img_b64, n_monitors)
            self._log("[Eyes] Vision API ответил")
        except Exception as e:
            self._log(f"[Eyes] Ошибка Vision API: {e}")
            return f"Не удалось получить описание экрана: {e}"

        return self._ask_main_llm(user_text, screen_desc, n_monitors)

    def _call_vision_api(self, image_b64: str, n_monitors: int) -> str:
        """Отправляет объединённый скриншот в OpenRouter Vision. Возвращает описание."""
        api_key = self._config["api_key"].strip()
        model = self._config.get("model", DEFAULT_CONFIG["model"])
        prompt_text = _build_vision_prompt(n_monitors)

        payload = {
            "model": model,
            "max_tokens": 400,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                ]
            }]
        }

        body = json.dumps(payload).encode("utf-8")
        conn = http.client.HTTPSConnection("openrouter.ai", timeout=25)
        conn.request("POST", "/api/v1/chat/completions", body=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://deks.app",
        })
        resp = conn.getresponse()
        data = json.loads(resp.read().decode("utf-8"))
        conn.close()

        if resp.status != 200:
            error = data.get("error", {}).get("message", str(data))
            raise Exception(f"API error {resp.status}: {error}")

        return data["choices"][0]["message"]["content"].strip()

    def _ask_main_llm(self, user_question: str, screen_desc: str, n_monitors: int) -> str:
        """Отправляет описание экрана + вопрос пользователя в главный LLM."""
        try:
            from config import load_api_keys
            keys = load_api_keys()
            api_key = keys.get("controller_key", "")
            api_url = keys.get("controller_url",
                               "https://api.groq.com/openai/v1/chat/completions")
            model = keys.get("controller_model", "llama-3.3-70b-versatile")
        except Exception as e:
            self._log(f"[Eyes] Не удалось загрузить конфиг контроллера: {e}")
            return screen_desc

        monitors_info = f"{n_monitors} монитор{'а' if 2 <= n_monitors <= 4 else 'ов' if n_monitors > 4 else ''}"
        prompt = (
            f"Пользователь спросил: \"{user_question}\"\n\n"
            f"На экране ({monitors_info}):\n{screen_desc}\n\n"
            f"Ответь пользователю кратко и по делу. "
            f"Стиль — Джарвис. Только русский язык. Максимум 2-3 предложения."
        )

        payload = {
            "model": model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
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
                return screen_desc

            return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            self._log(f"[Eyes] Ошибка главного LLM: {e}")
            return screen_desc

    # ── Утилиты ──────────────────────────────────────────────────────────────

    def _clear_context(self):
        with self._lock:
            self._last_screenshot = None
            self._last_n_monitors = 1
            self._context_active = False

    def _log(self, msg: str):
        try:
            import logging
            logging.getLogger("DEKS").info(msg)
        except Exception:
            pass
