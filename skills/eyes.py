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

VISION_PROMPT = (
    "Опиши кратко что видишь на этом скриншоте: "
    "какие приложения открыты, что происходит на экране. "
    "Только факты, без лишних слов. На русском языке. "
    "Максимум 3 предложения."
)

VISION_MONITOR_PROMPT = (
    "Это монитор {n} из {total}. "
    "Опиши кратко что видишь: какие приложения открыты, что происходит. "
    "Только факты, на русском языке. Максимум 2 предложения."
)


class EyesSkill(BaseSkill):

    def __init__(self, app, name):
        super().__init__(app, name)
        self._config = self._load_config()
        self._last_screenshots: list = []
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

    def get_data_files(self) -> list:
        return ["eyes_config.json"]

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
        self._config.update(updates)
        path = self._config_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)

    def is_configured(self) -> bool:
        return bool(self._config.get("api_key", "").strip())

    # ── Контракт скилла ───────────────────────────────────────────────────────

    def handle(self, command: str) -> str | None:
        cmd = command.lower().strip()

        if self._context_active and self._last_screenshots:
            if self.is_hit(cmd, "eyes_clarify"):
                return self._ask_vision(cmd, reuse_screenshot=True)

        if self.is_hit(cmd, "eyes_look"):
            return self._ask_vision(cmd, reuse_screenshot=False)

        if self._context_active:
            self._clear_context()

        return None

    # ── Скриншот ──────────────────────────────────────────────────────────────

    def _take_all_screenshots(self) -> list:
        """Снимает каждый монитор отдельно. Возвращает список base64."""
        if not MSS_AVAILABLE or not PIL_AVAILABLE:
            return []
        results = []
        try:
            with mss.mss() as sct:
                n_monitors = len(sct.monitors) - 1  # monitors[0] — виртуальный
                for i in range(1, n_monitors + 1):
                    try:
                        raw = sct.grab(sct.monitors[i])
                        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                        img.thumbnail((1280, 720), Image.LANCZOS)
                        buf = BytesIO()
                        img.save(buf, format="JPEG", quality=85)
                        buf.seek(0)
                        results.append(base64.b64encode(buf.read()).decode("utf-8"))
                    except Exception as e:
                        self._log(f"[Eyes] Ошибка снимка монитора {i}: {e}")
                        results.append(None)
        except Exception as e:
            self._log(f"[Eyes] Ошибка mss: {e}")
        return results

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

        if reuse_screenshot and self._last_screenshots:
            screenshots = self._last_screenshots
        else:
            screenshots = self._take_all_screenshots()
            if not any(s for s in screenshots if s):
                return "Не удалось сделать скриншот экрана."
            with self._lock:
                self._last_screenshots = screenshots
                self._context_active = True

        valid = [(i, s) for i, s in enumerate(screenshots) if s]
        total = len(valid)

        # Отправляем каждый монитор по очереди в Vision API
        monitor_descriptions = []
        for order, (idx, img_b64) in enumerate(valid):
            monitor_num = order + 1
            try:
                desc = self._call_vision_api(img_b64, monitor_num, total)
                monitor_descriptions.append(f"Монитор {monitor_num}: {desc}")
                self._log(f"[Eyes] Монитор {monitor_num} описан")
            except Exception as e:
                self._log(f"[Eyes] Ошибка монитора {monitor_num}: {e}")
                monitor_descriptions.append(f"Монитор {monitor_num}: не удалось получить данные ({e})")

        if not monitor_descriptions:
            return "Не удалось получить данные с мониторов."

        # Все описания готовы — отправляем в главный LLM
        return self._ask_main_llm(user_text, monitor_descriptions)

    def _call_vision_api(self, image_b64: str, monitor_n: int, total: int) -> str:
        """Отправляет скриншот одного монитора в Vision API. Возвращает описание."""
        api_key = self._config["api_key"].strip()
        api_url = self._config.get("api_url", DEFAULT_CONFIG["api_url"])
        model = self._config.get("model", DEFAULT_CONFIG["model"])

        if total > 1:
            prompt_text = VISION_MONITOR_PROMPT.format(n=monitor_n, total=total)
        else:
            prompt_text = VISION_PROMPT

        payload = {
            "model": model,
            "max_tokens": 300,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
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

    def _ask_main_llm(self, user_question: str, monitor_descriptions: list) -> str:
        """Отправляет описания мониторов + вопрос пользователя в главный LLM."""
        try:
            from config import load_api_keys
            keys = load_api_keys()
            api_key = keys.get("controller_key", "")
            api_url = keys.get("controller_url",
                                "https://api.groq.com/openai/v1/chat/completions")
            model = keys.get("controller_model", "llama-3.3-70b-versatile")
        except Exception as e:
            self._log(f"[Eyes] Не удалось загрузить конфиг контроллера: {e}")
            return "\n".join(monitor_descriptions)

        screen_context = "\n".join(monitor_descriptions)
        prompt = (
            f"Пользователь спросил: \"{user_question}\"\n\n"
            f"На экране:\n{screen_context}\n\n"
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
                return "\n".join(monitor_descriptions)

            return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            self._log(f"[Eyes] Ошибка главного LLM: {e}")
            return "\n".join(monitor_descriptions)

    # ── Утилиты ───────────────────────────────────────────────────────────────

    def _clear_context(self):
        with self._lock:
            self._last_screenshots = []
            self._context_active = False

    def _log(self, msg: str):
        try:
            import logging
            logging.getLogger("DEKS").info(msg)
        except Exception:
            pass
