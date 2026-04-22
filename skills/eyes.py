"""
eyes.py — навык зрения DEKS.
Анализирует экран локально через SmolVLM-256M.
Скриншоты только в RAM, на диск не сохраняются.
Не требует API ключей. Работает полностью офлайн.
"""

import threading
import urllib.parse
import http.client
import json
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

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def _build_prompt(n_monitors: int) -> str:
    if n_monitors == 1:
        return (
            "Describe briefly what you see on this screenshot: "
            "which apps are open, what is happening on screen. "
            "Facts only. Answer in Russian. Max 3 sentences."
        )
    return (
        f"This image combines {n_monitors} monitors side by side (left to right). "
        f"Briefly describe what is on each monitor in order: "
        f"which apps are open, what is happening. "
        f"Facts only. Answer in Russian. Max 2 sentences per monitor."
    )


class EyesSkill(BaseSkill):

    def __init__(self, app, name):
        super().__init__(app, name)
        self._last_img = None
        self._last_n_monitors = 1
        self._context_active = False
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._model_lock = threading.Lock()
        self._model_ready = False
        # Прогреваем модель в фоне сразу при старте
        threading.Thread(target=self._load_model, daemon=True).start()

    # ── Контракт маркета ─────────────────────────────────────────────────────

    def get_settings(self) -> list:
        return []  # Ключи не нужны — всё локально

    def get_data_files(self) -> list:
        return []  # Нет конфиг-файлов

    def is_configured(self) -> bool:
        return True  # Всегда готов если установлен

    # ── handle ───────────────────────────────────────────────────────────────

    def handle(self, command: str) -> str | None:
        cmd = command.lower().strip()

        if self._context_active and self._last_img is not None:
            if self.is_hit(cmd, "eyes_clarify"):
                return self._ask_vision(cmd, reuse=True)

        if self.is_hit(cmd, "eyes_look"):
            return self._ask_vision(cmd, reuse=False)

        if self._context_active:
            self._clear_context()

        return None

    # ── Модель ───────────────────────────────────────────────────────────────

    def _load_model(self) -> bool:
        with self._model_lock:
            if self._model_ready:
                return True
            import torch
            self._log("[Eyes] Загружаю SmolVLM в память...")

            # Попытка 1: новое имя (transformers 5+)
            try:
                from transformers import AutoProcessor, AutoModelForImageTextToText
                self._processor = AutoProcessor.from_pretrained(MODEL_ID)
                self._model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_ID,
                    dtype=torch.float32,
                    _attn_implementation="eager",
                )
                self._model_ready = True
                self._log("[Eyes] Модель готова")
                return True
            except Exception as e1:
                self._log(f"[Eyes] ImageTextToText не удался ({e1}), пробую Vision2Seq...")

            # Попытка 2: старое имя (transformers 4.x)
            try:
                from transformers import AutoProcessor, AutoModelForVision2Seq
                self._processor = AutoProcessor.from_pretrained(MODEL_ID)
                self._model = AutoModelForVision2Seq.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float32,
                    _attn_implementation="eager",
                )
                self._model_ready = True
                self._log("[Eyes] Модель готова")
                return True
            except Exception as e2:
                self._log(f"[Eyes] Ошибка загрузки модели: {e2}")
                return False

    # ── Скриншот ─────────────────────────────────────────────────────────────

    def _take_screenshot(self):
        if not MSS_AVAILABLE or not PIL_AVAILABLE:
            return None, 0
        try:
            with mss.mss() as sct:
                n = len(sct.monitors) - 1
                raw = sct.grab(sct.monitors[0])
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                img.thumbnail((1280, 720), Image.LANCZOS)
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
                return "Не удалось сделать скриншот экрана."
            with self._lock:
                self._last_img = img
                self._last_n_monitors = n
                self._context_active = True

        self._log(f"[Eyes] Скриншот готов, мониторов: {n}")

        if not self._model_ready:
            if not self._load_model():
                return "Модель зрения не загружена. Переустановите навык Eyes."

        try:
            desc = self._run_inference(img, n)
            self._log("[Eyes] Описание получено")
        except Exception as e:
            self._log(f"[Eyes] Ошибка inference: {e}")
            return f"Ошибка модели: {e}"

        return self._ask_main_llm(user_text, desc, n)

    def _run_inference(self, img: "Image.Image", n_monitors: int) -> str:
        import torch
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": _build_prompt(n_monitors)}
            ]
        }]
        prompt = self._processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self._processor(text=prompt, images=[img], return_tensors="pt")
        with torch.no_grad():
            ids = self._model.generate(**inputs, max_new_tokens=300)
        texts = self._processor.batch_decode(ids, skip_special_tokens=True)
        result = texts[0] if texts else ""
        if "Assistant:" in result:
            result = result.split("Assistant:")[-1].strip()
        return result

    def _ask_main_llm(self, user_question: str, screen_desc: str, n_monitors: int) -> str:
        try:
            from config import load_api_keys
            keys = load_api_keys()
            api_key = keys.get("controller_key", "")
            api_url = keys.get("controller_url",
                               "https://api.groq.com/openai/v1/chat/completions")
            model = keys.get("controller_model", "llama-3.3-70b-versatile")
        except Exception as e:
            self._log(f"[Eyes] Конфиг недоступен: {e}")
            return screen_desc

        n_word = (
            f"{n_monitors} монитор{'а' if 2 <= n_monitors <= 4 else 'ов' if n_monitors > 4 else ''}"
        )
        prompt = (
            f"Пользователь спросил: \"{user_question}\"\n\n"
            f"На экране ({n_word}):\n{screen_desc}\n\n"
            f"Ответь кратко и по делу. Стиль — Джарвис. "
            f"Только русский язык. Максимум 2-3 предложения."
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
            self._log(f"[Eyes] Ошибка LLM: {e}")
            return screen_desc

    # ── Утилиты ──────────────────────────────────────────────────────────────

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
