"""
PlaywrightBackend — конкретная реализация ViewerController на Playwright.

Ключевые фичи (по советам GPT):
- Персистентный Chrome-профиль (логины, cookies сохраняются)
- Реальный Chrome с поддержкой DRM (Netflix, Spotify)
- DOM-first наблюдение: rich snapshot с кнопками, полями, оверлеями
- Авто-dismiss попапов перед любым взаимодействием
- Авто-плей после загрузки страницы
"""

import os
import time

from deks_viver.controller import ViewerController

# Персистентный профиль Chrome — логины и куки сохраняются между сессиями
_PROFILE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "DEKS_DATA", "deks_viver", "chrome_profile")
)

# Популярные Play-кнопки на разных сайтах
_PLAY_SELECTORS = [
    'button[aria-label*="play" i]',
    'button[title*="play" i]',
    'button[aria-label*="воспроизвести" i]',
    '[data-testid*="play-button" i]',
    '[data-testid*="playButton" i]',
    '[class*="PlayButton"]',
    '[class*="play-btn" i]',
    '[class*="play-button" i]',
    '.ytp-play-button',          # YouTube
    '[aria-label="Play"]',
    '[aria-label="Воспроизвести"]',
    'button.play',
]

# Consent / cookie banner кнопки
_ACCEPT_SELECTORS = [
    'button:has-text("Accept all")',
    'button:has-text("Accept All")',
    'button:has-text("Принять всё")',
    'button:has-text("Принять")',
    'button:has-text("Agree")',
    'button:has-text("I Agree")',
    'button:has-text("OK")',
    'button:has-text("Got it")',
    'button:has-text("Okay")',
    'button[id*="accept" i]',
    'button[class*="accept" i]',
    '#accept-all',
    '#onetrust-accept-btn-handler',
    '[data-testid*="accept" i]',
    '[aria-label*="accept" i]',
    '.cookie-accept',
    '.fc-cta-consent',
]


class PlaywrightBackend(ViewerController):
    """Playwright — полноценный браузерный движок с automation."""

    def __init__(self, page, ctx):
        self.page = page
        self._ctx  = ctx   # PersistentContext

    # ── Навигация ─────────────────────────────────────────────────────────

    def open_url(self, url: str):
        self.page.goto(url, wait_until="domcontentloaded", timeout=20000)
        time.sleep(1.2)
        self.dismiss_popups()
        time.sleep(0.6)
        self._try_autoplay()

    def back(self):
        self.page.go_back(timeout=8000)

    def forward(self):
        self.page.go_forward(timeout=8000)

    def reload(self):
        self.page.reload(wait_until="domcontentloaded", timeout=15000)

    # ── Медиа ─────────────────────────────────────────────────────────────

    def force_play(self):
        """3 метода подряд: JS → селекторы → клавиша k."""
        # 1. Прямой JS
        try:
            self.page.evaluate(
                "var v=document.querySelector('video');if(v&&v.paused)v.play();"
            )
        except Exception:
            pass
        time.sleep(0.35)

        # 2. Клик по кнопке play
        for sel in _PLAY_SELECTORS:
            try:
                loc = self.page.locator(sel)
                if loc.count() > 0:
                    loc.first.click(timeout=1500)
                    break
            except Exception:
                continue

        # 3. Клавиша k (YouTube и большинство плееров)
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
        """Сначала fullscreen видео-элемента, потом F11."""
        try:
            self.page.evaluate(
                "var v=document.querySelector('video');"
                "if(v){"
                "  var fn=v.requestFullscreen||v.webkitRequestFullscreen||v.mozRequestFullScreen;"
                "  if(fn){fn.call(v);}"
                "  else{document.documentElement.requestFullscreen();}"
                "}else{"
                "  document.documentElement.requestFullscreen();"
                "}"
            )
        except Exception:
            pass
        time.sleep(0.3)
        try:
            self.page.keyboard.press("F11")
        except Exception:
            pass

    def _try_autoplay(self):
        """Тихая попытка auto-play после загрузки."""
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

    # ── UI-взаимодействие ─────────────────────────────────────────────────

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

    def key(self, k: str):
        self.page.keyboard.press(k)

    def scroll(self, direction: str = "down", amount: int = 3):
        kb = "PageDown" if direction == "down" else "PageUp"
        for _ in range(amount):
            self.page.keyboard.press(kb)
            time.sleep(0.05)

    def dismiss_popups(self) -> bool:
        for sel in _ACCEPT_SELECTORS:
            try:
                loc = self.page.locator(sel)
                if loc.count() > 0:
                    loc.first.click(timeout=1200)
                    return True
            except Exception:
                continue
        return False

    # ── Наблюдение ────────────────────────────────────────────────────────

    def get_dom_state(self) -> dict:
        """Минимальный быстрый снапшот."""
        result = {}
        try:
            result["url"]       = self.page.url
            result["title"]     = self.page.title()
            result["playing"]   = self.page.evaluate(
                "var v=document.querySelector('video');v?!v.paused:false"
            )
            result["video_src"] = self.page.evaluate(
                "var v=document.querySelector('video');"
                "v?(v.currentSrc||v.src||''):null"
            )
        except Exception:
            pass
        return result

    def get_dom_snapshot(self) -> dict:
        """
        Богатый снапшот для агента (DOM-first, как советовал GPT).
        Возвращает: state + buttons + inputs + overlays + video details.
        """
        snap = self.get_dom_state()
        try:
            # Видимые кликабельные элементы
            snap["buttons"] = self.page.evaluate("""
                Array.from(
                    document.querySelectorAll('button, a[href], [role="button"], [role="link"]')
                ).slice(0, 25).map(el => ({
                    text: (el.innerText || el.title || el.getAttribute('aria-label') || '').trim().slice(0, 50),
                    visible: el.offsetParent !== null,
                    tag: el.tagName.toLowerCase()
                })).filter(b => b.text && b.visible)
            """)

            # Поля ввода
            snap["inputs"] = self.page.evaluate("""
                Array.from(
                    document.querySelectorAll('input:not([type=hidden]), textarea, select')
                ).slice(0, 10).map(el => ({
                    type: el.type || el.tagName.toLowerCase(),
                    placeholder: (el.placeholder || '').slice(0, 40),
                    name: (el.name || el.id || '').slice(0, 30),
                    visible: el.offsetParent !== null
                })).filter(i => i.visible)
            """)

            # Есть ли модальное окно/оверлей
            snap["has_overlay"] = self.page.evaluate("""
                !!(document.querySelector(
                    '[class*="modal" i], [class*="overlay" i], [class*="popup" i],' +
                    '[class*="dialog" i], [role="dialog"], [role="alertdialog"]'
                ))
            """)

            # Детальное состояние видео
            snap["video_duration"] = self.page.evaluate(
                "var v=document.querySelector('video');"
                "v&&!isNaN(v.duration)?Math.round(v.duration):null"
            )
            snap["video_time"] = self.page.evaluate(
                "var v=document.querySelector('video');"
                "v&&!isNaN(v.currentTime)?Math.round(v.currentTime):null"
            )
            snap["video_muted"] = self.page.evaluate(
                "var v=document.querySelector('video');v?v.muted:null"
            )

        except Exception:
            pass
        return snap

    def capture(self) -> str:
        """Скриншот через Playwright — нативно, без внешних инструментов."""
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

    def close(self):
        try:
            self._ctx.close()
        except Exception:
            pass

    # ── Запуск с персистентным профилем ──────────────────────────────────

    @staticmethod
    def launch(pw) -> tuple:
        """
        Запустить браузер с персистентным профилем.
        Логины, cookies, сессии — всё сохраняется между запусками.
        Возвращает (PlaywrightBackend, context).
        """
        os.makedirs(_PROFILE_DIR, exist_ok=True)
        print(f"[VIVER] Chrome profile: {_PROFILE_DIR}")

        launch_kwargs = dict(
            headless=False,
            args=[
                "--start-maximized",
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--disable-default-apps",
            ],
            no_viewport=True,
        )

        # Реальный Chrome (DRM, Widevine для Netflix/Spotify)
        try:
            ctx = pw.chromium.launch_persistent_context(
                _PROFILE_DIR,
                channel="chrome",
                **launch_kwargs,
            )
            print("[VIVER] Using real Chrome (DRM supported)")
        except Exception as e:
            print(f"[VIVER] Chrome not found ({e}), falling back to Chromium")
            ctx = pw.chromium.launch_persistent_context(
                _PROFILE_DIR,
                **launch_kwargs,
            )

        page = ctx.new_page()
        page.set_extra_http_headers({"Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8"})
        return PlaywrightBackend(page, ctx), ctx
