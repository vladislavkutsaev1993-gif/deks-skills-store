"""
ViewerController — абстрактный интерфейс backend-агента.

Как GPT и сказал: если вдруг Playwright сменить на Selenium,
remote browser, Android device — DEKS этого не заметит.
Меняем только реализацию, не API.
"""


class ViewerController:
    """
    Абстракция над браузерным движком.
    Все конкретные реализации (PlaywrightBackend, ...) наследуют этот класс.
    """

    # ── Навигация ─────────────────────────────────────────────────────────

    def open_url(self, url: str):
        """Открыть URL, подождать загрузку, dismiss попапы, авто-плей."""
        raise NotImplementedError

    def back(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def reload(self):
        raise NotImplementedError

    # ── Медиа ─────────────────────────────────────────────────────────────

    def force_play(self):
        """Запустить воспроизведение всеми доступными методами."""
        raise NotImplementedError

    def pause(self):
        raise NotImplementedError

    def fullscreen(self):
        """Fullscreen: видео-элемент → F11 браузера."""
        raise NotImplementedError

    # ── UI-взаимодействие ─────────────────────────────────────────────────

    def click(self, selector: str = None, x: int = None, y: int = None):
        raise NotImplementedError

    def type_text(self, text: str, selector: str = None):
        raise NotImplementedError

    def key(self, key: str):
        raise NotImplementedError

    def scroll(self, direction: str = "down", amount: int = 3):
        raise NotImplementedError

    def dismiss_popups(self) -> bool:
        """Авто-закрыть cookie-баннеры, consent-диалоги."""
        raise NotImplementedError

    # ── Наблюдение (DOM-first, vision только как fallback) ─────────────────

    def get_dom_state(self) -> dict:
        """
        Минимальное состояние: url, title, playing.
        Быстро и дёшево.
        """
        raise NotImplementedError

    def get_dom_snapshot(self) -> dict:
        """
        Богатый снапшот: url, title, playing, buttons, inputs,
        overlays, video_duration, video_time.
        Основной метод наблюдения для агента.
        """
        raise NotImplementedError

    def capture(self) -> str:
        """
        Скриншот → путь к PNG.
        Vision-fallback когда DOM недостаточно.
        """
        raise NotImplementedError

    def execute_js(self, code: str):
        """Выполнить произвольный JS в контексте страницы."""
        raise NotImplementedError

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self):
        raise NotImplementedError
