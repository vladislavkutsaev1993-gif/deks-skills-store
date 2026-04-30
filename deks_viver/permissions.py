"""
Action Permissions — sandbox для вивера.

Как советовал GPT: агент не должен иметь неограниченный доступ.
Опасные действия (форм-сабмит, покупки, загрузки) блокируются или
требуют явного подтверждения пользователя.
"""

# ── Безопасные действия — выполняются сразу ───────────────────────────────
SAFE_ACTIONS = {
    "open_url",
    "play", "force_play", "pause",
    "fullscreen", "fullscreen_page",
    "back", "forward", "reload",
    "scroll",
    "key",
    "click",           # клик по элементам страницы (не формы)
    "dismiss_popups",
    "execute_js",
    "capture",
    "get_state",
    "get_snapshot",
    "execute_goal",
    "close",
}

# ── Требуют подтверждения — DEKS спрашивает перед выполнением ─────────────
CONFIRM_ACTIONS = {
    "type",            # ввод текста (может быть пароль)
    "click_submit",    # submit любой формы
    "download",        # скачивание файлов
}

# ── Заблокированы полностью ───────────────────────────────────────────────
BLOCKED_ACTIONS = {
    "purchase",        # любая оплата
    "login_submit",    # отправка credentials
    "delete_account",  # удаление аккаунтов
}


def check(action: str) -> tuple:
    """
    Проверить разрешение на действие.
    Возвращает (allowed, message):
      (True, "")         → выполнять сразу
      (None, "confirm")  → нужно подтверждение
      (False, reason)    → заблокировано
    """
    if action in BLOCKED_ACTIONS:
        return False, f"Действие '{action}' заблокировано системой безопасности."
    if action in CONFIRM_ACTIONS:
        return None, "confirm"
    # SAFE_ACTIONS и всё неизвестное — разрешено (fail-open для MVP)
    return True, ""


def describe(action: str) -> str:
    """Человекочитаемое описание политики для action."""
    if action in BLOCKED_ACTIONS:
        return "заблокировано"
    if action in CONFIRM_ACTIONS:
        return "требует подтверждения"
    return "разрешено"
