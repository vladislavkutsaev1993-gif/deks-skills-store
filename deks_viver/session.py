"""
ViewerSession — состояние и память вивера.

Хранит: текущий URL/title, историю, что смотрели,
неудачные действия, состояние логина по доменам.
Персистируется в DEKS_DATA/deks_viver/session.json.
"""

import os
import json
from datetime import datetime

_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "DEKS_DATA", "deks_viver"
)
_SESSION_FILE = os.path.join(_DATA_DIR, "session.json")


class ViewerSession:
    """
    Полная память вивера — не только текущее состояние,
    но и история, ошибки, логины.
    """

    def __init__(self):
        # ── Текущее состояние ──────────────────────────────────────────
        self.url:        str  = ""
        self.title:      str  = ""
        self.playing:    bool = False
        self.fullscreen: bool = False

        # ── Память ─────────────────────────────────────────────────────
        self.history:        list = []   # list[str] — посещённые URL
        self.watched:        list = []   # list[dict] — {title, url, ts}
        self.failed_actions: list = []   # list[dict] — {action, reason, ts}
        self.last_goal:      str  = ""
        self.login_state:    dict = {}   # {domain: True/False}

    # ── Обновление из DOM ──────────────────────────────────────────────

    def update_from_dom(self, dom: dict):
        url = dom.get("url", "")
        if url and url not in ("about:blank", "null", "", None):
            if url != self.url:
                self.url = url
                if url not in self.history:
                    self.history.append(url)
        title = dom.get("title", "")
        if title and title not in ("", "null", None):
            self.title = title
        if "playing" in dom:
            self.playing = bool(dom["playing"])

    # ── Запись событий ─────────────────────────────────────────────────

    def record_watched(self, title: str, url: str):
        """Запомнить что посмотрели."""
        item = {"title": title, "url": url, "ts": _now()}
        # Не дублируем
        if not any(w.get("url") == url for w in self.watched):
            self.watched.append(item)
        self.save()

    def record_failed(self, action: str, reason: str):
        """Запомнить неудачное действие для последующего анализа."""
        self.failed_actions.append({
            "action": action,
            "reason": reason,
            "ts": _now(),
        })

    def mark_login(self, domain: str, logged_in: bool = True):
        self.login_state[domain] = logged_in

    # ── Сериализация ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "url":            self.url,
            "title":          self.title,
            "playing":        self.playing,
            "fullscreen":     self.fullscreen,
            "history":        self.history[-30:],
            "watched":        self.watched[-15:],
            "failed_actions": self.failed_actions[-10:],
            "last_goal":      self.last_goal,
            "login_state":    self.login_state,
        }

    def save(self):
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(_SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls) -> "ViewerSession":
        sess = cls()
        if os.path.exists(_SESSION_FILE):
            try:
                with open(_SESSION_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sess.history     = data.get("history", [])
                sess.watched     = data.get("watched", [])
                sess.login_state = data.get("login_state", {})
                sess.failed_actions = data.get("failed_actions", [])
            except Exception:
                pass
        return sess


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
