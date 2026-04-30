"""
GoalPlanner — динамическое выполнение целей.

Цикл как описал GPT:
  goal
   → observe (DOM snapshot)
   → plan    (какие шаги нужны)
   → act     (выполнить шаги)
   → verify  (достигнута ли цель)
   → retry   (если нет — replanning)

В отличие от простого command-based подхода:
  НЕ "нажми play"
  А  "сделай чтобы видео играло"
"""

import time
from deks_viver.controller import ViewerController
from deks_viver.session    import ViewerSession


class GoalPlanner:

    MAX_RETRIES = 3

    def execute(
        self,
        goal:    str,
        url:     str,
        ctrl:    ViewerController,
        session: ViewerSession,
    ) -> dict:
        """
        Выполнить высокоуровневую цель.
        Возвращает финальное состояние сессии.
        """
        session.last_goal = goal
        goal_lower = goal.lower()

        # 1. Открыть URL если указан
        if url:
            ctrl.open_url(url)
            session.update_from_dom(ctrl.get_dom_state())

        # 2. Начальное наблюдение (DOM-first)
        observation = ctrl.get_dom_snapshot()
        session.update_from_dom(observation)

        # 3. Цикл: plan → act → verify → retry
        success = False
        for attempt in range(self.MAX_RETRIES):
            steps = self._plan(goal_lower, observation)
            print(f"[GOAL] Attempt {attempt+1}, steps: {steps}")

            for step in steps:
                ok, err = self._act(step, ctrl)
                if not ok:
                    print(f"[GOAL] Step '{step}' failed: {err}")
                time.sleep(0.3)

            # Наблюдаем результат
            observation = ctrl.get_dom_snapshot()
            session.update_from_dom(observation)

            if self._verify(goal_lower, observation):
                success = True
                print(f"[GOAL] Goal achieved after {attempt+1} attempt(s)")
                break
            else:
                print(f"[GOAL] Not achieved yet, replanning...")

        if not success:
            session.record_failed(goal, "max_retries_exceeded")

        # 4. Fullscreen если в цели
        if any(w in goal_lower for w in ["полный", "весь экран", "fullscreen", "развернуть"]):
            time.sleep(0.5)
            ctrl.fullscreen()

        # 5. Запись в историю просмотров
        if session.title and session.url:
            session.record_watched(session.title, session.url)

        return session.to_dict()

    # ── Планирование ──────────────────────────────────────────────────────

    def _plan(self, goal_lower: str, obs: dict) -> list:
        """
        На основе цели и текущего наблюдения строим список шагов.
        Это простой rule-based планировщик.
        В будущем сюда можно вставить LLM для динамического плана.
        """
        steps = []

        # Сначала всегда пробуем убрать попапы если есть оверлей
        if obs.get("has_overlay"):
            steps.append("dismiss_popups")

        # Цель: воспроизвести медиа
        play_words = {"играет", "запусти", "включи", "play", "плей", "воспроизведи", "смотреть"}
        if any(w in goal_lower for w in play_words):
            if not obs.get("playing"):
                steps.append("force_play")

        # Цель: пауза
        pause_words = {"пауза", "останови", "pause", "stop"}
        if any(w in goal_lower for w in pause_words):
            if obs.get("playing"):
                steps.append("pause")

        # Цель: перемотка / навигация
        if "назад" in goal_lower and "вивер" not in goal_lower:
            steps.append("back")
        if any(w in goal_lower for w in ["обнови", "перезагрузи"]):
            steps.append("reload")

        # Если пустой план — хотя бы обновить состояние
        return steps or ["get_state"]

    # ── Выполнение шага ───────────────────────────────────────────────────

    def _act(self, step: str, ctrl: ViewerController) -> tuple:
        """Выполнить один шаг. Возвращает (success, error_msg)."""
        try:
            if step == "dismiss_popups":
                ctrl.dismiss_popups()
            elif step == "force_play":
                ctrl.force_play()
            elif step == "pause":
                ctrl.pause()
            elif step == "fullscreen":
                ctrl.fullscreen()
            elif step == "back":
                ctrl.back()
            elif step == "reload":
                ctrl.reload()
            elif step == "get_state":
                ctrl.get_dom_state()
            else:
                print(f"[GOAL] Unknown step: {step}")
            return True, ""
        except Exception as ex:
            return False, str(ex)

    # ── Проверка достижения цели ──────────────────────────────────────────

    def _verify(self, goal_lower: str, obs: dict) -> bool:
        """
        Проверяем: цель достигнута?
        DOM-based — без скриншота.
        """
        play_words = {"играет", "запусти", "включи", "play", "плей", "воспроизведи"}
        if any(w in goal_lower for w in play_words):
            return obs.get("playing", False)

        pause_words = {"пауза", "останови", "pause"}
        if any(w in goal_lower for w in pause_words):
            return not obs.get("playing", True)

        # По умолчанию считаем что успешно
        return True
