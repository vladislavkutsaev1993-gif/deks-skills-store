"""
test_skill.py — Тестовый навык DEKS Shop.
Команды: подбросить монетку, случайный ответ да/нет.
"""
import random
from skills.base_skill import BaseSkill


class TestSkill(BaseSkill):
    def handle(self, command: str) -> str | None:
        cmd = command.lower().strip()

        if self.is_hit(cmd, "монетка"):
            result = random.choice(["Орёл", "Решка"])
            return f"Монетка: {result}"

        if self.is_hit(cmd, "да_нет"):
            result = random.choice([
                "Да, однозначно.",
                "Нет, не советую.",
                "Скорее да.",
                "Скорее нет.",
                "Знаки говорят — да.",
                "Сомнительно.",
            ])
            return result

        return None
