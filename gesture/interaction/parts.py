"""部品リスト管理"""
from __future__ import annotations
import random
from typing import Optional


class PartsManager:
    """部品リストからランダムに候補を提示する"""

    def __init__(self, parts: list[str]):
        self._parts = list(parts)
        self._current: Optional[str] = None
        self.pick_random()

    @property
    def current(self) -> Optional[str]:
        return self._current

    def pick_random(self):
        """ランダムに次の候補を選ぶ"""
        self._current = random.choice(self._parts)

    def select(self, part_name: str) -> bool:
        """指さしで指定された部品を現在の候補に変更。リストに存在すれば True"""
        if part_name in self._parts:
            self._current = part_name
            return True
        return False

    def __repr__(self):
        return f"PartsManager({self.current})"
