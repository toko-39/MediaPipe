"""手を差し出す動作の判定器。

判定ロジック:
  - 掌が上向き（手のひらの法線ベクトルの y 成分で判定）
  - 掌が上向きの状態で一定時間静止している
"""

from collections import deque
from config import COMMON, OFFER
from detectors.base import BaseDetector


class OfferDetector(BaseDetector):
    """手を差し出す動作の検出（掌上向き+静止）。"""

    def __init__(self):
        super().__init__(
            window_sec=COMMON["window_sec"],
            cooldown_sec=OFFER["cooldown_sec"],
            smooth_n=COMMON["smooth_n"],
        )
        self.palm_up_threshold = OFFER["palm_up_threshold"]
        self.hold_duration_ms = OFFER["hold_duration_ms"]
        self.still_tolerance = OFFER["still_tolerance"]

        self._left_history: deque[tuple[int, float, float, bool]] = deque(maxlen=90)
        self._right_history: deque[tuple[int, float, float, bool]] = deque(maxlen=90)

    def update(self, timestamp_ms: int, features: dict) -> str | None:
        self._record(timestamp_ms, features)

        if self._in_cooldown(timestamp_ms):
            return None

        for _, history in [("left", self._left_history),
                           ("right", self._right_history)]:
            if self._check_offer(timestamp_ms, history):
                self._mark_detected(timestamp_ms)
                return "offer"

        return None

    def _record(self, timestamp_ms: int, features: dict):
        for side, history in [("left", self._left_history),
                              ("right", self._right_history)]:
            normal_y = features.get(f"{side}_palm_normal_y")
            wrist_x = features.get(f"{side}_hand_wrist_x")
            wrist_y = features.get(f"{side}_hand_wrist_y")

            if normal_y is None or wrist_x is None or wrist_y is None:
                continue

            palm_up = abs(normal_y) > self.palm_up_threshold
            history.append((timestamp_ms, wrist_x, wrist_y, palm_up))

    def _check_offer(self, timestamp_ms: int,
                     history: deque[tuple[int, float, float, bool]]) -> bool:
        recent = [(ts, wx, wy, pu) for ts, wx, wy, pu in history
                  if ts >= timestamp_ms - self.hold_duration_ms]

        if len(recent) < 5:
            return False

        if not all(pu for _, _, _, pu in recent):
            return False

        xs = [wx for _, wx, _, _ in recent]
        ys = [wy for _, _, wy, _ in recent]
        if max(xs) - min(xs) > self.still_tolerance:
            return False
        if max(ys) - min(ys) > self.still_tolerance:
            return False

        return True
