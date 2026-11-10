"""指さしの判定器。

判定ロジック:
  - 人差し指だけが伸びている（中指・薬指・小指は曲がっている）
  - その状態で一定時間(300ms)静止している
  - 人差し指の付け根→先端のベクトルから方向（左・右・上・下）を判定
"""

import math
from collections import deque
from detectors.base import BaseDetector


def _angle_to_direction(angle: float) -> str:
    """角度（度）を方向ラベルに変換する。

    画像座標系: 右=0°, 下=90°, 左=±180°, 上=-90°
    カメラは鏡像なので左右を反転する。
    """
    if -45 <= angle < 45:
        return "左"    # 画像上の右 = 実際の左
    elif 45 <= angle < 135:
        return "下"
    elif -135 <= angle < -45:
        return "上"
    else:
        return "右"   # 画像上の左 = 実際の右


class PointDetector(BaseDetector):
    """指さしの検出と方向判定。"""

    def __init__(self, hold_duration_ms: int = 300,
                 angle_tolerance: float = 30.0,
                 **kwargs):
        super().__init__(cooldown_sec=1.5, **kwargs)
        self.hold_duration_ms = hold_duration_ms
        self.angle_tolerance = angle_tolerance  # 静止中の角度変動の許容（度）

        # (timestamp_ms, is_pointing, angle)
        self._left_history: deque[tuple[int, bool, float | None]] = deque(maxlen=90)
        self._right_history: deque[tuple[int, bool, float | None]] = deque(maxlen=90)

    def update(self, timestamp_ms: int, features: dict) -> tuple[str, str] | None:
        """特徴量辞書を受け取り、検出時に ("point", 方向) を返す。"""
        self._record(timestamp_ms, features)

        if self._in_cooldown(timestamp_ms):
            return None

        for _, history in [("left", self._left_history),
                           ("right", self._right_history)]:
            result = self._check_point(timestamp_ms, history)
            if result:
                self._mark_detected(timestamp_ms)
                return result

        return None

    def _record(self, timestamp_ms: int, features: dict):
        for side, history in [("left", self._left_history),
                              ("right", self._right_history)]:
            is_pointing = features.get(f"{side}_is_pointing")
            angle = features.get(f"{side}_point_angle")

            if is_pointing is None:
                continue

            history.append((timestamp_ms, bool(is_pointing), angle))

    def _check_point(self, timestamp_ms: int,
                     history: deque[tuple[int, bool, float | None]]) -> tuple[str, str] | None:
        recent = [(ts, pt, ang) for ts, pt, ang in history
                  if ts >= timestamp_ms - self.hold_duration_ms]

        if len(recent) < 5:
            return None

        # 全フレームで指さし状態
        if not all(pt for _, pt, _ in recent):
            return None

        # 角度が全て有効
        angles = [ang for _, _, ang in recent if ang is not None]
        if len(angles) < 5:
            return None

        # 方向が安定しているか（角度の変動が小さい）
        angle_range = max(angles) - min(angles)
        # 180°/-180° をまたぐ場合の補正
        if angle_range > 180:
            adjusted = [a if a >= 0 else a + 360 for a in angles]
            angle_range = max(adjusted) - min(adjusted)

        if angle_range > self.angle_tolerance:
            return None

        # 平均角度から方向を判定
        avg_angle = sum(angles) / len(angles)
        direction = _angle_to_direction(avg_angle)

        return ("point", direction)
