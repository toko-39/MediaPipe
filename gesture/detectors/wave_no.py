"""手を横に振る「NO」（日本式の「いえいえ」）の判定器。

判定ロジック:
  - Gesture Recognizer の指先(人差し指・中指・薬指)の平均 x 座標を使用
  - 指先 x を手の大きさで正規化し、カメラ距離の影響を除去
  - 正規化後の x が短時間に左右に往復（折り返し3回以上）
  - 手の位置が顔〜胸の高さ（バイバイ除外）
  - 掌が上向きの場合は除外（offer との区別）
"""

from config import COMMON, WAVE_NO
from detectors.base import BaseDetector, SignalBuffer


class WaveNoDetector(BaseDetector):
    """手を横に振るNOの検出。"""

    def __init__(self):
        super().__init__(
            window_sec=COMMON["window_sec"],
            cooldown_sec=WAVE_NO["cooldown_sec"],
            smooth_n=COMMON["smooth_n"],
        )
        self.min_amplitude = WAVE_NO["min_amplitude"]
        self.min_reversals = WAVE_NO["min_reversals"]
        self.max_wrist_y = WAVE_NO["max_wrist_y"]
        self.min_wrist_y = WAVE_NO["min_wrist_y"]
        self.palm_up_reject = WAVE_NO["palm_up_reject"]
        self._left_buf = SignalBuffer(self.window_sec, self.smooth_n)
        self._right_buf = SignalBuffer(self.window_sec, self.smooth_n)

    def update(self, timestamp_ms: int, features: dict) -> str | None:
        self._add_to_buffers(timestamp_ms, features)

        if self._in_cooldown(timestamp_ms):
            return None

        for side in ("left", "right"):
            buf = self._left_buf if side == "left" else self._right_buf
            fingertip_y = features.get(f"{side}_fingertip_avg_y")

            if buf.size < 8 or fingertip_y is None:
                continue

            if fingertip_y < self.min_wrist_y or fingertip_y > self.max_wrist_y:
                continue

            palm_normal_y = features.get(f"{side}_palm_normal_y")
            if (palm_normal_y is not None
                    and abs(palm_normal_y) > self.palm_up_reject):
                continue

            reversals = buf.count_reversals(self.min_amplitude)
            if reversals >= self.min_reversals:
                self._mark_detected(timestamp_ms)
                return "wave_no"

        return None

    def _add_to_buffers(self, timestamp_ms: int, features: dict):
        for side, buf in [("left", self._left_buf), ("right", self._right_buf)]:
            fingertip_x = features.get(f"{side}_fingertip_avg_x")
            hand_size = features.get(f"{side}_hand_size")

            if (fingertip_x is not None
                    and hand_size is not None
                    and hand_size > 0.01):
                normalized_x = fingertip_x / hand_size
                buf.add(timestamp_ms, normalized_x)
