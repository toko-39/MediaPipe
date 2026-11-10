"""首の縦振り（うなずき）・横振りの判定器。

判定ロジック:
  - Face Landmarker の変換行列から算出した pitch / yaw を使用
  - スライディングウィンドウ内で:
    - pitch が min_amplitude 以上の折り返しを min_reversals 回以上
      かつ yaw の range が max_other_range 以下 → うなずき
    - yaw が同条件を満たし pitch の range が小さい → 横振り
"""

from config import COMMON, NOD, SHAKE
from detectors.base import BaseDetector, SignalBuffer


class HeadNodDetector(BaseDetector):
    """首を縦に振る（うなずき）の検出。"""

    def __init__(self):
        super().__init__(
            window_sec=COMMON["window_sec"],
            cooldown_sec=NOD["cooldown_sec"],
            smooth_n=COMMON["smooth_n"],
        )
        self.min_amplitude = NOD["min_amplitude"]
        self.min_reversals = NOD["min_reversals"]
        self.max_yaw_range = NOD["max_yaw_range"]
        self._pitch_buf = SignalBuffer(self.window_sec, self.smooth_n)
        self._yaw_buf = SignalBuffer(self.window_sec, self.smooth_n)

    def update(self, timestamp_ms: int, head_pitch, head_yaw) -> str | None:
        if head_pitch is None or head_yaw is None:
            return None

        self._pitch_buf.add(timestamp_ms, head_pitch)
        self._yaw_buf.add(timestamp_ms, head_yaw)

        if self._in_cooldown(timestamp_ms):
            return None

        reversals = self._pitch_buf.count_reversals(self.min_amplitude)
        yaw_range = self._yaw_buf.get_range()

        if reversals >= self.min_reversals and yaw_range <= self.max_yaw_range:
            self._mark_detected(timestamp_ms)
            return "nod"

        return None


class HeadShakeDetector(BaseDetector):
    """首を横に振るの検出。"""

    def __init__(self):
        super().__init__(
            window_sec=COMMON["window_sec"],
            cooldown_sec=SHAKE["cooldown_sec"],
            smooth_n=COMMON["smooth_n"],
        )
        self.min_amplitude = SHAKE["min_amplitude"]
        self.min_reversals = SHAKE["min_reversals"]
        self.max_pitch_range = SHAKE["max_pitch_range"]
        self._pitch_buf = SignalBuffer(self.window_sec, self.smooth_n)
        self._yaw_buf = SignalBuffer(self.window_sec, self.smooth_n)

    def update(self, timestamp_ms: int, head_pitch, head_yaw) -> str | None:
        if head_pitch is None or head_yaw is None:
            return None

        self._pitch_buf.add(timestamp_ms, head_pitch)
        self._yaw_buf.add(timestamp_ms, head_yaw)

        if self._in_cooldown(timestamp_ms):
            return None

        reversals = self._yaw_buf.count_reversals(self.min_amplitude)
        pitch_range = self._pitch_buf.get_range()

        if reversals >= self.min_reversals and pitch_range <= self.max_pitch_range:
            self._mark_detected(timestamp_ms)
            return "shake"

        return None
