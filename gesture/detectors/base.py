"""判定器の共通基底クラスとユーティリティ。"""

from collections import deque


class BaseDetector:
    """全判定器の基底クラス。

    共通処理:
    - スライディングウィンドウ（時刻付きの値を保持）
    - 移動平均による平滑化
    - 折り返し（方向転換）のカウント
    - 検出後のクールダウン
    """

    def __init__(self, window_sec: float = 1.5, cooldown_sec: float = 1.5,
                 smooth_n: int = 5, frame_interval_ms: int = 33):
        self.window_sec = window_sec
        self.cooldown_sec = cooldown_sec
        self.smooth_n = smooth_n
        self.frame_interval_ms = frame_interval_ms

        self._last_detection_ms = -999999

    def _in_cooldown(self, timestamp_ms: int) -> bool:
        return (timestamp_ms - self._last_detection_ms) < self.cooldown_sec * 1000

    def _mark_detected(self, timestamp_ms: int):
        self._last_detection_ms = timestamp_ms


class SignalBuffer:
    """1次元の時系列信号を管理し、平滑化・折り返しカウントを提供する。"""

    def __init__(self, window_sec: float = 1.5, smooth_n: int = 5):
        self.window_sec = window_sec
        self.smooth_n = smooth_n
        self._raw: deque[tuple[int, float]] = deque()  # (timestamp_ms, value)

    def add(self, timestamp_ms: int, value: float):
        """値を追加し、ウィンドウ外の古いデータを削除する。"""
        self._raw.append((timestamp_ms, value))
        cutoff = timestamp_ms - self.window_sec * 1000
        while self._raw and self._raw[0][0] < cutoff:
            self._raw.popleft()

    def get_smoothed(self) -> list[tuple[int, float]]:
        """移動平均で平滑化した値を返す。"""
        raw = list(self._raw)
        if len(raw) < self.smooth_n:
            return raw
        result = []
        for i in range(len(raw)):
            start = max(0, i - self.smooth_n + 1)
            subset = raw[start:i + 1]
            avg = sum(v for _, v in subset) / len(subset)
            result.append((raw[i][0], avg))
        return result

    def count_reversals(self, min_amplitude: float) -> int:
        """平滑化済み信号の折り返し回数を数える。

        min_amplitude: この振幅以上の方向転換のみカウントする。
        """
        smoothed = self.get_smoothed()
        if len(smoothed) < 3:
            return 0

        values = [v for _, v in smoothed]

        # 極値を見つける
        extrema = []
        for i in range(1, len(values) - 1):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                extrema.append(values[i])
            elif values[i] < values[i - 1] and values[i] < values[i + 1]:
                extrema.append(values[i])

        if len(extrema) < 2:
            return 0

        # 十分な振幅の折り返しを数える
        reversals = 0
        for i in range(1, len(extrema)):
            if abs(extrema[i] - extrema[i - 1]) >= min_amplitude:
                reversals += 1

        return reversals

    def get_range(self) -> float:
        """ウィンドウ内の値の範囲(max-min)を返す。"""
        if not self._raw:
            return 0.0
        values = [v for _, v in self._raw]
        return max(values) - min(values)

    def get_std(self) -> float:
        """ウィンドウ内の値の標準偏差を返す。"""
        if len(self._raw) < 2:
            return 0.0
        values = [v for _, v in self._raw]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5

    @property
    def size(self) -> int:
        return len(self._raw)
