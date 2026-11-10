"""模擬動作完了タイマー — ACTION_DONE / HANDOVER_DONE を一定時間後に発火"""
from __future__ import annotations
import threading
from interaction.events import Event, EventKind, Stage


class SimulatedCompletion:
    """指定秒後に event_kind の Event をコールバックで返す"""

    def __init__(self):
        self._timer: threading.Timer | None = None

    def start(self, duration: float, event_kind: EventKind,
              callback):
        self.cancel()
        self._timer = threading.Timer(
            duration,
            lambda: callback(Event(event_kind, Stage.CONFIRM)),
        )
        self._timer.daemon = True
        self._timer.start()

    def cancel(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
