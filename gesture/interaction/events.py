"""イベント定義 — ジェスチャー/キーボードから状態機械への入力"""
from enum import Enum, auto


class EventKind(Enum):
    WANT = auto()
    YES = auto()
    NO = auto()
    CANCEL = auto()
    POINT = auto()
    ACTION_DONE = auto()
    HANDOVER_DONE = auto()
    TIMEOUT = auto()


class Stage(Enum):
    ONSET = auto()
    CONFIRM = auto()


class Event:
    __slots__ = ("kind", "stage", "payload")

    def __init__(self, kind: EventKind, stage: Stage = Stage.CONFIRM, payload=None):
        self.kind = kind
        self.stage = stage
        self.payload = payload

    def __repr__(self):
        s = f"Event({self.kind.name}, {self.stage.name}"
        if self.payload is not None:
            s += f", {self.payload}"
        return s + ")"
