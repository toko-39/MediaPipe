"""シグナル定義と SignalBus (pub/sub)"""
from __future__ import annotations
from enum import Enum, auto
from typing import Any, Callable
import time


class SignalKind(Enum):
    # 合図 (onset 受信時)
    LISTENING = auto()       # WANT onset → 「聞いています」
    WATCHING = auto()        # YES/NO onset → 「見ています」

    # 動作指示
    ACCEPT = auto()          # 受諾動作を開始
    ACK = auto()             # 了解動作を開始
    HANDOVER = auto()        # 受け渡し動作を開始

    # 通知
    ATTENTION = auto()       # 注意喚起
    OFFER_PART = auto()      # 部品を提示 (payload=部品名)
    CONFIRM_PART = auto()    # 復唱確認 (payload=部品名)
    FLOW_RESET = auto()      # フローリセット (IDLE に戻った)
    FLOW_COMPLETE = auto()   # 全部品完了


class Signal:
    __slots__ = ("kind", "payload", "timestamp")

    def __init__(self, kind: SignalKind, payload: Any = None):
        self.kind = kind
        self.payload = payload
        self.timestamp = time.time()

    def __repr__(self):
        s = f"Signal({self.kind.name}"
        if self.payload is not None:
            s += f", {self.payload}"
        return s + ")"


class SignalBus:
    """簡易 pub/sub バス"""

    def __init__(self):
        self._subscribers: list[Callable[[Signal], None]] = []

    def subscribe(self, callback: Callable[[Signal], None]):
        self._subscribers.append(callback)

    def emit(self, signal: Signal):
        for cb in self._subscribers:
            cb(signal)
