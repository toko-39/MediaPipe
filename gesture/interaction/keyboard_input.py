"""キーボード入力 → Event 変換 (Stage 1 模擬用)"""
from interaction.events import Event, EventKind, Stage

# キーマッピング
#   w = WANT (onset → confirm を2段階模擬: 小文字=onset, 大文字=confirm)
#   y = YES
#   n = NO
#   x = CANCEL
#   1-5 = POINT (P1〜P5)
#
# Stage 1 では簡易化: 1回押し = confirm (onset スキップ)
# Shift+キー (W, Y, N) = onset のみ

_CONFIRM_MAP = {
    "w": EventKind.WANT,
    "y": EventKind.YES,
    "n": EventKind.NO,
    "x": EventKind.CANCEL,
}

_ONSET_MAP = {
    "W": EventKind.WANT,
    "Y": EventKind.YES,
    "N": EventKind.NO,
}

_POINT_MAP = {
    "1": "P1",
    "2": "P2",
    "3": "P3",
    "4": "P4",
    "5": "P5",
}


def key_to_event(key: str) -> Event | None:
    """キー文字列を Event に変換。該当しなければ None"""
    if key in _CONFIRM_MAP:
        return Event(_CONFIRM_MAP[key], Stage.CONFIRM)
    if key in _ONSET_MAP:
        return Event(_ONSET_MAP[key], Stage.ONSET)
    if key in _POINT_MAP:
        return Event(EventKind.POINT, Stage.CONFIRM, _POINT_MAP[key])
    return None
