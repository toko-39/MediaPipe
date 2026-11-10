"""シグナルの出力先 (Sink)"""
from __future__ import annotations
import json
import time
from pathlib import Path

from interaction.signals import Signal


class ConsoleSink:
    """シグナルをコンソールに表示"""

    def __call__(self, signal: Signal):
        ts = time.strftime("%H:%M:%S", time.localtime(signal.timestamp))
        payload = f"  ({signal.payload})" if signal.payload is not None else ""
        print(f"[{ts}] {signal.kind.name}{payload}")


class JsonlSink:
    """シグナルを JSONL ファイルに追記"""

    def __init__(self, log_dir: str = "logs"):
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._path = self._dir / f"signals_{ts}.jsonl"
        self._file = open(self._path, "a", encoding="utf-8")

    def __call__(self, signal: Signal):
        record = {
            "time": signal.timestamp,
            "kind": signal.kind.name,
            "payload": signal.payload,
        }
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self):
        self._file.close()
