"""ラベル付き特徴量を CSV に記録するモジュール。

使い方:
    rec = Recorder()
    # 毎フレーム
    rec.add(features_dict)
    # キー操作でラベルを設定
    rec.set_label("nod")
    # 終了時
    rec.save("data/recording_001.csv")
"""

import csv
import os
from datetime import datetime


# ラベル定義（キー → ラベル名）
LABEL_KEYS = {
    ord("1"): "nod",        # 首を縦に振る
    ord("2"): "shake",      # 首を横に振る
    ord("3"): "offer",      # 手を差し出す
    ord("4"): "wave_no",    # 手を横に振るNO
    ord("0"): "none",       # ラベルなし（通常状態）
}


class Recorder:
    def __init__(self):
        self.rows: list[dict] = []
        self.current_label: str = "none"
        self._columns: list[str] | None = None

    def set_label(self, label: str):
        """現在のラベルを設定する。"""
        self.current_label = label

    def add(self, features: dict):
        """1フレーム分の特徴量を追加する。現在のラベルが自動付与される。"""
        row = dict(features)
        row["label"] = self.current_label
        self.rows.append(row)
        if self._columns is None:
            self._columns = list(row.keys())

    def save(self, filepath: str):
        """記録をCSVに保存する。"""
        if not self.rows:
            print("記録データがありません")
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._columns)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"保存完了: {filepath} ({len(self.rows)} フレーム)")

    def auto_filepath(self, base_dir: str = "data") -> str:
        """タイムスタンプ付きのファイルパスを自動生成する。"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(base_dir, f"recording_{ts}.csv")

    @property
    def frame_count(self) -> int:
        return len(self.rows)
