"""記録済み CSV に対して判定器をオフライン実行し、精度を確認する。

使い方:
    python verify_detectors.py data/recording_20261110_173028.csv
"""

import sys
import pandas as pd
from detectors import HeadNodDetector, HeadShakeDetector, WaveNoDetector, OfferDetector


def main():
    if len(sys.argv) < 2:
        print("使い方: python verify_detectors.py <CSVファイル>")
        sys.exit(1)

    filepath = sys.argv[1]
    df = pd.read_csv(filepath)
    print(f"読み込み: {filepath} ({len(df)} フレーム)")
    print(f"ラベル分布:\n{df['label'].value_counts().to_string()}\n")

    # 判定器を生成
    nod_det = HeadNodDetector()
    shake_det = HeadShakeDetector()
    wave_det = WaveNoDetector()
    offer_det = OfferDetector()

    # フレームごとに実行
    detections = []  # (timestamp_ms, detected_label, true_label)
    for _, row in df.iterrows():
        ts = int(row["timestamp_ms"])
        true_label = row["label"]

        pitch = row.get("head_pitch")
        yaw = row.get("head_yaw")
        if pd.isna(pitch):
            pitch = None
        if pd.isna(yaw):
            yaw = None

        feats = row.to_dict()
        # NaN → None
        for k, v in feats.items():
            if pd.isna(v):
                feats[k] = None

        det_nod = nod_det.update(ts, pitch, yaw)
        det_shake = shake_det.update(ts, pitch, yaw)
        det_wave = wave_det.update(ts, feats)
        det_offer = offer_det.update(ts, feats)

        for det in (det_nod, det_shake, det_wave, det_offer):
            if det:
                detections.append((ts, det, true_label))

    # 結果表示
    print("=" * 60)
    print("検出結果:")
    print("=" * 60)
    if not detections:
        print("  検出なし")
    else:
        correct = 0
        for ts, detected, true_label in detections:
            match = "OK" if detected == true_label else "MISS"
            if detected == true_label:
                correct += 1
            print(f"  t={ts:>6}ms  detected={detected:<10} true={true_label:<10} {match}")

        print()
        print(f"検出数: {len(detections)}")
        print(f"正解数: {correct}")
        print(f"精度:   {correct / len(detections) * 100:.1f}%")

    # ラベルごとの検出/未検出
    print()
    print("ラベルごとの検出状況:")
    for label in ["nod", "shake", "wave_no", "offer"]:
        total_frames = len(df[df["label"] == label])
        detected_count = sum(1 for _, d, _ in detections if d == label)
        false_pos = sum(1 for _, d, t in detections if d == label and t != label)
        print(f"  {label:<10}: 対象{total_frames:>4}f, 検出{detected_count}回, 誤検出{false_pos}回")


if __name__ == "__main__":
    main()
