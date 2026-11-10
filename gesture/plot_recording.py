"""記録した CSV を読み込み、特徴量の波形をグラフで表示する。

使い方:
    python plot_recording.py data/recording_20261110_123456.csv
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ラベルごとの背景色
LABEL_COLORS = {
    "nod": "#FF000030",
    "shake": "#0000FF30",
    "offer": "#00FF0030",
    "wave_no": "#FF00FF30",
    "none": "#00000000",
}


def shade_labels(ax, df):
    """ラベル区間を背景色で塗る。"""
    labels = df["label"].values
    timestamps = df["timestamp_ms"].values
    prev_label = labels[0]
    start = timestamps[0]

    for i in range(1, len(labels)):
        if labels[i] != prev_label or i == len(labels) - 1:
            end = timestamps[i]
            color = LABEL_COLORS.get(prev_label, "#00000000")
            if prev_label != "none":
                ax.axvspan(start, end, color=color)
            prev_label = labels[i]
            start = timestamps[i]


def main():
    if len(sys.argv) < 2:
        print("使い方: python plot_recording.py <CSVファイル>")
        sys.exit(1)

    filepath = sys.argv[1]
    df = pd.read_csv(filepath)
    t = df["timestamp_ms"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Feature Recording: {filepath}", fontsize=12)

    # 1. 頭の Pitch / Yaw
    ax = axes[0]
    ax.set_title("Head Rotation")
    ax.plot(t, df["head_pitch"], label="Pitch (up/down)", linewidth=0.8)
    ax.plot(t, df["head_yaw"], label="Yaw (left/right)", linewidth=0.8)
    ax.set_ylabel("Degrees")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    shade_labels(ax, df)

    # 2. 肘角度
    ax = axes[1]
    ax.set_title("Elbow Angles")
    ax.plot(t, df["left_elbow_angle"], label="Left Elbow", linewidth=0.8)
    ax.plot(t, df["right_elbow_angle"], label="Right Elbow", linewidth=0.8)
    ax.set_ylabel("Degrees")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    shade_labels(ax, df)

    # 3. 手首の X 座標（Pose）
    ax = axes[2]
    ax.set_title("Wrist X Position (Pose)")
    ax.plot(t, df["left_wrist_x"], label="Left Wrist X", linewidth=0.8)
    ax.plot(t, df["right_wrist_x"], label="Right Wrist X", linewidth=0.8)
    ax.set_ylabel("Normalized X")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    shade_labels(ax, df)

    # 4. 手首の X 座標（Hand Landmark）
    ax = axes[3]
    ax.set_title("Hand Wrist X (Gesture Recognizer)")
    if "left_hand_wrist_x" in df.columns:
        ax.plot(t, df["left_hand_wrist_x"], label="Left Hand X", linewidth=0.8)
    if "right_hand_wrist_x" in df.columns:
        ax.plot(t, df["right_hand_wrist_x"], label="Right Hand X", linewidth=0.8)
    ax.set_ylabel("Normalized X")
    ax.set_xlabel("Timestamp (ms)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    shade_labels(ax, df)

    # 凡例（ラベル色）
    patches = [
        mpatches.Patch(color=c.replace("30", "80"), label=l)
        for l, c in LABEL_COLORS.items() if l != "none"
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               title="Action Labels")

    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.show()


if __name__ == "__main__":
    main()
