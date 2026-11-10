"""判定器の閾値パラメータを一元管理する設定ファイル。

コードを変更せず、ここの数値を調整するだけでチューニングできる。
"""

# ───────── 共通 ─────────
COMMON = {
    "window_sec": 1.5,       # スライディングウィンドウ幅(秒)
    "smooth_n": 5,           # 移動平均のフレーム数
}

# ───────── うなずき (HeadNodDetector) ─────────
NOD = {
    "min_amplitude": 6.0,    # pitch の折り返し最小振幅(度)
    "min_reversals": 3,      # 必要な折り返し回数
    "max_yaw_range": 12.0,   # yaw の許容 range(度)。これ以下なら横振りでない
    "cooldown_sec": 1.5,
}

# ───────── 首横振り (HeadShakeDetector) ─────────
SHAKE = {
    "min_amplitude": 6.0,    # yaw の折り返し最小振幅(度)
    "min_reversals": 3,      # 必要な折り返し回数
    "max_pitch_range": 8.0,  # pitch の許容 range(度)。これ以下なら縦振りでない
    "cooldown_sec": 1.5,
}

# ───────── 手を横に振るNO (WaveNoDetector) ─────────
WAVE_NO = {
    "min_amplitude": 0.3,    # 手の大きさで正規化後の指先x振幅
    "min_reversals": 3,      # 必要な折り返し回数(2往復分)
    "max_wrist_y": 0.85,     # 指先y上限(これより下=腰より下は除外)
    "min_wrist_y": 0.2,      # 指先y下限(これより上=頭より上は除外=バイバイ圏)
    "palm_up_reject": 0.5,   # |palm_normal_y|がこれ以上なら掌上向き→除外
    "cooldown_sec": 1.5,
}

# ───────── 手を差し出す (OfferDetector) ─────────
OFFER = {
    "palm_up_threshold": 0.4,  # |palm_normal_y| の閾値。掌が上向き判定
    "hold_duration_ms": 300,   # 静止を要求する時間(ms)
    "still_tolerance": 0.06,   # 手首位置の変動許容(正規化座標)
    "cooldown_sec": 2.0,
}
