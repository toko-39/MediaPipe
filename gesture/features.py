"""毎フレームの特徴量を計算するモジュール。

計算する特徴量:
- 頭の pitch(上下) / yaw(左右) [度]  ← Face Landmarker の変換行列
- 左右の肘の角度 [度]                ← Pose Landmarker
- 左右の手首位置(正規化画像座標)      ← Pose Landmarker
- 肩幅で正規化した手首-肩距離        ← Pose Landmarker
- 手首の x 座標(正規化)              ← Gesture Recognizer の手ランドマーク
- 手の大きさ(手首-中指付け根距離)    ← Gesture Recognizer の手ランドマーク
- 手のひらの法線ベクトル              ← Gesture Recognizer の手ランドマーク
- ジェスチャーラベル                  ← Gesture Recognizer
"""

import math
import numpy as np


# ───────── 頭の回転 ─────────

def head_rotation(face_result) -> dict:
    """Face Landmarker の変換行列から pitch / yaw / roll を度で返す。

    Returns:
        {"pitch": float, "yaw": float, "roll": float} or None
    """
    if (not face_result
            or not face_result.facial_transformation_matrixes
            or len(face_result.facial_transformation_matrixes) == 0):
        return None

    mat = face_result.facial_transformation_matrixes[0]  # 4x4 ndarray
    # 回転行列の上3x3から Euler 角を計算 (XYZ 順)
    R = mat[:3, :3]

    # pitch (X軸回転), yaw (Y軸回転), roll (Z軸回転)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0

    return {
        "pitch": math.degrees(pitch),
        "yaw": math.degrees(yaw),
        "roll": math.degrees(roll),
    }


# ───────── 肘の角度 ─────────

def _angle_3points(a, b, c) -> float:
    """3点 a-b-c の b における角度を度で返す。各点は (x,y,z)。"""
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos = np.clip(cos, -1.0, 1.0)
    return math.degrees(math.acos(cos))


def elbow_angles(pose_result) -> dict:
    """左右の肘の角度(肩-肘-手首)を度で返す。

    Returns:
        {"left_elbow": float, "right_elbow": float} or None
    """
    if not pose_result or not pose_result.pose_landmarks:
        return None
    lm = pose_result.pose_landmarks[0]
    # Pose landmarks: 11=左肩, 13=左肘, 15=左手首, 12=右肩, 14=右肘, 16=右手首
    left = _angle_3points(lm[11], lm[13], lm[15])
    right = _angle_3points(lm[12], lm[14], lm[16])
    return {"left_elbow": left, "right_elbow": right}


# ───────── 手首の位置（Pose） ─────────

def wrist_positions(pose_result) -> dict:
    """Pose の手首位置(正規化画像座標)と肩幅で正規化した手首-肩距離を返す。

    Returns:
        {
            "left_wrist_x": float, "left_wrist_y": float,
            "right_wrist_x": float, "right_wrist_y": float,
            "left_wrist_shoulder_dist": float,
            "right_wrist_shoulder_dist": float,
            "shoulder_width": float,
        } or None
    """
    if not pose_result or not pose_result.pose_landmarks:
        return None
    lm = pose_result.pose_landmarks[0]

    # 肩幅（正規化座標での距離）
    sw = math.sqrt((lm[11].x - lm[12].x) ** 2 + (lm[11].y - lm[12].y) ** 2)
    if sw < 1e-6:
        sw = 1e-6

    def dist(a, b):
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    return {
        "left_wrist_x": lm[15].x,
        "left_wrist_y": lm[15].y,
        "right_wrist_x": lm[16].x,
        "right_wrist_y": lm[16].y,
        "left_wrist_shoulder_dist": dist(lm[15], lm[11]) / sw,
        "right_wrist_shoulder_dist": dist(lm[16], lm[12]) / sw,
        "shoulder_width": sw,
    }


# ───────── 手の特徴量（Gesture Recognizer） ─────────

def hand_features(gesture_result) -> list[dict]:
    """各検出手について特徴量を返す。

    Returns:
        [
            {
                "handedness": "Left" or "Right",
                "gesture": str,
                "gesture_score": float,
                "wrist_x": float, "wrist_y": float,
                "hand_size": float,  # 手首(0)→中指付け根(9) の距離
                "fingertip_avg_x": float,  # 指先平均x (人差し指・中指・薬指)
                "fingertip_avg_y": float,  # 指先平均y
                "palm_normal_x": float,
                "palm_normal_y": float,
                "palm_normal_z": float,
                # 指さし判定用 (Phase 4以降で追加)
                "index_extended": bool,    # 人差し指が伸びているか
                "middle_extended": bool,   # 中指が伸びているか
                "ring_extended": bool,     # 薬指が伸びているか
                "pinky_extended": bool,    # 小指が伸びているか
                "is_pointing": bool,       # 指さし判定 (人差し指のみ伸展)
                "point_angle": float|None, # 指さし方向の角度 [度]
                "index_dir_x": float,      # 人差し指の方向ベクトルx
                "index_dir_y": float,      # 人差し指の方向ベクトルy
            },
            ...
        ]
    """
    if not gesture_result or not gesture_result.hand_landmarks:
        return []

    results = []
    for i, landmarks in enumerate(gesture_result.hand_landmarks):
        info = {}

        # 左右
        if gesture_result.handedness and i < len(gesture_result.handedness):
            info["handedness"] = gesture_result.handedness[i][0].category_name
        else:
            info["handedness"] = "Unknown"

        # ジェスチャー
        if gesture_result.gestures and i < len(gesture_result.gestures):
            g = gesture_result.gestures[i]
            if g:
                info["gesture"] = g[0].category_name
                info["gesture_score"] = g[0].score
            else:
                info["gesture"] = "None"
                info["gesture_score"] = 0.0
        else:
            info["gesture"] = "None"
            info["gesture_score"] = 0.0

        # 手首位置
        wrist = landmarks[0]
        info["wrist_x"] = wrist.x
        info["wrist_y"] = wrist.y

        # 手の大きさ: 手首(0) → 中指付け根(9) の距離
        mid_base = landmarks[9]
        info["hand_size"] = math.sqrt(
            (wrist.x - mid_base.x) ** 2
            + (wrist.y - mid_base.y) ** 2
            + (wrist.z - mid_base.z) ** 2
        )

        # 指先の x 座標平均（人差し指8, 中指12, 薬指16 の先端）
        # 手首より指先のほうが「いえいえ」の左右振動の振幅が大きい
        fingertip_indices = [8, 12, 16]
        ftx = [landmarks[j].x for j in fingertip_indices]
        fty = [landmarks[j].y for j in fingertip_indices]
        info["fingertip_avg_x"] = sum(ftx) / len(ftx)
        info["fingertip_avg_y"] = sum(fty) / len(fty)

        # 手のひらの法線: 手首(0)→人差し指付け根(5) × 手首(0)→小指付け根(17)
        idx_base = landmarks[5]
        pinky_base = landmarks[17]
        v1 = np.array([
            idx_base.x - wrist.x,
            idx_base.y - wrist.y,
            idx_base.z - wrist.z,
        ])
        v2 = np.array([
            pinky_base.x - wrist.x,
            pinky_base.y - wrist.y,
            pinky_base.z - wrist.z,
        ])
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-8:
            normal = normal / norm_len
        info["palm_normal_x"] = float(normal[0])
        info["palm_normal_y"] = float(normal[1])
        info["palm_normal_z"] = float(normal[2])

        # 指さし判定用: 各指が伸びているか + 人差し指の方向
        # 指の先端と付け根のランドマーク番号:
        #   親指: tip=4, ip=3, mcp=2
        #   人差し指: tip=8, pip=6, mcp=5
        #   中指: tip=12, pip=10, mcp=9
        #   薬指: tip=16, pip=14, mcp=13
        #   小指: tip=20, pip=18, mcp=17
        # 指が曲がっている = tip が pip より手首に近い (y が大きいとは限らないので距離で判定)
        def finger_extended(tip_idx, pip_idx, mcp_idx):
            tip = landmarks[tip_idx]
            pip_ = landmarks[pip_idx]
            mcp = landmarks[mcp_idx]
            # tip-mcp 間の距離 vs pip-mcp 間の距離
            tip_mcp = math.sqrt((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2)
            pip_mcp = math.sqrt((pip_.x - mcp.x)**2 + (pip_.y - mcp.y)**2)
            return tip_mcp > pip_mcp * 1.2

        info["index_extended"] = finger_extended(8, 6, 5)
        info["middle_extended"] = finger_extended(12, 10, 9)
        info["ring_extended"] = finger_extended(16, 14, 13)
        info["pinky_extended"] = finger_extended(20, 18, 17)

        # 人差し指の方向ベクトル: 付け根(5) → 先端(8)
        idx_base = landmarks[5]
        idx_tip = landmarks[8]
        info["index_dir_x"] = idx_tip.x - idx_base.x
        info["index_dir_y"] = idx_tip.y - idx_base.y

        # 指さし判定: 人差し指が伸びていて、他の3本が曲がっている
        info["is_pointing"] = (
            info["index_extended"]
            and not info["middle_extended"]
            and not info["ring_extended"]
            and not info["pinky_extended"]
        )

        # 指さし方向（角度、度）: 右=0°, 下=90°, 左=180°/-180°, 上=-90°
        if info["is_pointing"]:
            info["point_angle"] = math.degrees(
                math.atan2(info["index_dir_y"], info["index_dir_x"])
            )
        else:
            info["point_angle"] = None

        results.append(info)

    return results


# ───────── 全特徴量をまとめて返す ─────────

def compute_all(face_result, pose_result, gesture_result, timestamp_ms: int) -> dict:
    """全特徴量を1つの辞書にまとめて返す。CSV 記録用。"""
    row = {"timestamp_ms": timestamp_ms}

    # 頭の回転
    hr = head_rotation(face_result)
    if hr:
        row["head_pitch"] = round(hr["pitch"], 2)
        row["head_yaw"] = round(hr["yaw"], 2)
        row["head_roll"] = round(hr["roll"], 2)
    else:
        row["head_pitch"] = None
        row["head_yaw"] = None
        row["head_roll"] = None

    # 肘角度
    ea = elbow_angles(pose_result)
    if ea:
        row["left_elbow_angle"] = round(ea["left_elbow"], 2)
        row["right_elbow_angle"] = round(ea["right_elbow"], 2)
    else:
        row["left_elbow_angle"] = None
        row["right_elbow_angle"] = None

    # 手首位置（Pose）
    wp = wrist_positions(pose_result)
    if wp:
        row["left_wrist_x"] = round(wp["left_wrist_x"], 4)
        row["left_wrist_y"] = round(wp["left_wrist_y"], 4)
        row["right_wrist_x"] = round(wp["right_wrist_x"], 4)
        row["right_wrist_y"] = round(wp["right_wrist_y"], 4)
        row["left_wrist_shoulder_dist"] = round(wp["left_wrist_shoulder_dist"], 4)
        row["right_wrist_shoulder_dist"] = round(wp["right_wrist_shoulder_dist"], 4)
        row["shoulder_width"] = round(wp["shoulder_width"], 4)
    else:
        for k in ("left_wrist_x", "left_wrist_y", "right_wrist_x", "right_wrist_y",
                   "left_wrist_shoulder_dist", "right_wrist_shoulder_dist", "shoulder_width"):
            row[k] = None

    # 手の特徴量（最大2手）
    hf = hand_features(gesture_result)
    for side in ("Left", "Right"):
        prefix = side.lower()
        match = [h for h in hf if h["handedness"] == side]
        if match:
            h = match[0]
            row[f"{prefix}_hand_gesture"] = h["gesture"]
            row[f"{prefix}_hand_gesture_score"] = round(h["gesture_score"], 4)
            row[f"{prefix}_hand_wrist_x"] = round(h["wrist_x"], 4)
            row[f"{prefix}_hand_wrist_y"] = round(h["wrist_y"], 4)
            row[f"{prefix}_hand_size"] = round(h["hand_size"], 4)
            row[f"{prefix}_palm_normal_x"] = round(h["palm_normal_x"], 4)
            row[f"{prefix}_palm_normal_y"] = round(h["palm_normal_y"], 4)
            row[f"{prefix}_palm_normal_z"] = round(h["palm_normal_z"], 4)
            row[f"{prefix}_fingertip_avg_x"] = round(h["fingertip_avg_x"], 4)
            row[f"{prefix}_fingertip_avg_y"] = round(h["fingertip_avg_y"], 4)
            row[f"{prefix}_is_pointing"] = h["is_pointing"]
            row[f"{prefix}_point_angle"] = round(h["point_angle"], 2) if h["point_angle"] is not None else None
            row[f"{prefix}_index_dir_x"] = round(h["index_dir_x"], 4)
            row[f"{prefix}_index_dir_y"] = round(h["index_dir_y"], 4)
        else:
            row[f"{prefix}_hand_gesture"] = None
            row[f"{prefix}_hand_gesture_score"] = None
            row[f"{prefix}_hand_wrist_x"] = None
            row[f"{prefix}_hand_wrist_y"] = None
            row[f"{prefix}_hand_size"] = None
            row[f"{prefix}_palm_normal_x"] = None
            row[f"{prefix}_palm_normal_y"] = None
            row[f"{prefix}_palm_normal_z"] = None
            row[f"{prefix}_fingertip_avg_x"] = None
            row[f"{prefix}_fingertip_avg_y"] = None
            row[f"{prefix}_is_pointing"] = None
            row[f"{prefix}_point_angle"] = None
            row[f"{prefix}_index_dir_x"] = None
            row[f"{prefix}_index_dir_y"] = None

    return row
