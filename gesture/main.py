"""カメラ映像に対して Face / Pose / Gesture の3タスクを同時実行し、
ランドマーク描画・FPS 表示・特徴量計算・動作検出・記録を行う。

操作:
  q     : 終了
  r     : 記録の開始/停止トグル
  1     : ラベル「nod」(うなずき) に切り替え
  2     : ラベル「shake」(首横振り) に切り替え
  3     : ラベル「offer」(手を差し出す) に切り替え
  4     : ラベル「wave_no」(手を横に振るNO) に切り替え
  0     : ラベル「none」(通常) に切り替え
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    drawing_utils,
    drawing_styles,
    FaceLandmarksConnections,
    PoseLandmarksConnections,
    HandLandmarksConnections,
)

from trackers import (
    create_face_landmarker,
    create_pose_landmarker,
    create_gesture_recognizer,
)
from features import compute_all, head_rotation
from recorder import Recorder, LABEL_KEYS
from detectors import HeadNodDetector, HeadShakeDetector, WaveNoDetector, OfferDetector

# ───────── 描画ユーティリティ ─────────


def draw_face(frame, face_result):
    if not face_result or not face_result.face_landmarks:
        return
    for landmarks in face_result.face_landmarks:
        drawing_utils.draw_landmarks(
            frame, landmarks,
            FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
            is_drawing_landmarks=False,
        )
        drawing_utils.draw_landmarks(
            frame, landmarks,
            FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
            is_drawing_landmarks=False,
        )


def draw_pose(frame, pose_result):
    if not pose_result or not pose_result.pose_landmarks:
        return
    for landmarks in pose_result.pose_landmarks:
        drawing_utils.draw_landmarks(
            frame, landmarks,
            PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
        )


def draw_hands(frame, gesture_result):
    if not gesture_result or not gesture_result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    for i, landmarks in enumerate(gesture_result.hand_landmarks):
        drawing_utils.draw_landmarks(
            frame, landmarks,
            HandLandmarksConnections.HAND_CONNECTIONS,
            landmark_drawing_spec=drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=drawing_styles.get_default_hand_connections_style(),
        )
        if gesture_result.gestures and i < len(gesture_result.gestures):
            gesture = gesture_result.gestures[i]
            if gesture:
                label = gesture[0].category_name
                score = gesture[0].score
                wrist = landmarks[0]
                cx, cy = int(wrist.x * w), int(wrist.y * h) - 20
                cv2.putText(
                    frame, f"{label} ({score:.2f})", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                )
        if gesture_result.handedness and i < len(gesture_result.handedness):
            hand_label = gesture_result.handedness[i][0].category_name
            wrist = landmarks[0]
            cx, cy = int(wrist.x * w), int(wrist.y * h) + 20
            cv2.putText(
                frame, hand_label, (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
            )


# ───────── 検出結果の表示 ─────────

# 検出時に画面中央に表示するテキストと色 (BGR)
_DETECTION_DISPLAY = {
    "nod":     ("はい",     (0, 255, 0)),
    "shake":   ("いいえ",   (255, 100, 0)),
    "wave_no": ("戻る",     (255, 0, 255)),
    "offer":   ("ください", (0, 200, 255)),
}
_DISPLAY_DURATION_MS = 1500  # 検出表示の持続時間

# 日本語描画用フォント (Pillow)
from PIL import ImageFont, ImageDraw, Image as PILImage
import os

def _load_jp_font(size: int = 64):
    """Windows の日本語太字フォントを読み込む。"""
    font_candidates = [
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", f)
        # メイリオ太字 → メイリオ → ゴシック → 游ゴシック
        for f in ("meiryob.ttc", "meiryo.ttc", "msgothic.ttc", "YuGothB.ttc")
    ]
    for path in font_candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

_JP_FONT = _load_jp_font(64)


def draw_jp_text(frame, text: str, pos: tuple, color_bgr: tuple):
    """OpenCV の frame に日本語テキストを描画する。"""
    img_pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(pos, text, font=_JP_FONT, fill=color_rgb)
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ───────── メインループ ─────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした")
        return

    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"カメラ解像度: {cap_w}x{cap_h}")

    print("モデルを読み込み中...")
    face_landmarker = create_face_landmarker()
    pose_landmarker = create_pose_landmarker()
    gesture_recognizer = create_gesture_recognizer()
    print("モデル読み込み完了")

    # 判定器
    nod_det = HeadNodDetector()
    shake_det = HeadShakeDetector()
    wave_det = WaveNoDetector()
    offer_det = OfferDetector()

    # 記録
    recorder = Recorder()
    recording = False

    # 検出結果表示用
    last_detection = None
    last_detection_ms = 0

    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    timestamp_ms = 0

    print("実行中...")
    print("  q: 終了  r: 記録開始/停止")
    print("  1: nod  2: shake  3: offer  4: wave_no  0: none")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += 33

            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            gesture_result = gesture_recognizer.recognize_for_video(
                mp_image, timestamp_ms
            )

            # 描画
            draw_face(frame, face_result)
            draw_pose(frame, pose_result)
            draw_hands(frame, gesture_result)

            # 特徴量計算
            feats = compute_all(face_result, pose_result, gesture_result, timestamp_ms)

            # 記録中なら追加
            if recording:
                recorder.add(feats)

            # ───── 動作検出 ─────
            hr = head_rotation(face_result)
            pitch = hr["pitch"] if hr else None
            yaw = hr["yaw"] if hr else None

            det_nod = nod_det.update(timestamp_ms, pitch, yaw)
            det_shake = shake_det.update(timestamp_ms, pitch, yaw)
            det_wave = wave_det.update(timestamp_ms, feats)
            det_offer = offer_det.update(timestamp_ms, feats)

            # 優先度: 手の動作 > 首の動作
            if det_wave:
                det_shake = None
            if det_offer:
                det_nod = None

            for det in (det_offer, det_wave, det_nod, det_shake):
                if det:
                    last_detection = det
                    last_detection_ms = timestamp_ms
                    print(f"  >> 検出: {det}")

            # FPS 計算
            frame_count += 1
            now = time.time()
            elapsed = now - prev_time
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                prev_time = now

            # ───── 情報表示 ─────
            # 左上: FPS・検出状況
            info_lines = [f"FPS: {fps:.1f}"]
            if face_result and face_result.face_landmarks:
                info_lines.append(f"Face: {len(face_result.face_landmarks)}")
            else:
                info_lines.append("Face: -")
            if pose_result and pose_result.pose_landmarks:
                info_lines.append(f"Pose: {len(pose_result.pose_landmarks)}")
            else:
                info_lines.append("Pose: -")
            if gesture_result and gesture_result.hand_landmarks:
                info_lines.append(f"Hands: {len(gesture_result.hand_landmarks)}")
            else:
                info_lines.append("Hands: -")
            base_count = len(info_lines)  # ここまで黄緑
            if hr:
                info_lines.append(f"Pitch: {hr['pitch']:+.1f}")
                info_lines.append(f"Yaw: {hr['yaw']:+.1f}")

            for i, line in enumerate(info_lines):
                color = (0, 255, 0) if i < base_count else (255, 105, 180)
                cv2.putText(
                    frame, line, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                )

            # 右下: 検出結果（日本語、背景なし）
            if (last_detection
                    and (timestamp_ms - last_detection_ms) < _DISPLAY_DURATION_MS):
                text, color = _DETECTION_DISPLAY.get(
                    last_detection, (last_detection, (255, 255, 255))
                )
                bbox = _JP_FONT.getbbox(text)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                tx = cap_w - tw - 20
                ty = cap_h - th - 60
                draw_jp_text(frame, text, (tx, ty), color)

            # 下部: 記録状態
            if recording:
                status = f"REC [{recorder.current_label}] ({recorder.frame_count} frames)"
                cv2.putText(
                    frame, status, (10, cap_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )
            else:
                cv2.putText(
                    frame, "STANDBY (press 'r' to record)", (10, cap_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
                )

            cv2.imshow("Gesture Recognition - Phase 3", frame)

            # キー入力
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                if not recording:
                    recording = True
                    print(f"記録開始 (ラベル: {recorder.current_label})")
                else:
                    recording = False
                    filepath = recorder.auto_filepath()
                    recorder.save(filepath)
                    recorder = Recorder()
            elif key in LABEL_KEYS:
                label = LABEL_KEYS[key]
                recorder.set_label(label)
                print(f"ラベル変更: {label}")

    finally:
        face_landmarker.close()
        pose_landmarker.close()
        gesture_recognizer.close()
        cap.release()
        cv2.destroyAllWindows()
        print("終了しました")


if __name__ == "__main__":
    main()
