"""Stage 2 (I-2): カメラ入力によるインタラクションフロー実行

検出結果 → イベント変換:
  offer   → WANT
  nod     → YES
  shake   → NO
  wave_no → CANCEL

操作:
  q     : 終了
  1-5   : POINT (P1〜P5) ※指さしは未実装のためキーボード模擬
  スペース: step_mode 時のみ、模擬動作を手動完了
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
from detectors import HeadNodDetector, HeadShakeDetector, WaveNoDetector, OfferDetector

from interaction.events import Event, EventKind, Stage
from interaction.signals import SignalBus, SignalKind
from interaction.sinks import ConsoleSink, JsonlSink
from interaction.state_machine import StateMachine, State

# ───────── 検出→イベント変換 ─────────

_DETECTION_TO_EVENT = {
    "offer":   EventKind.WANT,
    "nod":     EventKind.YES,
    "shake":   EventKind.NO,
    "wave_no": EventKind.CANCEL,
}

# ───────── 表示設定 ─────────

_STATE_DISPLAY = {
    State.IDLE:                ("待機中",       (200, 200, 200)),
    State.REQ_DETECTING:       ("要求検出中",   (0, 255, 255)),
    State.ACCEPTING:           ("受諾中...",     (0, 200, 255)),
    State.WAIT_REACTION:       ("反応待ち",     (0, 255, 0)),
    State.REACTION_DETECTING:  ("反応検出中",   (0, 255, 255)),
    State.WAIT_POINTING:       ("指さし待ち",   (255, 200, 0)),
    State.CONFIRMING:          ("確認中",       (255, 200, 0)),
    State.ACKING:              ("了解中...",     (0, 200, 255)),
    State.HANDING_OVER:        ("受け渡し中...", (0, 200, 255)),
}

_DETECTION_DISPLAY = {
    "nod":     ("はい",     (0, 255, 0)),
    "shake":   ("いいえ",   (255, 100, 0)),
    "wave_no": ("戻る",     (255, 0, 255)),
    "offer":   ("ください", (0, 200, 255)),
}
_DISPLAY_DURATION_MS = 1500

# 日本語描画用フォント (Pillow)
from PIL import ImageFont, ImageDraw, Image as PILImage

def _load_jp_font(size: int = 64):
    font_candidates = [
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", f)
        for f in ("meiryob.ttc", "meiryo.ttc", "msgothic.ttc", "YuGothB.ttc")
    ]
    for path in font_candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

_JP_FONT_L = _load_jp_font(64)
_JP_FONT_S = _load_jp_font(20)


def draw_jp_text(frame, text, pos, color_bgr, font=None):
    if font is None:
        font = _JP_FONT_L
    img_pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(pos, text, font=font, fill=color_rgb)
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ───────── 描画 ─────────

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


# ───────── POINT キーマッピング ─────────

_POINT_KEYS = {
    ord("1"): "P1", ord("2"): "P2", ord("3"): "P3",
    ord("4"): "P4", ord("5"): "P5",
}

# ───────── メイン ─────────

def main():
    # step_mode: コマンドライン引数 --step で有効化
    step_mode = "--step" in sys.argv

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    bus = SignalBus()
    console = ConsoleSink()
    jsonl = JsonlSink(os.path.join(os.path.dirname(__file__), "logs"))
    bus.subscribe(console)
    bus.subscribe(jsonl)

    sm = StateMachine(config_path, bus, step_mode=step_mode)

    # カメラ
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

    # 検出結果表示用
    last_detection = None
    last_detection_ms = 0

    prev_time = time.time()
    fps = 0.0
    frame_count = 0
    timestamp_ms = 0

    mode_label = "STEP" if step_mode else "AUTO"
    print(f"実行中... (モード: {mode_label})")
    print("  q: 終了  1-5: POINT")
    if step_mode:
        print("  スペース: 模擬動作を手動完了")

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

            # ───── 検出→イベント→状態機械 ─────
            for det in (det_offer, det_wave, det_nod, det_shake):
                if det:
                    last_detection = det
                    last_detection_ms = timestamp_ms
                    event_kind = _DETECTION_TO_EVENT.get(det)
                    if event_kind:
                        ev = Event(event_kind, Stage.CONFIRM)
                        sm.handle(ev)
                        print(f"  >> 検出: {det} → {ev}")

            # FPS 計算
            frame_count += 1
            now = time.time()
            elapsed = now - prev_time
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                prev_time = now

            # ───── 情報表示 ─────
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
            base_count = len(info_lines)
            if hr:
                info_lines.append(f"Pitch: {hr['pitch']:+.1f}")
                info_lines.append(f"Yaw: {hr['yaw']:+.1f}")

            for i, line in enumerate(info_lines):
                color = (0, 255, 0) if i < base_count else (255, 105, 180)
                cv2.putText(
                    frame, line, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                )

            # ───── フロー状態表示 (左下) ─────
            state_text, state_color = _STATE_DISPLAY.get(
                sm.state, (sm.state.name, (255, 255, 255)))
            part_text = f"部品: {sm.parts.current}" if sm.parts.current else ""

            # step_mode の保留イベントヒント
            pending_text = ""
            if step_mode and sm.pending:
                pending_text = f"  [Space→{sm.pending.kind.name}]"

            draw_jp_text(frame, f"{state_text}  {part_text}{pending_text}",
                         (10, cap_h - 100), state_color, _JP_FONT_S)

            # 右下: 検出結果
            if (last_detection
                    and (timestamp_ms - last_detection_ms) < _DISPLAY_DURATION_MS):
                text, color = _DETECTION_DISPLAY.get(
                    last_detection, (last_detection, (255, 255, 255)))
                bbox = _JP_FONT_L.getbbox(text)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                tx = cap_w - tw - 20
                ty = cap_h - th - 60
                draw_jp_text(frame, text, (tx, ty), color)

            cv2.imshow("Interaction Flow", frame)

            # キー入力
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" ") and step_mode and sm.pending:
                print(f"  >> step: {sm.pending}")
                sm.step()
            elif key in _POINT_KEYS:
                part = _POINT_KEYS[key]
                ev = Event(EventKind.POINT, Stage.CONFIRM, part)
                sm.handle(ev)
                print(f"  >> POINT: {part}")

    finally:
        face_landmarker.close()
        pose_landmarker.close()
        gesture_recognizer.close()
        cap.release()
        cv2.destroyAllWindows()
        jsonl.close()
        print("終了しました")


if __name__ == "__main__":
    main()
