"""MediaPipe Tasks API のラッパー（Face / Pose / Gesture の3タスク）

NOTE: プロジェクトパスに日本語が含まれるため、model_asset_path ではなく
model_asset_buffer（バイト列）でモデルを読み込む。
"""

import os
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    GestureRecognizer,
    GestureRecognizerOptions,
    RunningMode,
)

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def _load_model(filename: str) -> bytes:
    """モデルファイルをバイト列として読み込む。"""
    path = os.path.join(_MODELS_DIR, filename)
    with open(path, "rb") as f:
        return f.read()


def create_face_landmarker() -> FaceLandmarker:
    """Face Landmarker を VIDEO モードで生成する。"""
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_buffer=_load_model("face_landmarker.task")
        ),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
    )
    return FaceLandmarker.create_from_options(options)


def create_pose_landmarker() -> PoseLandmarker:
    """Pose Landmarker (lite) を VIDEO モードで生成する。"""
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_buffer=_load_model("pose_landmarker_lite.task")
        ),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )
    return PoseLandmarker.create_from_options(options)


def create_gesture_recognizer() -> GestureRecognizer:
    """Gesture Recognizer を VIDEO モードで生成する。"""
    options = GestureRecognizerOptions(
        base_options=BaseOptions(
            model_asset_buffer=_load_model("gesture_recognizer.task")
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    )
    return GestureRecognizer.create_from_options(options)
