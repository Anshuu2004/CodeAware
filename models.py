"""
Wraps the four OpenVINO IRs we use into thin loader/inference helpers.

Models (all from Open Model Zoo):
    - face-detection-adas-0001          672x384 BGR  -> bboxes
    - facial-landmarks-35-adas-0002     60x60   BGR  -> 35 (x,y) points
    - head-pose-estimation-adas-0001    60x60   BGR  -> yaw / pitch / roll
    - open-closed-eye-0001              32x32   BGR  -> [open, closed] softmax
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from openvino.runtime import Core


# ---------------------------------------------------------------------------
# discovery helpers
# ---------------------------------------------------------------------------

# the omz_downloader output layout puts intel models in models/intel/<name>/<precision>/<name>.xml
_PRECISION_DIRS = ("FP16", "FP32", "FP16-INT8")


def find_model_xml(models_root: str | Path, model_name: str, precision: str = "FP16") -> Path:
    root = Path(models_root)

    # primary: intel/<name>/<precision>/<name>.xml
    candidate = root / "intel" / model_name / precision / f"{model_name}.xml"
    if candidate.exists():
        return candidate

    # fall back: search any precision dir we recognise
    for prec in _PRECISION_DIRS:
        c = root / "intel" / model_name / prec / f"{model_name}.xml"
        if c.exists():
            return c

    # last resort: walk the tree
    for p in root.rglob(f"{model_name}.xml"):
        return p

    raise FileNotFoundError(
        f"Could not find {model_name}.xml under {root}. "
        f"Run `python download_models.py` first."
    )


# ---------------------------------------------------------------------------
# small dataclasses we pass around
# ---------------------------------------------------------------------------

@dataclass
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)


@dataclass
class HeadPose:
    yaw: float
    pitch: float
    roll: float


@dataclass
class EyeState:
    left_closed: bool
    right_closed: bool

    @property
    def both_closed(self) -> bool:
        return self.left_closed and self.right_closed


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------

def _to_blob(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize -> NCHW float32, no mean/scale (the OMZ models for this set don't need it)."""
    resized = cv2.resize(img, (w, h))
    blob = resized.transpose(2, 0, 1)               # HWC -> CHW
    blob = np.expand_dims(blob, 0).astype(np.float32)
    return blob


# ---------------------------------------------------------------------------
# wrappers
# ---------------------------------------------------------------------------

class FaceDetector:
    """face-detection-adas-0001"""

    INPUT_W = 672
    INPUT_H = 384

    def __init__(self, core: Core, xml_path: Path, device: str = "CPU"):
        model = core.read_model(str(xml_path))
        self.compiled = core.compile_model(model, device)
        self.input_name = self.compiled.input(0).get_any_name()
        self.output_name = self.compiled.output(0).get_any_name()

    def detect(self, frame_bgr: np.ndarray, conf_thresh: float = 0.5) -> List[FaceBox]:
        h, w = frame_bgr.shape[:2]
        blob = _to_blob(frame_bgr, self.INPUT_W, self.INPUT_H)
        out = self.compiled([blob])[self.output_name]
        # shape is [1, 1, N, 7] -> [image_id, label, conf, x_min, y_min, x_max, y_max] (normalised)
        detections = out.reshape(-1, 7)

        faces: List[FaceBox] = []
        for det in detections:
            conf = float(det[2])
            if conf < conf_thresh:
                continue
            x1 = int(max(0, det[3] * w))
            y1 = int(max(0, det[4] * h))
            x2 = int(min(w - 1, det[5] * w))
            y2 = int(min(h - 1, det[6] * h))
            if x2 <= x1 or y2 <= y1:
                continue
            faces.append(FaceBox(x1, y1, x2, y2, conf))
        return faces

    @staticmethod
    def largest(faces: List[FaceBox]) -> FaceBox | None:
        if not faces:
            return None
        return max(faces, key=lambda f: f.area)


class LandmarkRegressor:
    """facial-landmarks-35-adas-0002 — returns 35 (x,y) pixel coords for a face crop."""

    INPUT_W = 60
    INPUT_H = 60

    def __init__(self, core: Core, xml_path: Path, device: str = "CPU"):
        model = core.read_model(str(xml_path))
        self.compiled = core.compile_model(model, device)
        self.output_name = self.compiled.output(0).get_any_name()

    def predict(self, face_crop_bgr: np.ndarray) -> np.ndarray:
        """Returns landmarks in *face-crop pixel coords*, shape (35, 2)."""
        h, w = face_crop_bgr.shape[:2]
        blob = _to_blob(face_crop_bgr, self.INPUT_W, self.INPUT_H)
        out = self.compiled([blob])[self.output_name].reshape(-1)
        # 70 floats -> 35 (x,y), each in [0,1] relative to the crop
        pts = out.reshape(35, 2).copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        return pts


class HeadPoseEstimator:
    """head-pose-estimation-adas-0001 — yaw / pitch / roll in degrees."""

    INPUT_W = 60
    INPUT_H = 60

    def __init__(self, core: Core, xml_path: Path, device: str = "CPU"):
        model = core.read_model(str(xml_path))
        self.compiled = core.compile_model(model, device)
        # this model has three named outputs, one per angle
        self.out_yaw = "angle_y_fc"
        self.out_pitch = "angle_p_fc"
        self.out_roll = "angle_r_fc"

    def estimate(self, face_crop_bgr: np.ndarray) -> HeadPose:
        blob = _to_blob(face_crop_bgr, self.INPUT_W, self.INPUT_H)
        result = self.compiled([blob])
        yaw = float(np.array(result[self.out_yaw]).flatten()[0])
        pitch = float(np.array(result[self.out_pitch]).flatten()[0])
        roll = float(np.array(result[self.out_roll]).flatten()[0])
        return HeadPose(yaw=yaw, pitch=pitch, roll=roll)


class EyeStateClassifier:
    """open-closed-eye-0001 — softmax over [open, closed]."""

    INPUT_W = 32
    INPUT_H = 32

    def __init__(self, core: Core, xml_path: Path, device: str = "CPU"):
        model = core.read_model(str(xml_path))
        self.compiled = core.compile_model(model, device)
        self.output_name = self.compiled.output(0).get_any_name()

    def classify(self, eye_crop_bgr: np.ndarray) -> Tuple[bool, float]:
        """Returns (is_closed, closed_probability)."""
        if eye_crop_bgr.size == 0:
            return False, 0.0
        blob = _to_blob(eye_crop_bgr, self.INPUT_W, self.INPUT_H)
        out = np.array(self.compiled([blob])[self.output_name]).flatten()
        # by Open Model Zoo spec: index 0 = closed, index 1 = open (verify in your downloaded copy)
        closed_prob = float(out[0])
        open_prob = float(out[1])
        is_closed = closed_prob > open_prob
        return is_closed, closed_prob


# ---------------------------------------------------------------------------
# eye-region cropping using the 35-point landmark layout
# ---------------------------------------------------------------------------

# Landmark indices for facial-landmarks-35-adas-0002 (per OMZ docs):
#   0..3  -> right eye (outer corner, top, inner corner, bottom-ish)
#   4..7  -> left eye  (inner corner, top, outer corner, bottom-ish)
# Using 0-3 and 4-7 to bound each eye works well enough in practice.
RIGHT_EYE_IDX = (0, 1, 2, 3)
LEFT_EYE_IDX = (4, 5, 6, 7)


def crop_eye(face_crop: np.ndarray, landmarks: np.ndarray, eye_indices: tuple, pad: float = 0.45) -> np.ndarray:
    pts = landmarks[list(eye_indices)]
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    half = max(x_max - x_min, y_max - y_min) * (0.5 + pad)
    x1 = int(max(0, cx - half))
    y1 = int(max(0, cy - half))
    x2 = int(min(face_crop.shape[1] - 1, cx + half))
    y2 = int(min(face_crop.shape[0] - 1, cy + half))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return face_crop[y1:y2, x1:x2].copy()
