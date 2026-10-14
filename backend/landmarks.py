"""MediaPipe Face Landmarker integration for high-precision eye tracking.

Computes physical Eye Aspect Ratio (EAR) from 6 standard 3D landmarks per eye
and extracts eyeBlinkLeft / eyeBlinkRight blendshape coefficients.
Provides fallback capability when MediaPipe is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("drishtidrive.landmarks")

# Standard MediaPipe 468/478 Face Mesh landmark indices for eyes
# p1=outer_corner, p2=upper_outer, p3=upper_inner, p4=inner_corner, p5=lower_inner, p6=lower_outer
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]


@dataclass
class EyeLandmarksResult:
    """Structured landmark telemetry for ocular apertures and facial blendshapes."""

    detected: bool = False
    left_ear: float = 0.28
    right_ear: float = 0.28
    mean_ear: float = 0.28
    blink_left: float = 0.0
    blink_right: float = 0.0
    mean_blink: float = 0.0
    landmarks_confidence: float = 0.0
    face_box: Optional[Tuple[int, int, int, int]] = None
    left_eye_points: List[Tuple[float, float]] = field(default_factory=list)
    right_eye_points: List[Tuple[float, float]] = field(default_factory=list)
    left_eye_crop: Optional[np.ndarray] = None
    right_eye_crop: Optional[np.ndarray] = None
    message: str = ""


class MediaPipeFaceLandmarker:
    """Extracts 3D facial landmarks and blendshapes via MediaPipe Tasks API."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize landmarker with optional task model asset path."""
        self._landmarker = None
        self.is_available = False
        self._mp = None

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

            self._mp = mp
            resolved_path = self._resolve_model_path(model_path)
            if not resolved_path or not resolved_path.exists():
                logger.warning(
                    "MediaPipe model asset not found at '%s'. Landmarker running in degraded mode.",
                    model_path or "backend/models/face_landmarker.task",
                )
                return

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(resolved_path)),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=True,
            )
            self._landmarker = FaceLandmarker.create_from_options(options)
            self.is_available = True
            logger.info("MediaPipe Face Landmarker initialized from %s", resolved_path)
        except Exception as exc:
            logger.warning("MediaPipe initialization failed: %s. Falling back to YOLO pose.", exc)
            self.is_available = False

    def _resolve_model_path(self, model_path: Optional[str]) -> Optional[Path]:
        """Locate landmarker task model file across expected directories."""
        if model_path is not None:
            p = Path(model_path)
            return p if p.exists() else None
        candidates = [
            Path("backend/models/face_landmarker.task"),
            Path(__file__).resolve().parent / "models" / "face_landmarker.task",
            Path.home() / ".cache" / "drishtidrive" / "face_landmarker.task",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def detect(self, frame: np.ndarray) -> EyeLandmarksResult:
        """Process BGR image frame and return landmark ocular telemetry."""
        if not self.is_available or self._landmarker is None or frame is None or frame.size == 0:
            return EyeLandmarksResult(detected=False, message="MediaPipe Landmarker unavailable.")

        h, w = frame.shape[:2]
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
            result = self._landmarker.detect(mp_image)
        except Exception as err:
            logger.debug("MediaPipe forward pass error: %s", err)
            return EyeLandmarksResult(detected=False, message=f"Inference error: {err}")

        if not result or not result.face_landmarks:
            return EyeLandmarksResult(detected=False, message="No face detected by landmarker.")

        lms = result.face_landmarks[0]
        # Extract 6 eye points for left and right eyes
        left_pts = [(lms[idx].x * w, lms[idx].y * h) for idx in LEFT_EYE_INDICES]
        right_pts = [(lms[idx].x * w, lms[idx].y * h) for idx in RIGHT_EYE_INDICES]

        # Compute Eye Aspect Ratio (EAR) from euclidean landmark distances
        left_ear = self._compute_ear(left_pts)
        right_ear = self._compute_ear(right_pts)
        mean_ear = round(float((left_ear + right_ear) / 2.0), 4)

        # Extract blendshapes (eyeBlinkLeft, eyeBlinkRight)
        blink_left, blink_right = 0.0, 0.0
        if result.face_blendshapes and len(result.face_blendshapes) > 0:
            for category in result.face_blendshapes[0]:
                if category.category_name == "eyeBlinkLeft":
                    blink_left = float(category.score)
                elif category.category_name == "eyeBlinkRight":
                    blink_right = float(category.score)
        mean_blink = round(float((blink_left + blink_right) / 2.0), 4)

        # Estimate tight bounding box around full face from landmarks
        all_x = [pt.x * w for pt in lms]
        all_y = [pt.y * h for pt in lms]
        face_box = (
            max(0, int(min(all_x))),
            max(0, int(min(all_y))),
            min(w, int(max(all_x))),
            min(h, int(max(all_y))),
        )

        # Extract landmark-aligned eye crops with anatomical padding
        left_crop = self._extract_eye_crop(frame, left_pts, w, h)
        right_crop = self._extract_eye_crop(frame, right_pts, w, h)

        return EyeLandmarksResult(
            detected=True,
            left_ear=round(left_ear, 4),
            right_ear=round(right_ear, 4),
            mean_ear=mean_ear,
            blink_left=round(blink_left, 4),
            blink_right=round(blink_right, 4),
            mean_blink=mean_blink,
            landmarks_confidence=0.95,
            face_box=face_box,
            left_eye_points=left_pts,
            right_eye_points=right_pts,
            left_eye_crop=left_crop,
            right_eye_crop=right_crop,
            message="Landmarks and blendshapes computed.",
        )

    @staticmethod
    def _compute_ear(pts: List[Tuple[float, float]]) -> float:
        """Calculate physical Eye Aspect Ratio (EAR) from 6 2D points.

        EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
        """
        if len(pts) < 6:
            return 0.28
        p1, p2, p3, p4, p5, p6 = pts[:6]
        v1 = math.hypot(p2[0] - p6[0], p2[1] - p6[1])
        v2 = math.hypot(p3[0] - p5[0], p3[1] - p5[1])
        horiz = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
        if horiz < 1e-4:
            return 0.28
        ear = (v1 + v2) / (2.0 * horiz)
        return float(np.clip(ear, 0.02, 0.50))

    @staticmethod
    def _extract_eye_crop(
        frame: np.ndarray,
        pts: List[Tuple[float, float]],
        w: int,
        h: int,
        pad_x_ratio: float = 0.35,
        pad_y_ratio: float = 0.60,
    ) -> Optional[np.ndarray]:
        """Extract rectangular crop bounded by eye landmarks with proportional padding."""
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        ew = max(10.0, max_x - min_x)
        eh = max(6.0, max_y - min_y)
        pad_x = ew * pad_x_ratio
        pad_y = max(eh * pad_y_ratio, pad_x * 0.7)

        x1 = max(0, int(min_x - pad_x))
        y1 = max(0, int(min_y - pad_y))
        x2 = min(w, int(max_x + pad_x))
        y2 = min(h, int(max_y + pad_y))

        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return frame[y1:y2, x1:x2].copy()
