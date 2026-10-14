"""Face and eye region detection service for DrishtiDrive AI.

Locates face and eye regions using facial pose keypoints or fallback detectors.
Explicitly detects and flags edge cases: missing faces, multiple occupants,
occlusions, low confidence, and poor frame quality (blur/darkness).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import List, Optional, Tuple
import cv2
import numpy as np

from backend.config import DetectionConfig
from backend.face_geometry import compute_crop_mar, extract_crop_rect


class DetectionStatus(str, Enum):
    """Enumeration of detection outcomes and degradation states."""

    SUCCESS = "SUCCESS"
    NO_FACE = "NO_FACE"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    OCCLUSION = "OCCLUSION"
    POOR_QUALITY = "POOR_QUALITY"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


@dataclass
class EyeDetectionResult:
    """Structured result containing localized eye and mouth bounding boxes and status."""

    status: DetectionStatus
    face_box: Optional[Tuple[int, int, int, int]] = None
    left_eye_box: Optional[Tuple[int, int, int, int]] = None
    right_eye_box: Optional[Tuple[int, int, int, int]] = None
    mouth_box: Optional[Tuple[int, int, int, int]] = None
    left_eye_crop: Optional[np.ndarray] = None
    right_eye_crop: Optional[np.ndarray] = None
    face_confidence: float = 0.0
    left_eye_confidence: float = 0.0
    right_eye_confidence: float = 0.0
    ear: Optional[float] = None
    left_ear: Optional[float] = None
    right_ear: Optional[float] = None
    mar: float = 0.14
    is_yawning: bool = False
    keypoints_xy: Optional[np.ndarray] = None
    keypoints_conf: Optional[np.ndarray] = None
    message: str = ""

    @property
    def is_valid(self) -> bool:
        """Return True if observation is valid for confident downstream inference."""
        return self.status == DetectionStatus.SUCCESS and (
            self.left_eye_crop is not None or self.right_eye_crop is not None
        )


class BaseEyeDetector:
    """Abstract base detector interface."""

    def detect(self, frame: np.ndarray) -> EyeDetectionResult:
        """Detect face and eye regions from input frame."""
        raise NotImplementedError


class PoseEyeDetector(BaseEyeDetector):
    """Detects face and eye regions using Ultralytics YOLO Pose keypoints."""

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        """Initialize detector with configuration and load pose model."""
        self.config = config or DetectionConfig()
        # pyrefly: ignore [missing-import]
        from pathlib import Path
        from ultralytics import YOLO

        weights_path = Path(self.config.pose_weights)
        if not weights_path.exists():
            candidate = Path(__file__).resolve().parent / "models" / weights_path.name
            if candidate.exists():
                weights_path = candidate
        self.model = YOLO(str(weights_path))

    def detect(self, frame: np.ndarray) -> EyeDetectionResult:
        """Execute pose detection and extract verified eye regions."""
        working_frame = frame
        quality_status, quality_msg = self._check_frame_quality(frame)
        if quality_status != DetectionStatus.SUCCESS:
            from backend.preprocessing import EyePreprocessor
            norm_frame = EyePreprocessor.normalize_illumination(frame)
            norm_status, _ = self._check_frame_quality(norm_frame)
            if norm_status == DetectionStatus.SUCCESS:
                working_frame = norm_frame
            else:
                return EyeDetectionResult(status=quality_status, message=quality_msg)

        # Step 2: YOLO Pose Inference
        results = self.model.predict(working_frame, verbose=False)
        if not results or len(results) == 0:
            return EyeDetectionResult(
                status=DetectionStatus.NO_FACE, message="No person or face detected in frame."
            )

        boxes = results[0].boxes
        keypoints = results[0].keypoints

        num_people = len(boxes) if boxes is not None else 0
        if num_people == 0 or keypoints is None or len(keypoints) == 0:
            return EyeDetectionResult(
                status=DetectionStatus.NO_FACE, message="No face keypoints found in frame."
            )

        # Step 3: Handle multiple occupants
        if num_people > 1 and not self.config.allow_multiple_faces:
            return EyeDetectionResult(
                status=DetectionStatus.MULTIPLE_FACES,
                message=f"Multiple occupants detected ({num_people}). Ambiguous driver context.",
            )

        # Step 4: Extract primary occupant box & keypoints (largest bounding box)
        all_boxes = boxes.xyxy.cpu().numpy().astype(int)
        all_confs = boxes.conf.cpu().numpy()
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in all_boxes]
        primary_idx = int(np.argmax(areas))

        primary_box = all_boxes[primary_idx]
        face_conf = float(all_confs[primary_idx])
        if face_conf < self.config.min_face_confidence:
            return EyeDetectionResult(
                status=DetectionStatus.LOW_CONFIDENCE,
                face_confidence=round(face_conf, 3),
                message="Face confidence below required threshold.",
            )

        # Keypoints: 1=left_eye, 2=right_eye (COCO indexing)
        kpts_xy = keypoints.xy[primary_idx].cpu().numpy()
        kpts_conf = (
            keypoints.conf[primary_idx].cpu().numpy()
            if keypoints.conf is not None
            else np.ones(len(kpts_xy))
        )

        left_pt, left_conf = kpts_xy[1], float(kpts_conf[1])
        right_pt, right_conf = kpts_xy[2], float(kpts_conf[2])

        min_kconf = self.config.min_keypoint_confidence
        left_valid = left_conf >= min_kconf and left_pt[0] > 0 and left_pt[1] > 0
        right_valid = right_conf >= min_kconf and right_pt[0] > 0 and right_pt[1] > 0

        if not left_valid and not right_valid:
            return EyeDetectionResult(
                status=DetectionStatus.OCCLUSION,
                face_confidence=round(face_conf, 3),
                left_eye_confidence=round(left_conf, 3),
                right_eye_confidence=round(right_conf, 3),
                message="Both eye regions occluded or keypoints below confidence threshold.",
            )

        # Step 5: Compute crop regions based on inter-ocular distance or face scale
        face_w = float(primary_box[2] - primary_box[0])
        nominal_iod = face_w * 0.30
        iod = (
            math.hypot(right_pt[0] - left_pt[0], right_pt[1] - left_pt[1])
            if (left_valid and right_valid)
            else nominal_iod
        )
        # Guard against keypoint jitter or perspective compression
        effective_iod = max(nominal_iod * 0.65, min(nominal_iod * 1.35, iod))
        box_size = max(24, int(effective_iod * 0.62))

        h, w = frame.shape[:2]
        left_crop, left_box = self._extract_crop(frame, left_pt, box_size, w, h) if left_valid else (None, None)
        right_crop, right_box = self._extract_crop(frame, right_pt, box_size, w, h) if right_valid else (None, None)

        # Step 6: Compute physical Eye Aspect Ratio (EAR) from cropped ocular fissures
        left_ear = self._compute_crop_ear(left_crop) if left_crop is not None else 0.28
        right_ear = self._compute_crop_ear(right_crop) if right_crop is not None else 0.28
        valid_ears = [e for c, e in [(left_crop, left_ear), (right_crop, right_ear)] if c is not None]
        mean_ear = round(float(np.mean(valid_ears)), 3) if valid_ears else 0.28

        # Step 7: Localize mouth region and calculate Mouth Aspect Ratio (MAR)
        mouth_box = None
        mar_val = 0.14
        nose_pt = kpts_xy[0]
        if left_valid and right_valid and nose_pt[0] > 0 and nose_pt[1] > 0:
            eye_y = (left_pt[1] + right_pt[1]) / 2.0
            nose_dist = max(12.0, float(nose_pt[1] - eye_y))
            mouth_cx = int(nose_pt[0])
            mouth_cy = int(nose_pt[1] + nose_dist * 0.95)
            mw = max(14, int(effective_iod * 0.40))
            mh = max(10, int(nose_dist * 0.55))
            m_crop, m_box = extract_crop_rect(frame, mouth_cx, mouth_cy, mw * 2, mh * 2, w, h)
            if m_crop is not None:
                mouth_box = m_box
                mar_val = compute_crop_mar(m_crop)

        return EyeDetectionResult(
            status=DetectionStatus.SUCCESS,
            face_box=(primary_box[0], primary_box[1], primary_box[2], primary_box[3]),
            left_eye_box=left_box,
            right_eye_box=right_box,
            mouth_box=mouth_box,
            left_eye_crop=left_crop,
            right_eye_crop=right_crop,
            face_confidence=round(face_conf, 3),
            left_eye_confidence=round(left_conf, 3),
            right_eye_confidence=round(right_conf, 3),
            ear=mean_ear,
            left_ear=left_ear,
            right_ear=right_ear,
            mar=mar_val,
            is_yawning=mar_val >= 0.65,
            keypoints_xy=kpts_xy,
            keypoints_conf=kpts_conf,
            message="Eyes successfully detected and cropped.",
        )

    def _compute_crop_ear(self, crop: Optional[np.ndarray]) -> float:
        """Estimate physical Eye Aspect Ratio (EAR) from vertical ocular aperture."""
        if crop is None or crop.size == 0:
            return 0.28
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        h, w = gray.shape
        if h < 8 or w < 8:
            return 0.28

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        eq = clahe.apply(gray)
        blurred = cv2.GaussianBlur(eq, (3, 3), 0)

        roi = blurred[int(h * 0.20) : int(h * 0.80), int(w * 0.15) : int(w * 0.85)]
        if roi.size == 0:
            return 0.28

        grad_y = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        vert_profile = np.mean(np.abs(grad_y), axis=1)
        thresh = float(np.mean(vert_profile)) * 0.90
        edges = np.where(vert_profile > thresh)[0]

        if len(edges) >= 2:
            aperture = float(edges[-1] - edges[0])
        else:
            _, bin_mask = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            col_counts = np.sum(bin_mask > 0, axis=0)
            nz = col_counts[col_counts > 0]
            aperture = float(np.percentile(nz, 85)) if len(nz) > 0 else 0.0

        raw_ear = aperture / max(1.0, float(w * 0.70))
        return round(float(np.clip(raw_ear, 0.08, 0.38)), 3)

    def _extract_crop(
        self, frame: np.ndarray, pt: np.ndarray, size: int, width: int, height: int
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Crop square bounding box around eye center with border clamping."""
        half = size // 2
        cx, cy = int(pt[0]), int(pt[1])
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(width, cx + half)
        y2 = min(height, cy + half)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None, None
        return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    def _check_frame_quality(self, frame: np.ndarray) -> Tuple[DetectionStatus, str]:
        """Verify brightness and focus sharpness of input frame."""
        if frame is None or frame.size == 0:
            return DetectionStatus.POOR_QUALITY, "Empty or invalid frame."

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        brightness = float(np.mean(gray))
        if brightness < self.config.min_brightness:
            return DetectionStatus.POOR_QUALITY, f"Frame too dark ({brightness:.1f} < {self.config.min_brightness})."

        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < self.config.blur_threshold:
            return DetectionStatus.POOR_QUALITY, f"Frame excessively blurry ({laplacian_var:.1f} < {self.config.blur_threshold})."

        return DetectionStatus.SUCCESS, "Quality OK."


# Re-export mock detector for testing convenience
from backend.mock_detector import MockEyeDetector

__all__ = [
    "BaseEyeDetector",
    "DetectionStatus",
    "EyeDetectionResult",
    "MockEyeDetector",
    "PoseEyeDetector",
]

