"""Mock eye detector for deterministic unit tests and simulated fault injection.

Provides programmatic simulation of occlusions, missing faces, low confidence,
and custom bounding boxes without requiring live video frames or YOLO pose models.
"""
from __future__ import annotations

from typing import Optional
import numpy as np

from backend.eye_detection import BaseEyeDetector, DetectionStatus, EyeDetectionResult


class MockEyeDetector(BaseEyeDetector):
    """Mock eye detector for deterministic unit tests and simulated fault injection."""

    def __init__(self, canned_result: Optional[EyeDetectionResult] = None) -> None:
        """Initialize mock detector with optional preset response."""
        self.canned_result = canned_result

    def set_result(self, result: EyeDetectionResult) -> None:
        """Update canned result returned by detect()."""
        self.canned_result = result

    def detect(self, frame: np.ndarray) -> EyeDetectionResult:
        """Return preset canned result or synthesize a valid dummy result."""
        if self.canned_result is not None:
            return self.canned_result

        h, w = frame.shape[:2] if frame is not None and frame.size > 0 else (100, 100)
        dummy_crop = np.zeros((48, 48, 3), dtype=np.uint8)
        return EyeDetectionResult(
            status=DetectionStatus.SUCCESS,
            face_box=(10, 10, w - 10, h - 10),
            left_eye_box=(20, 30, 40, 50),
            right_eye_box=(60, 30, 80, 50),
            left_eye_crop=dummy_crop.copy(),
            right_eye_crop=dummy_crop.copy(),
            face_confidence=0.99,
            left_eye_confidence=0.98,
            right_eye_confidence=0.97,
            ear=0.28,
            left_ear=0.28,
            right_ear=0.28,
            message="Mock detection success.",
        )
