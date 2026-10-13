"""
Camera Tamper & Occlusion Detection Module
===========================================
Detects physical tampering or camera obstruction:
1. Complete lens occlusion / black cover (mean brightness < 8.0)
2. Direct blinding / laser / intense glare saturation (mean brightness > 245.0)
3. Heavy defocus blur / tape / frosted glass (Laplacian variance < 6.0)

Requires persistent anomalous frames across a temporal window to prevent false
triggers during momentary lighting shifts (e.g. tunnels or headlight flash).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2


@dataclass
class TamperStatus:
    """Structured result of tamper assessment."""
    is_tampered: bool
    tamper_type: Optional[str]  # "OCCLUDED", "BLINDED", "DEFOCUSED", or None
    consecutive_frames: int
    mean_brightness: float
    blur_score: float
    description: str


class TamperDetector:
    """
    Monitors camera frames for physical occlusion, blinding, or blur tampering.
    """

    def __init__(
        self,
        min_brightness_threshold: float = 8.0,
        max_brightness_threshold: float = 246.0,
        min_blur_threshold: float = 6.0,
        consecutive_frames_required: int = 10,
    ):
        """
        Initialize tamper detection parameters.

        Args:
            min_brightness_threshold: Mean intensity below which frame is considered occluded.
            max_brightness_threshold: Mean intensity above which frame is considered blinded.
            min_blur_threshold: Laplacian variance below which frame is considered out-of-focus.
            consecutive_frames_required: Frame count needed before triggering alert.
        """
        self.min_brightness = min_brightness_threshold
        self.max_brightness = max_brightness_threshold
        self.min_blur = min_blur_threshold
        self.required_streak = consecutive_frames_required

        self._streak_count: int = 0
        self._current_type: Optional[str] = None

    def analyze(self, frame: np.ndarray) -> TamperStatus:
        """
        Analyze a single video frame for tampering indicators.

        Args:
            frame: BGR numpy image frame from the video source.

        Returns:
            TamperStatus summarizing tamper presence and metrics.
        """
        if frame is None or frame.size == 0:
            self._streak_count += 1
            return TamperStatus(
                is_tampered=self._streak_count >= self.required_streak,
                tamper_type="NO_FEED",
                consecutive_frames=self._streak_count,
                mean_brightness=0.0,
                blur_score=0.0,
                description="Empty or missing camera frame feed",
            )

        # Convert to grayscale for illumination & texture evaluation
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Compute mean brightness and blur score (variance of Laplacian)
        mean_val = float(np.mean(gray))
        blur_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        detected_type: Optional[str] = None
        desc = "Camera feed normal"

        if mean_val < self.min_brightness:
            detected_type = "OCCLUDED"
            desc = f"Camera lens blocked or covered (mean brightness {mean_val:.1f} < {self.min_brightness})"
        elif mean_val > self.max_brightness:
            detected_type = "BLINDED"
            desc = f"Camera sensor blinded or saturated (mean brightness {mean_val:.1f} > {self.max_brightness})"
        elif blur_val < self.min_blur:
            detected_type = "DEFOCUSED"
            desc = f"Severe blur or lens obstruction (focus score {blur_val:.1f} < {self.min_blur})"

        if detected_type is not None:
            if self._current_type == detected_type:
                self._streak_count += 1
            else:
                self._current_type = detected_type
                self._streak_count = 1
        else:
            # Decay streak quickly to recover once camera is unobstructed
            self._streak_count = max(0, self._streak_count - 2)
            if self._streak_count == 0:
                self._current_type = None

        is_tampered = self._streak_count >= self.required_streak
        return TamperStatus(
            is_tampered=is_tampered,
            tamper_type=self._current_type if is_tampered else None,
            consecutive_frames=self._streak_count,
            mean_brightness=round(mean_val, 2),
            blur_score=round(blur_val, 2),
            description=desc if is_tampered else "Camera feed operating within normal parameters",
        )

    def reset(self) -> None:
        """Reset internal temporal streak counters."""
        self._streak_count = 0
        self._current_type = None
