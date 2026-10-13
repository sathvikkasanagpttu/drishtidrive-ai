"""Driver distraction detection module.

Monitors real-time distraction indicators:
1. Extended inattention / looking away from the road (> N seconds).
2. Mobile phone usage (hand/wrist elevated to ear/face zone or phone object).
3. Seatbelt absence (diagonal chest contour verification).

Produces distinct distraction events with custom alert dispatching.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from backend.head_pose import HeadPoseResult


class DistractionType(str, Enum):
    """Specific categories of driver distraction."""

    NONE = "NONE"
    HEAD_POSE_AWAY = "HEAD_POSE_AWAY"
    PHONE_DISTRACTION = "PHONE_DISTRACTION"
    NO_SEATBELT = "NO_SEATBELT"


@dataclass
class DistractionStatus:
    """Real-time assessment of distraction metrics."""

    active_type: DistractionType = DistractionType.NONE
    is_distracted: bool = False
    away_duration_sec: float = 0.0
    phone_confidence: float = 0.0
    seatbelt_present: bool = True
    message: str = ""
    event_timestamp: float = 0.0


class DistractionDetector:
    """Tracks driver attentiveness and flags behavioral distractions."""

    def __init__(
        self,
        max_away_seconds: float = 2.0,
        phone_wrist_ear_ratio: float = 0.70,
        seatbelt_check_enabled: bool = True,
    ) -> None:
        """Initialize detector with temporal thresholds."""
        self.max_away_seconds = max_away_seconds
        self.phone_wrist_ear_ratio = phone_wrist_ear_ratio
        self.seatbelt_check_enabled = seatbelt_check_enabled

        self._away_start_time: Optional[float] = None
        self._phone_event_frames: int = 0
        self._seatbelt_missing_frames: int = 0

    def update(
        self,
        head_pose: HeadPoseResult,
        keypoints_xy: Optional[np.ndarray],
        keypoints_conf: Optional[np.ndarray],
        frame: Optional[np.ndarray] = None,
        timestamp: float = 0.0,
    ) -> DistractionStatus:
        """Process frame observations and return current distraction state."""
        # 1. Looking away tracking
        away_duration = 0.0
        if head_pose.is_looking_away:
            if self._away_start_time is None:
                self._away_start_time = timestamp
            away_duration = max(0.0, timestamp - self._away_start_time)
        else:
            self._away_start_time = None

        if away_duration >= self.max_away_seconds:
            return DistractionStatus(
                active_type=DistractionType.HEAD_POSE_AWAY,
                is_distracted=True,
                away_duration_sec=round(away_duration, 2),
                seatbelt_present=True,
                message=f"Driver looking away from road for {away_duration:.1f}s.",
                event_timestamp=timestamp,
            )

        # 2. Phone near ear / face detection
        phone_detected, phone_conf = self._detect_phone_usage(keypoints_xy, keypoints_conf)
        if phone_detected:
            self._phone_event_frames += 1
            if self._phone_event_frames >= 3:  # 3 consecutive frames to filter glitch
                return DistractionStatus(
                    active_type=DistractionType.PHONE_DISTRACTION,
                    is_distracted=True,
                    away_duration_sec=round(away_duration, 2),
                    phone_confidence=round(phone_conf, 2),
                    seatbelt_present=True,
                    message="Driver mobile phone usage detected.",
                    event_timestamp=timestamp,
                )
        else:
            self._phone_event_frames = max(0, self._phone_event_frames - 1)

        # 3. Seatbelt verification
        seatbelt_ok = True
        if self.seatbelt_check_enabled and keypoints_xy is not None and frame is not None:
            seatbelt_ok = self._verify_seatbelt(keypoints_xy, keypoints_conf, frame)
            if not seatbelt_ok:
                self._seatbelt_missing_frames += 1
                if self._seatbelt_missing_frames >= 15:  # ~0.5s sustained
                    return DistractionStatus(
                        active_type=DistractionType.NO_SEATBELT,
                        is_distracted=True,
                        seatbelt_present=False,
                        message="Driver seatbelt not fastened.",
                        event_timestamp=timestamp,
                    )
            else:
                self._seatbelt_missing_frames = max(0, self._seatbelt_missing_frames - 2)

        return DistractionStatus(
            active_type=DistractionType.NONE,
            is_distracted=False,
            away_duration_sec=round(away_duration, 2),
            phone_confidence=0.0,
            seatbelt_present=seatbelt_ok,
            message="Driver attentive and focused on driving.",
            event_timestamp=timestamp,
        )

    def _detect_phone_usage(
        self,
        kpts: Optional[np.ndarray],
        confs: Optional[np.ndarray],
    ) -> Tuple[bool, float]:
        """Detect hand/wrist close to ear/mouth using COCO pose keypoints."""
        if kpts is None or len(kpts) < 11:
            return False, 0.0

        conf_arr = confs if confs is not None else np.ones(len(kpts))
        # Left ear: 3, Right ear: 4, Nose: 0
        # Left wrist: 9, Right wrist: 10
        # Left shoulder: 5, Right shoulder: 6
        if conf_arr[5] < 0.25 or conf_arr[6] < 0.25:
            return False, 0.0

        shoulder_span = math.hypot(kpts[6][0] - kpts[5][0], kpts[6][1] - kpts[5][1])
        if shoulder_span < 15.0:
            return False, 0.0

        max_proximity_dist = shoulder_span * self.phone_wrist_ear_ratio

        # Check right wrist (10) near right ear (4) or nose (0)
        if conf_arr[10] >= 0.3:
            dist_ear_r = math.hypot(kpts[10][0] - kpts[4][0], kpts[10][1] - kpts[4][1])
            dist_nose_r = math.hypot(kpts[10][0] - kpts[0][0], kpts[10][1] - kpts[0][1])
            if dist_ear_r < max_proximity_dist or dist_nose_r < max_proximity_dist * 0.8:
                return True, float(conf_arr[10])

        # Check left wrist (9) near left ear (3) or nose (0)
        if conf_arr[9] >= 0.3:
            dist_ear_l = math.hypot(kpts[9][0] - kpts[3][0], kpts[9][1] - kpts[3][1])
            dist_nose_l = math.hypot(kpts[9][0] - kpts[0][0], kpts[9][1] - kpts[0][1])
            if dist_ear_l < max_proximity_dist or dist_nose_l < max_proximity_dist * 0.8:
                return True, float(conf_arr[9])

        return False, 0.0

    def _verify_seatbelt(
        self,
        kpts: np.ndarray,
        confs: Optional[np.ndarray],
        frame: np.ndarray,
    ) -> bool:
        """Verify presence of diagonal seatbelt strap across driver chest."""
        conf_arr = confs if confs is not None else np.ones(len(kpts))
        # Left shoulder: 5, Right shoulder: 6
        if len(kpts) < 7 or conf_arr[5] < 0.3 or conf_arr[6] < 0.3:
            return True  # Insufficient torso keypoints, do not falsely flag

        ls, rs = kpts[5], kpts[6]
        mid_x = int((ls[0] + rs[0]) / 2.0)
        mid_y = int((ls[1] + rs[1]) / 2.0)
        torso_w = int(abs(rs[0] - ls[0]))
        torso_h = int(torso_w * 0.9)

        h, w = frame.shape[:2]
        x1 = max(0, mid_x - torso_w // 2)
        y1 = max(0, mid_y)
        x2 = min(w, mid_x + torso_w // 2)
        y2 = min(h, mid_y + torso_h)

        if x2 - x1 < 20 or y2 - y1 < 20:
            return True

        chest_roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(chest_roi, cv2.COLOR_BGR2GRAY) if len(chest_roi.shape) == 3 else chest_roi
        # Diagonal edge filter (seatbelt creates prominent diagonal gradient)
        edges = cv2.Canny(gray, 40, 120)
        diag_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=15, maxLineGap=10)
        if diag_lines is None:
            return False  # No diagonal strap detected across chest

        for line in diag_lines:
            coords = line.flatten()
            if len(coords) < 4:
                continue
            lx1, ly1, lx2, ly2 = coords[:4]
            angle = abs(math.degrees(math.atan2(ly2 - ly1, lx2 - lx1)))
            if 25.0 <= angle <= 65.0:  # Typical diagonal seatbelt belt slope
                return True

        return False

    def reset(self) -> None:
        """Clear temporal states."""
        self._away_start_time = None
        self._phone_event_frames = 0
        self._seatbelt_missing_frames = 0
