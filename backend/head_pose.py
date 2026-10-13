"""Head pose estimation module using facial and body pose keypoints.

Estimates 3D head orientation (pitch, yaw, roll) using Perspective-n-Point
(cv2.solvePnP) or geometric keypoint trigonometry from YOLO Pose keypoints.
Detects nodding-off (forward head drops) and inattention (looking away from road).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import cv2
import numpy as np


@dataclass
class HeadPoseResult:
    """Structured result containing 3D Euler angles and behavioral flags."""

    pitch: float = 0.0  # degrees: positive = looking down (nodding), negative = looking up
    yaw: float = 0.0    # degrees: positive = looking left, negative = looking right
    roll: float = 0.0   # degrees: positive = tilted right, negative = tilted left
    is_nodding: bool = False       # Pitch exceeds forward droop threshold
    is_looking_away: bool = False  # Yaw/Pitch exceeds driver safe field of view
    confidence: float = 0.0
    method: str = "solvePnP"


class HeadPoseEstimator:
    """Estimates head pose Euler angles from 2D keypoints and camera geometry."""

    # 3D canonical facial model points in millimeter scale
    # Reference: Nose tip (0,0,0), Eyes (-30,-30,-30) / (30,-30,-30), Ears (-70,-10,-70) / (70,-10,-70)
    CANONICAL_MODEL_PTS = np.array([
        [0.0, 0.0, 0.0],          # Nose (kpt 0)
        [-32.0, -28.0, -30.0],    # Left eye (kpt 1)
        [32.0, -28.0, -30.0],     # Right eye (kpt 2)
        [-68.0, -10.0, -60.0],    # Left ear (kpt 3)
        [68.0, -10.0, -60.0],     # Right ear (kpt 4)
    ], dtype=np.float64)

    def __init__(
        self,
        nod_pitch_threshold: float = 16.0,
        away_yaw_threshold: float = 24.0,
        away_pitch_threshold: float = 22.0,
    ) -> None:
        """Initialize estimator with calibrated angle thresholds in degrees."""
        self.nod_pitch_threshold = nod_pitch_threshold
        self.away_yaw_threshold = away_yaw_threshold
        self.away_pitch_threshold = away_pitch_threshold

    def estimate(
        self,
        keypoints_xy: np.ndarray,
        keypoints_conf: Optional[np.ndarray] = None,
        frame_shape: Tuple[int, int] = (480, 640),
    ) -> HeadPoseResult:
        """Estimate 3D head pose from 17-point COCO keypoint array.

        Keypoint indices:
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
        """
        if keypoints_xy is None or len(keypoints_xy) < 5:
            return HeadPoseResult(method="none")

        h, w = frame_shape[:2]
        confs = keypoints_conf if keypoints_conf is not None else np.ones(len(keypoints_xy))
        valid_indices = [i for i in range(5) if confs[i] >= 0.25 and keypoints_xy[i][0] > 0]

        # Use Perspective-n-Point when at least 4 keypoints are valid
        if len(valid_indices) >= 4:
            pnp_res = self._solve_pnp(keypoints_xy, valid_indices, w, h)
            if pnp_res is not None:
                pitch, yaw, roll = pnp_res
                return self._build_result(pitch, yaw, roll, float(np.mean([confs[i] for i in valid_indices])), "solvePnP")

        # Fallback to direct geometric ratio estimation
        return self._estimate_geometric(keypoints_xy, confs)

    def _solve_pnp(
        self,
        kpts: np.ndarray,
        valid_idx: list[int],
        w: int,
        h: int,
    ) -> Optional[Tuple[float, float, float]]:
        """Compute Euler angles via cv2.solvePnP with camera calibration matrix."""
        model_pts = self.CANONICAL_MODEL_PTS[valid_idx]
        img_pts = kpts[valid_idx].astype(np.float64)

        focal_length = float(w)
        cam_matrix = np.array([
            [focal_length, 0.0, float(w) / 2.0],
            [0.0, focal_length, float(h) / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, _ = cv2.solvePnP(
            model_pts, img_pts, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
        )
        if not success:
            return None

        # Convert rotation vector to 3x3 rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            pitch = math.atan2(R[2, 1], R[2, 2])
            yaw = math.atan2(-R[2, 0], sy)
            roll = math.atan2(R[1, 0], R[0, 0])
        else:
            pitch = math.atan2(-R[1, 2], R[1, 1])
            yaw = math.atan2(-R[2, 0], sy)
            roll = 0.0

        return -math.degrees(pitch), math.degrees(yaw), math.degrees(roll)

    def _estimate_geometric(
        self,
        kpts: np.ndarray,
        confs: np.ndarray,
    ) -> HeadPoseResult:
        """Estimate head pose using planar trigonometric ratios."""
        nose, left_eye, right_eye = kpts[0], kpts[1], kpts[2]
        if confs[1] < 0.2 or confs[2] < 0.2:
            return HeadPoseResult(method="insufficient_kpts")

        # Roll: Eye slope angle
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        iod = math.hypot(dx, dy)
        roll = math.degrees(math.atan2(dy, dx)) if iod > 0 else 0.0

        # Yaw: Eye-to-nose horizontal displacement ratio
        eye_cx = (left_eye[0] + right_eye[0]) / 2.0
        yaw_disp = (nose[0] - eye_cx) / max(iod * 0.5, 1.0)
        yaw = float(np.clip(yaw_disp * 45.0, -60.0, 60.0))

        # Pitch: Eye-to-nose vertical displacement ratio
        eye_cy = (left_eye[1] + right_eye[1]) / 2.0
        expected_dy = iod * 0.65
        actual_dy = nose[1] - eye_cy
        pitch_disp = (actual_dy - expected_dy) / max(expected_dy, 1.0)
        pitch = float(np.clip(pitch_disp * 50.0, -45.0, 50.0))

        mean_conf = float(np.mean(confs[:3]))
        return self._build_result(pitch, yaw, roll, mean_conf, "geometric")

    def _build_result(
        self, pitch: float, yaw: float, roll: float, conf: float, method: str
    ) -> HeadPoseResult:
        """Construct structured result with threshold-evaluated flags."""
        is_nodding = pitch >= self.nod_pitch_threshold
        is_looking_away = abs(yaw) >= self.away_yaw_threshold or pitch <= -self.away_pitch_threshold
        return HeadPoseResult(
            pitch=round(pitch, 1),
            yaw=round(yaw, 1),
            roll=round(roll, 1),
            is_nodding=is_nodding,
            is_looking_away=is_looking_away,
            confidence=round(conf, 2),
            method=method,
        )
