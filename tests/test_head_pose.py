"""Unit tests for head pose estimation from 2D keypoints.

Verifies:
- 3D Euler angle estimation (pitch, yaw, roll).
- Nodding-off detection when forward pitch exceeds nod threshold.
- Looking-away inattention detection when yaw/pitch exceeds road focus envelope.
- Smooth fallback from solvePnP to planar trigonometry when keypoints are degraded.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.head_pose import HeadPoseEstimator, HeadPoseResult


@pytest.fixture
def estimator() -> HeadPoseEstimator:
    """Instantiate standard head pose estimator."""
    return HeadPoseEstimator(nod_pitch_threshold=16.0, away_yaw_threshold=24.0)


def test_neutral_frontal_pose(estimator: HeadPoseEstimator) -> None:
    """Verify looking straight ahead yields near-zero Euler angles and alert flags."""
    # Frontal keypoints: Nose=(320, 240), LeftEye=(290, 210), RightEye=(350, 210), LeftEar=(250, 230), RightEar=(390, 230)
    kpts = np.array([
        [320.0, 240.0],
        [290.0, 210.0],
        [350.0, 210.0],
        [250.0, 230.0],
        [390.0, 230.0],
    ], dtype=np.float32)
    confs = np.array([0.9, 0.9, 0.9, 0.85, 0.85], dtype=np.float32)

    res = estimator.estimate(kpts, confs, (480, 640))
    assert isinstance(res, HeadPoseResult)
    assert not res.is_nodding
    assert not res.is_looking_away
    assert abs(res.yaw) < 15.0
    assert abs(res.roll) < 10.0


def test_nodding_off_detection(estimator: HeadPoseEstimator) -> None:
    """Verify downward head pitch (>16 deg) flags is_nodding."""
    # Nodding: Nose drops significantly downward relative to eye/ear plane
    kpts = np.array([
        [320.0, 290.0],  # Dropped nose
        [290.0, 210.0],
        [350.0, 210.0],
        [250.0, 205.0],  # Ears appear higher relative to drooping face
        [390.0, 205.0],
    ], dtype=np.float32)
    confs = np.ones(5, dtype=np.float32)

    res = estimator.estimate(kpts, confs, (480, 640))
    assert res.is_nodding, f"Failed to detect nodding off: pitch={res.pitch}"


def test_looking_away_detection(estimator: HeadPoseEstimator) -> None:
    """Verify turning head sideways (>24 deg yaw) flags is_looking_away."""
    # Looking far left: Nose shifts heavily towards left eye/ear
    kpts = np.array([
        [270.0, 240.0],  # Shifted nose
        [255.0, 210.0],
        [340.0, 210.0],
        [240.0, 230.0],
        [380.0, 220.0],
    ], dtype=np.float32)
    confs = np.ones(5, dtype=np.float32)

    res = estimator.estimate(kpts, confs, (480, 640))
    assert res.is_looking_away, f"Failed to detect looking away: yaw={res.yaw}"


def test_graceful_fallback_with_partial_occlusion(estimator: HeadPoseEstimator) -> None:
    """Verify estimator falls back gracefully without crashing when ears are occluded."""
    kpts = np.array([
        [320.0, 240.0],
        [290.0, 210.0],
        [350.0, 210.0],
        [0.0, 0.0],  # Occluded left ear
        [0.0, 0.0],  # Occluded right ear
    ], dtype=np.float32)
    confs = np.array([0.9, 0.9, 0.9, 0.0, 0.0], dtype=np.float32)

    res = estimator.estimate(kpts, confs, (480, 640))
    assert res.method in ("geometric", "solvePnP")
    assert not res.is_looking_away
