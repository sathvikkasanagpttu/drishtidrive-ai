"""Unit tests for driver distraction detection module.

Verifies:
- Looking-away duration tracking and trigger (>2.0s).
- Phone use detection when hand/wrist keypoint is near ear/face zone.
- Seatbelt absence verification.
- Output of separate DistractionType events with distinct alert messages.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.distraction import DistractionDetector, DistractionStatus, DistractionType
from backend.head_pose import HeadPoseResult


@pytest.fixture
def detector() -> DistractionDetector:
    """Instantiate standard distraction detector."""
    return DistractionDetector(max_away_seconds=2.0)


def test_looking_away_duration_escalation(detector: DistractionDetector) -> None:
    """Verify looking away under 2.0s remains unflagged, then triggers HEAD_POSE_AWAY."""
    head_away = HeadPoseResult(is_looking_away=True, yaw=30.0)

    # 1. Looking away for 1.0s (under threshold)
    status1 = detector.update(head_away, None, None, timestamp=10.0)
    assert not status1.is_distracted
    assert status1.active_type == DistractionType.NONE

    status2 = detector.update(head_away, None, None, timestamp=11.5)
    assert not status2.is_distracted

    # 2. Looking away exceeds 2.0s
    status3 = detector.update(head_away, None, None, timestamp=12.2)
    assert status3.is_distracted
    assert status3.active_type == DistractionType.HEAD_POSE_AWAY
    assert status3.away_duration_sec >= 2.0

    # 3. Driver returns attention to road
    head_forward = HeadPoseResult(is_looking_away=False, yaw=2.0)
    status4 = detector.update(head_forward, None, None, timestamp=12.5)
    assert not status4.is_distracted
    assert status4.active_type == DistractionType.NONE


def test_phone_usage_detection(detector: DistractionDetector) -> None:
    """Verify wrist elevated close to ear flags PHONE_DISTRACTION after 3 frames."""
    head_forward = HeadPoseResult(is_looking_away=False)

    # 17 keypoints:
    # 0: nose, 3: left_ear, 4: right_ear, 5: left_shoulder, 6: right_shoulder, 10: right_wrist
    kpts = np.zeros((17, 2), dtype=np.float32)
    kpts[0] = [320.0, 240.0]   # nose
    kpts[3] = [260.0, 230.0]   # left ear
    kpts[4] = [380.0, 230.0]   # right ear
    kpts[5] = [220.0, 360.0]   # left shoulder
    kpts[6] = [420.0, 360.0]   # right shoulder (span = 200px)
    kpts[10] = [385.0, 235.0]  # right wrist right next to right ear!

    confs = np.ones(17, dtype=np.float32)

    # Need 3 frames to filter glitches
    s1 = detector.update(head_forward, kpts, confs, timestamp=1.0)
    assert not s1.is_distracted
    s2 = detector.update(head_forward, kpts, confs, timestamp=1.04)
    s3 = detector.update(head_forward, kpts, confs, timestamp=1.08)
    assert s3.is_distracted
    assert s3.active_type == DistractionType.PHONE_DISTRACTION
