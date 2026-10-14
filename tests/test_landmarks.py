"""Unit tests for MediaPipe 3D Face Landmarker and physical EAR calculation."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import pytest

from backend.landmarks import EyeLandmarksResult, MediaPipeFaceLandmarker


def test_ear_calculation_synthetic_open_and_closed():
    """Verify standard Soukupová-Čech formula on synthetic geometric coordinates."""
    # 1. Closed eye: horizontal width 30px, vertical slit 2px
    closed_pts = [
        (0.0, 10.0),   # p1 outer corner
        (10.0, 9.0),   # p2 upper outer
        (20.0, 9.0),   # p3 upper inner
        (30.0, 10.0),  # p4 inner corner
        (20.0, 11.0),  # p5 lower inner
        (10.0, 11.0),  # p6 lower outer
    ]
    ear_closed = MediaPipeFaceLandmarker._compute_ear(closed_pts)
    assert 0.05 <= ear_closed <= 0.10, f"Expected closed EAR ~0.067, got {ear_closed}"

    # 2. Open eye: horizontal width 30px, vertical opening 10px
    open_pts = [
        (0.0, 10.0),   # p1
        (10.0, 5.0),   # p2
        (20.0, 5.0),   # p3
        (30.0, 10.0),  # p4
        (20.0, 15.0),  # p5
        (10.0, 15.0),  # p6
    ]
    ear_open = MediaPipeFaceLandmarker._compute_ear(open_pts)
    assert 0.28 <= ear_open <= 0.38, f"Expected open EAR ~0.333, got {ear_open}"


def test_extract_eye_crop_bounding_and_padding():
    """Verify crop extraction adds anatomical padding and clamps to image bounds."""
    frame = np.ones((200, 200, 3), dtype=np.uint8) * 128
    pts = [(50.0, 50.0), (60.0, 45.0), (70.0, 45.0), (80.0, 50.0), (70.0, 55.0), (60.0, 55.0)]
    crop = MediaPipeFaceLandmarker._extract_eye_crop(frame, pts, 200, 200)

    assert crop is not None
    assert crop.shape[0] > (55 - 45)  # Height padded beyond min-max
    assert crop.shape[1] > (80 - 50)  # Width padded beyond min-max


def test_landmarker_fallback_graceful():
    """Verify landmarker gracefully reports unavailable when model path is invalid."""
    detector = MediaPipeFaceLandmarker(model_path="non_existent_model.task")
    assert not detector.is_available
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    res = detector.detect(dummy_frame)
    assert not res.detected
    assert "unavailable" in res.message.lower()


def test_landmarker_real_video_feed():
    """Verify landmarker extracts valid landmarks and blendshapes from real driver video."""
    video_path = Path("reports/test_driver_feed.mp4")
    if not video_path.exists():
        pytest.skip("reports/test_driver_feed.mp4 not found.")

    landmarker = MediaPipeFaceLandmarker()
    if not landmarker.is_available:
        pytest.skip("MediaPipe task asset face_landmarker.task not available.")

    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read first frame from driver video."

    res = landmarker.detect(frame)
    assert res.detected, "Landmarker failed to detect face in real driver feed."
    assert 0.05 <= res.mean_ear <= 0.40
    assert 0.0 <= res.mean_blink <= 1.0
    assert len(res.left_eye_points) == 6
    assert len(res.right_eye_points) == 6
    assert res.left_eye_crop is not None
    assert res.right_eye_crop is not None
