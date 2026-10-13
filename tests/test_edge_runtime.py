"""
Edge Runtime Unit Tests
=======================
Tests for GPIO actuator (mock mode), camera tamper detector,
systemd watchdog notification, edge headless runner, and ONNX inference.
"""

import os
import numpy as np
import pytest

from backend.edge.gpio_actuator import GPIOActuator
from backend.edge.tamper_detector import TamperDetector, TamperStatus
from backend.edge.edge_runner import EdgeWatchdog, EdgeRuntime
from backend.temporal_state import AlertStage


def test_gpio_actuator_mock_lifecycle():
    """Verify mock GPIO transitions and clean shutdown."""
    actuator = GPIOActuator(mock_hardware=True)
    assert actuator.is_mock is True

    # Test state transitions
    actuator.set_state(AlertStage.STAGE_0_NORMAL)
    assert actuator.get_current_stage() == AlertStage.STAGE_0_NORMAL

    actuator.set_state(AlertStage.STAGE_1_EARLY)
    assert actuator.get_current_stage() == AlertStage.STAGE_1_EARLY

    actuator.set_state(AlertStage.STAGE_2_DROWSY)
    assert actuator.get_current_stage() == AlertStage.STAGE_2_DROWSY

    actuator.set_state(AlertStage.STAGE_3_CRITICAL)
    assert actuator.get_current_stage() == AlertStage.STAGE_3_CRITICAL

    # Self test
    assert actuator.self_test() is True

    # Cleanup
    actuator.cleanup()
    assert actuator.get_current_stage() == AlertStage.STAGE_0_NORMAL


def test_tamper_detector_normal_frame():
    """Ensure normal high-texture frames do not trigger tamper alarms."""
    detector = TamperDetector(consecutive_frames_required=3)

    # Synthetic normal frame with contrast and texture
    np.random.seed(42)
    normal_frame = np.random.randint(40, 200, (100, 100, 3), dtype=np.uint8)

    status = None
    for _ in range(5):
        status = detector.analyze(normal_frame)

    assert status is not None
    assert status.is_tampered is False
    assert status.tamper_type is None
    assert status.blur_score > 6.0


def test_tamper_detector_occluded_black_frame():
    """Ensure pitch-black frames trigger occlusion tamper alarm after streak."""
    detector = TamperDetector(consecutive_frames_required=3)
    black_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Frame 1 and 2: accumulating streak
    s1 = detector.analyze(black_frame)
    assert s1.is_tampered is False
    assert s1.consecutive_frames == 1

    s2 = detector.analyze(black_frame)
    assert s2.is_tampered is False

    # Frame 3: threshold reached
    s3 = detector.analyze(black_frame)
    assert s3.is_tampered is True
    assert s3.tamper_type == "OCCLUDED"


def test_tamper_detector_blinded_and_defocused():
    """Ensure bright saturation and uniform flat blur trigger respective alarms."""
    detector = TamperDetector(consecutive_frames_required=2)

    # Blinding saturation
    bright_frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    detector.analyze(bright_frame)
    s_bright = detector.analyze(bright_frame)
    assert s_bright.is_tampered is True
    assert s_bright.tamper_type == "BLINDED"

    detector.reset()

    # Uniform grey frame (Laplacian variance = 0)
    flat_frame = np.full((100, 100, 3), 128, dtype=np.uint8)
    detector.analyze(flat_frame)
    s_flat = detector.analyze(flat_frame)
    assert s_flat.is_tampered is True
    assert s_flat.tamper_type == "DEFOCUSED"


def test_edge_watchdog_socket_fallback(monkeypatch):
    """Verify watchdog behavior without active systemd socket."""
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    dog = EdgeWatchdog(interval_sec=1.0)
    assert dog.send_notify("WATCHDOG=1") is False
    dog.heartbeat()  # Should safely no-op without exception


def test_edge_runtime_self_test():
    """Verify startup self-test report generation."""
    runtime = EdgeRuntime(camera_index=999, accelerator="cpu", mock_gpio=True)
    report = runtime.run_self_test()

    assert "model_weights" in report
    assert "gpio_actuators" in report
    assert report["gpio_actuators"] is True
    assert report["accelerator"] is True
