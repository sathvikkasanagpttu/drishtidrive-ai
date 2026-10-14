"""Unit tests for driver calibration service and API endpoints.

Verifies:
- 30-second calibration cycle lifecycle (IDLE -> CALIBRATING -> COMPLETED).
- Local filesystem persistence in driver_profile.json (zero cloud transmission).
- Extraction of driver-specific baselines (eye confidence, blink rate, neutral pitch/yaw).
- Dynamic threshold adaptation and profile reset.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from backend.calibration import CalibrationState, DriverCalibrationService, DriverProfile


@pytest.fixture
def tmp_profile_path(tmp_path: Path) -> Path:
    """Return isolated temporary profile path for test hermeticity."""
    return tmp_path / "test_driver_profile.json"


@pytest.fixture
def calibration_svc(tmp_profile_path: Path) -> DriverCalibrationService:
    """Instantiate calibration service with temporary profile path."""
    return DriverCalibrationService(profile_path=tmp_profile_path, calibration_duration_sec=3.0)


def test_calibration_lifecycle_and_persistence(calibration_svc: DriverCalibrationService) -> None:
    """Verify complete calibration session computes baselines and saves to disk."""
    assert calibration_svc.state == CalibrationState.IDLE
    calibration_svc.start_calibration(driver_id="trucker_42")
    assert calibration_svc.state == CalibrationState.CALIBRATING

    # Feed 35 frames across 3.5 seconds
    t = 100.0
    for i in range(35):
        is_blink = (i % 10 == 0)
        state = "CLOSED" if is_blink else "OPEN"
        calibration_svc.update(
            eye_state=state,
            confidence=0.92,
            pitch=3.5,
            yaw=-2.0,
            roll=0.5,
            is_blink=is_blink,
            timestamp=t,
        )
        t += 0.1

    assert calibration_svc.state == CalibrationState.COMPLETED
    assert calibration_svc.profile is not None
    assert calibration_svc.profile.driver_id == "trucker_42"
    assert calibration_svc.profile_path.exists()

    # Read back from disk to verify hermetic serialization
    with open(calibration_svc.profile_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["driver_id"] == "trucker_42"
    assert data["baseline_open_conf"] > 0.85
    assert data["neutral_pitch"] == pytest.approx(3.5, 0.5)

    # Test reset
    calibration_svc.reset()
    assert calibration_svc.state == CalibrationState.IDLE
    assert calibration_svc.profile is None
    assert not calibration_svc.profile_path.exists()


def test_guided_calibration_phases_and_ear_baselines(tmp_profile_path: Path) -> None:
    """Verify guided phases collect distinct open and closed EAR baselines."""
    svc = DriverCalibrationService(profile_path=tmp_profile_path, calibration_duration_sec=2.0)
    svc.start_calibration(driver_id="driver_guided")

    t = 10.0
    # Phase 1: Open eyes (0.0 to 1.4s)
    for _ in range(14):
        phase, instr = svc.get_phase(t)
        assert phase == "OPEN_PHASE"
        assert "OPEN" in instr
        svc.update(
            eye_state="OPEN",
            confidence=0.95,
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            is_blink=False,
            timestamp=t,
            mean_ear=0.32,
            mean_blink=0.05,
        )
        t += 0.1

    # Phase 2: Closed eyes (1.4 to 1.9s)
    for _ in range(6):
        phase, instr = svc.get_phase(t)
        assert phase == "CLOSED_PHASE"
        assert "CLOSE" in instr
        svc.update(
            eye_state="CLOSED",
            confidence=0.90,
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            is_blink=False,
            timestamp=t,
            mean_ear=0.10,
            mean_blink=0.85,
        )
        t += 0.1

    # Final step at >= 2.0s completes calibration
    svc.update(
        eye_state="CLOSED",
        confidence=0.90,
        pitch=0.0,
        yaw=0.0,
        roll=0.0,
        is_blink=False,
        timestamp=t,
        mean_ear=0.10,
        mean_blink=0.85,
    )

    assert svc.state == CalibrationState.COMPLETED
    assert svc.profile is not None
    assert svc.profile.open_ear_baseline == pytest.approx(0.32, 0.02)
    assert svc.profile.closed_ear_baseline == pytest.approx(0.10, 0.02)
    assert svc.profile.ear_close_threshold == pytest.approx(0.32 * 0.70, 0.02)


def test_hysteresis_eye_closure_behavior(tmp_profile_path: Path) -> None:
    """Verify calibrated hysteresis prevents flickering within the boundary band."""
    svc = DriverCalibrationService(profile_path=tmp_profile_path)
    # Mock profile with close=0.21, open=0.25
    svc.profile = DriverProfile(
        driver_id="test_hysteresis",
        open_ear_baseline=0.30,
        closed_ear_baseline=0.10,
        ear_close_threshold=0.21,
        ear_open_threshold=0.25,
        blink_close_threshold=0.55,
        blink_open_threshold=0.40,
    )

    # 1. Clear open: EAR 0.28
    assert not svc.is_eye_closed_calibrated(mean_ear=0.28, mean_blink=0.05, prev_closed=False)

    # 2. Drop below close threshold: EAR 0.19 -> triggers CLOSED
    assert svc.is_eye_closed_calibrated(mean_ear=0.19, mean_blink=0.10, prev_closed=False)

    # 3. In hysteresis band: EAR 0.23 (between 0.21 and 0.25)
    # If previously closed, MUST STAY CLOSED to avoid flickering
    assert svc.is_eye_closed_calibrated(mean_ear=0.23, mean_blink=0.20, prev_closed=True)
    # If previously open, MUST STAY OPEN
    assert not svc.is_eye_closed_calibrated(mean_ear=0.23, mean_blink=0.20, prev_closed=False)

    # 4. Clear open exit: EAR 0.26 above open threshold -> exits to OPEN
    assert not svc.is_eye_closed_calibrated(mean_ear=0.26, mean_blink=0.20, prev_closed=True)

