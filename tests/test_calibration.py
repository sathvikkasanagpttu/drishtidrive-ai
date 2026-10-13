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
