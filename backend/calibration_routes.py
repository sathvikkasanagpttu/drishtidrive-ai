"""FastAPI router for driver baseline calibration endpoints.

Manages individual driver profile calibration:
- POST /api/calibration/start: Initiate a 10-second guided calibration cycle
- GET  /api/calibration/status: Query real-time calibration progress, EAR & thresholds
- POST /api/calibration/reset: Revert to uncalibrated factory defaults
"""
from __future__ import annotations

import time
from typing import Any, Dict
from fastapi import APIRouter
from pydantic import BaseModel

from backend.calibration import DriverCalibrationService


class StartCalibrationRequest(BaseModel):
    """Payload to trigger driver calibration."""

    driver_id: str = "default_driver"
    duration_sec: float = 10.0


def create_calibration_router(calibration_service: DriverCalibrationService) -> APIRouter:
    """Factory creating APIRouter bound to the active calibration service."""
    router = APIRouter(prefix="/api/calibration", tags=["calibration"])

    @router.post("/start")
    def start_calibration(req: StartCalibrationRequest) -> Dict[str, Any]:
        """Start or restart a guided 10-second driver calibration window."""
        calibration_service.calibration_duration_sec = req.duration_sec
        calibration_service.start_calibration(req.driver_id)
        phase, instruction = calibration_service.get_phase(time.time())
        return {
            "status": "started",
            "driver_id": req.driver_id,
            "duration_sec": req.duration_sec,
            "state": calibration_service.state.value,
            "phase": phase,
            "instruction": instruction,
        }

    @router.get("/status")
    def get_calibration_status() -> Dict[str, Any]:
        """Fetch current calibration lifecycle state, progress, measured EAR and thresholds."""
        now = time.time()
        progress = calibration_service.get_progress(now)
        phase, instruction = calibration_service.get_phase(now)
        profile = calibration_service.profile

        return {
            "state": calibration_service.state.value,
            "progress_percent": progress,
            "phase": phase,
            "instruction": instruction,
            "measured_ear": round(calibration_service.last_measured_ear, 4),
            "measured_blink": round(calibration_service.last_measured_blink, 4),
            "is_calibrated": profile is not None,
            "profile": {
                "driver_id": profile.driver_id,
                "open_ear_baseline": profile.open_ear_baseline,
                "closed_ear_baseline": profile.closed_ear_baseline,
                "ear_close_threshold": profile.ear_close_threshold,
                "ear_open_threshold": profile.ear_open_threshold,
                "baseline_open_conf": profile.baseline_open_conf,
                "baseline_blink_rate": profile.baseline_blink_rate,
                "neutral_pitch": profile.neutral_pitch,
                "neutral_yaw": profile.neutral_yaw,
                "neutral_roll": profile.neutral_roll,
                "sample_count": profile.sample_count,
                "calibrated_at": profile.calibrated_at,
            } if profile else None,
        }

    @router.post("/reset")
    def reset_calibration() -> Dict[str, Any]:
        """Reset calibration profile and revert to default thresholds."""
        calibration_service.reset()
        return {"status": "reset", "state": calibration_service.state.value}

    return router
