"""FastAPI router for driver baseline calibration endpoints.

Manages individual driver profile calibration:
- POST /api/calibration/start: Initiate a 30-60 second calibration cycle
- GET  /api/calibration/status: Query real-time calibration progress & baselines
- POST /api/calibration/reset: Revert to uncalibrated factory defaults
"""
from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.calibration import CalibrationState, DriverCalibrationService


class StartCalibrationRequest(BaseModel):
    """Payload to trigger driver calibration."""

    driver_id: str = "default_driver"
    duration_sec: float = 30.0


def create_calibration_router(calibration_service: DriverCalibrationService) -> APIRouter:
    """Factory creating APIRouter bound to the active calibration service."""
    router = APIRouter(prefix="/api/calibration", tags=["calibration"])

    @router.post("/start")
    def start_calibration(req: StartCalibrationRequest) -> Dict[str, Any]:
        """Start or restart a driver calibration window."""
        calibration_service.calibration_duration_sec = req.duration_sec
        calibration_service.start_calibration(req.driver_id)
        return {
            "status": "started",
            "driver_id": req.driver_id,
            "duration_sec": req.duration_sec,
            "state": calibration_service.state.value,
        }

    @router.get("/status")
    def get_calibration_status() -> Dict[str, Any]:
        """Fetch current calibration lifecycle state and calibrated baselines."""
        import time
        now = time.time()
        progress = calibration_service.get_progress(now)
        profile = calibration_service.profile

        return {
            "state": calibration_service.state.value,
            "progress_percent": progress,
            "profile": {
                "driver_id": profile.driver_id,
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
