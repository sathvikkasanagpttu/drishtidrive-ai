"""
Fleet and Context REST API Endpoints
====================================
Exposes endpoints for vehicle speed telemetry, Stage 2 response acknowledgement,
offline event timeline retrieval, and fleet risk scoring.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from backend.context.gps_service import GPSService
from backend.context.response_check import DriverResponseChecker
from backend.fleet.fleet_store import FleetStore
from backend.fleet.fleet_sync import FleetSyncAdapter

logger = logging.getLogger("drishtidrive.api.fleet")

router = APIRouter(prefix="/api", tags=["fleet-and-context"])

# Global shared service singletons for API runtime
gps_service = GPSService()
response_checker = DriverResponseChecker(timeout_sec=5.0)
fleet_store = FleetStore()
fleet_sync = FleetSyncAdapter(store=fleet_store)


class SpeedUpdateRequest(BaseModel):
    speed_kmh: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class AcknowledgeRequest(BaseModel):
    source: str = "ui_button"


@router.post("/context/speed")
def update_speed_telemetry(req: SpeedUpdateRequest) -> Dict[str, Any]:
    """Update vehicle velocity and coordinates."""
    ctx = gps_service.update_telemetry(
        speed_kmh=req.speed_kmh,
        latitude=req.latitude,
        longitude=req.longitude,
    )
    return {
        "status": "success",
        "speed_kmh": ctx.speed_kmh,
        "is_moving": ctx.is_moving,
        "siren_allowed": ctx.siren_allowed,
        "threshold_scale": ctx.threshold_scale,
        "trip_duration_sec": ctx.trip_duration_sec,
        "break_recommended": ctx.break_recommended,
    }


@router.get("/context/status")
def get_context_status() -> Dict[str, Any]:
    """Retrieve instantaneous context metrics and driving duration."""
    ctx = gps_service.get_context()
    return {
        "speed_kmh": ctx.speed_kmh,
        "is_moving": ctx.is_moving,
        "siren_allowed": ctx.siren_allowed,
        "threshold_scale": ctx.threshold_scale,
        "trip_duration_sec": ctx.trip_duration_sec,
        "break_recommended": ctx.break_recommended,
        "latitude": ctx.latitude,
        "longitude": ctx.longitude,
    }


@router.post("/alerts/acknowledge")
def acknowledge_alert(req: AcknowledgeRequest) -> Dict[str, Any]:
    """Record driver acknowledgement for Stage 2 challenge."""
    accepted = response_checker.acknowledge(source=req.source)
    return {
        "status": "acknowledged" if accepted else "ignored_no_active_challenge",
        "accepted": accepted,
        "remaining_sec": response_checker.get_remaining_time(),
    }


@router.get("/alerts/response-status")
def get_response_status() -> Dict[str, Any]:
    """Check whether a Stage 2 attentiveness challenge is currently awaiting input."""
    return {
        "status": response_checker.status.value,
        "remaining_sec": round(response_checker.get_remaining_time(), 1),
    }


@router.get("/fleet/events")
def get_fleet_events(trip_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve safety events with decrypted metadata for dashboard timeline."""
    return fleet_store.get_trip_events(trip_id=trip_id, limit=limit)


@router.get("/fleet/driver-score")
def get_driver_score(driver_id: str = "driver_default") -> Dict[str, Any]:
    """Retrieve calculated safety risk score for driver."""
    score = fleet_store.calculate_driver_risk(driver_id=driver_id)
    return {
        "driver_id": driver_id,
        "risk_score": score,
        "rating": "LOW_RISK" if score < 25 else ("MODERATE_RISK" if score < 60 else "HIGH_RISK"),
    }


@router.post("/fleet/sync")
def sync_fleet_events() -> Dict[str, Any]:
    """Trigger synchronization of queued offline events."""
    return fleet_sync.sync_pending()
