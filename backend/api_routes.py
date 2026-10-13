"""REST API router for DrishtiDrive AI camera, settings, logs, and telemetry.

Decouples REST endpoint management from the main WebSocket streaming server,
ensuring clean separation of concerns and file sizes under ~200 lines.
"""
from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter
from pydantic import BaseModel

from backend.alert_catalog import AlertMessageCatalog
from backend.camera import CameraService
from backend.temporal_state import AlertStage

logger = logging.getLogger("drishtidrive.routes")


class CameraStartRequest(BaseModel):
    camera_index: Union[int, str] = 0
    resolution: Optional[str] = "1280x720"


class SettingsUpdateRequest(BaseModel):
    sensitivity: Optional[float] = None
    volume: Optional[float] = None
    language: Optional[str] = None


class AlertPreviewRequest(BaseModel):
    stage: int = 1


def create_api_router(server_state: Any) -> APIRouter:
    """Factory creating APIRouter bound to the application state container."""
    router = APIRouter(prefix="/api", tags=["core"])

    @router.get("/health")
    def get_health() -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "ok",
            "project": "DrishtiDrive AI",
            "tagline": "A Watchful Eye on Every Journey",
            "camera_active": server_state.is_camera_active,
            "model": server_state.config.inference.model_name,
        }

    @router.get("/telemetry")
    def get_telemetry() -> Dict[str, Any]:
        """Retrieve current instantaneous telemetry."""
        return server_state.last_telemetry

    @router.get("/camera/devices")
    def get_camera_devices() -> List[Dict[str, Any]]:
        """List available video input devices."""
        return [
            {"id": 0, "name": "Camera 0 (Default HD Webcam)"},
            {"id": "test_video", "name": "Demo Video Feed (reports/test_driver_feed.mp4)"},
        ]

    @router.post("/camera/start")
    def start_camera(req: CameraStartRequest) -> Dict[str, Any]:
        """Start live camera capture and processing pipeline."""
        if server_state.is_camera_active and getattr(server_state.pipeline, "_is_active", False) and server_state.selected_camera_index == req.camera_index:
            return {"status": "already_active", "camera_index": req.camera_index}

        if server_state.is_camera_active:
            server_state.pipeline.stop()
            server_state.is_camera_active = False

        server_state.selected_camera_index = req.camera_index
        idx_str = str(req.camera_index).lower()
        if "test" in idx_str or "demo" in idx_str or "file" in idx_str:
            server_state.config.camera.video_source = str(Path("reports/test_driver_feed.mp4").resolve())
        else:
            try:
                server_state.config.camera.camera_index = int(req.camera_index)
                server_state.config.camera.video_source = None
            except ValueError:
                server_state.config.camera.video_source = str(Path("reports/test_driver_feed.mp4").resolve())

        server_state.pipeline.camera = CameraService(server_state.config.camera)
        try:
            server_state.pipeline.start()
            server_state.is_camera_active = True
            server_state.last_telemetry["camera_status"] = "Active"
            server_state.logs.append({
                "timestamp": time.strftime("%H:%M:%S"),
                "event_type": "CAMERA_CONNECTED",
                "stage": 0,
                "description": f"Camera {req.camera_index} Connected ({req.resolution})",
                "measurements": "FPS Target: 30",
                "severity": "info",
            })
            return {"status": "started", "camera_index": req.camera_index}
        except Exception as exc:
            logger.warning("Camera open error: %s", exc)
            return {"status": "error", "error": str(exc)}

    @router.post("/camera/stop")
    def stop_camera() -> Dict[str, Any]:
        """Stop camera capture and release resources."""
        server_state.is_camera_active = False
        server_state.pipeline.stop()
        server_state.last_telemetry["camera_status"] = "Stopped"
        server_state.logs.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "event_type": "CAMERA_DISCONNECTED",
            "stage": 0,
            "description": "Camera capture stopped by operator",
            "measurements": "",
            "severity": "info",
        })
        return {"status": "stopped"}

    @router.get("/settings")
    def get_settings() -> Dict[str, Any]:
        """Retrieve active system parameters."""
        return {
            "camera": {"index": server_state.selected_camera_index, "resolution": server_state.resolution},
            "detection": {"sensitivity": 1.0, "single_frame_guard": True},
            "alerts": {
                "volume": server_state.config.alert.volume,
                "language": server_state.config.alert.language,
                "cooldown_sec": server_state.config.alert.cooldown_sec,
            },
        }

    @router.put("/settings")
    def update_settings(req: SettingsUpdateRequest) -> Dict[str, Any]:
        """Update runtime settings."""
        if req.volume is not None:
            server_state.pipeline.alert_manager.set_volume(req.volume)
        if req.language is not None:
            server_state.pipeline.alert_manager.set_language(req.language)
        return {"status": "updated"}

    @router.get("/logs")
    def get_logs(limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve event history."""
        all_logs = list(server_state.logs)
        return all_logs[-limit:]

    @router.delete("/logs")
    def clear_logs() -> Dict[str, Any]:
        """Clear event history."""
        server_state.logs.clear()
        return {"status": "cleared"}

    @router.post("/alerts/preview")
    def preview_alert(req: AlertPreviewRequest) -> Dict[str, Any]:
        """Trigger manual test of audio and visual alert stage."""
        target_stage = AlertStage(req.stage) if req.stage in (1, 2, 3) else AlertStage.STAGE_1_EARLY
        server_state.pipeline.alert_manager.trigger_manual_test(target_stage)
        voice_msg = AlertMessageCatalog.get_message(target_stage, server_state.config.alert.language)
        server_state.logs.append({
            "timestamp": time.strftime("%H:%M:%S"),
            "event_type": "MANUAL_ALERT_TEST",
            "stage": target_stage.value,
            "description": f"Manual Audio-Visual Test: Stage {target_stage.value}",
            "measurements": f'"{voice_msg[:40]}..."',
            "severity": "warning" if target_stage.value > 1 else "info",
        })
        return {"status": "preview_triggered", "stage": target_stage.value}

    return router
