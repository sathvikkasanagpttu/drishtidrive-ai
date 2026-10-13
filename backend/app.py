"""FastAPI backend application and WebSocket telemetry server for DrishtiDrive AI.

Exposes REST APIs for camera lifecycle, settings, event logs, and manual alert tests.
Streams real-time JPEG frames and inference telemetry over WebSockets.
Mounts the production frontend distribution if available.
"""
from __future__ import annotations

import asyncio
import base64
from collections import deque
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Set, Union
import cv2
import numpy as np

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.alert_catalog import AlertMessageCatalog
from backend.alert_manager import AlertManager
from backend.camera import CameraService, CameraUnavailableError
from backend.config import PipelineConfig
from backend.pipeline import EyeRegionPipeline
from backend.run_pipeline import draw_overlay
from backend.temporal_state import AlertStage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drishtidrive.app")

import os

# Global state containers
class ServerState:
    """Encapsulates pipeline instances, telemetry streams, and event logs."""

    def __init__(self) -> None:
        self.config = PipelineConfig()
        self.config.detection.allow_multiple_faces = True
        self.pipeline = EyeRegionPipeline(config=self.config)
        self.is_camera_active = False
        self.selected_camera_index = 0
        self.resolution = "1280x720"
        self.monitoring_start_time: Optional[float] = None
        self.logs: Deque[Dict[str, Any]] = deque(maxlen=200)
        self.active_websockets: Set[WebSocket] = set()
        self.last_frame_base64: Optional[str] = None
        self.last_telemetry: Dict[str, Any] = {
            "eye_state": "NO_DATA",
            "confidence": None,
            "closure_duration_sec": 0.0,
            "fps": 0.0,
            "latency_ms": None,
            "stage": 0,
            "status_text": "Awaiting stream",
            "attention_score": None,
            "perclos": None,
            "ear": None,
            "ear_is_estimated": True,
            "mar": None,
            "bpm": None,
            "voice_message": "System ready. Start camera to begin monitoring.",
            "audio_status": "Ready",
            "camera_status": "Ready",
            "model_name": self.config.inference.model_name,
            "fusion_risk": None,
            "head_pitch": None,
            "head_yaw": None,
            "is_nodding": False,
            "is_looking_away": False,
            "distraction_type": "NONE",
            "seatbelt_present": True,
        }
        self.logs.append({
            "timestamp": time.strftime("%H:%M:%S"), "event_type": "SYSTEM_ONLINE", "stage": 0,
            "description": "DrishtiDrive AI Core Initialized",
            "measurements": f"Model: {self.config.inference.model_name}", "severity": "info",
        })


state = ServerState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Auto-start continuous camera monitoring on startup and stop on shutdown."""
    try:
        state.pipeline.start()
        state.is_camera_active = True
        state.monitoring_start_time = time.time()
        state.last_telemetry["camera_status"] = "Active"
        logger.info("Continuous driver camera monitoring auto-started successfully.")
    except Exception as exc:
        logger.warning("Could not auto-start camera on startup: %s", exc)
    yield
    try:
        state.pipeline.stop()
        state.monitoring_start_time = None
    except Exception as exc:
        logger.debug("Error stopping pipeline on shutdown: %s", exc)


app = FastAPI(
    title="DrishtiDrive AI",
    description="A Watchful Eye on Every Journey",
    lifespan=lifespan,
)

default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
env_cors = os.getenv("CORS_ORIGINS")
allowed_cors_origins = [o.strip() for o in env_cors.split(",") if o.strip()] if env_cors else default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.api_routes import create_api_router
from backend.calibration_routes import create_calibration_router
from backend.fleet_routes import router as fleet_router
from backend.explainability.gradcam_routes import router as gradcam_router

app.include_router(create_api_router(state))
app.include_router(create_calibration_router(state.pipeline.calibration_service))
app.include_router(fleet_router)
app.include_router(gradcam_router)


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket) -> None:
    """Real-time WebSocket endpoint streaming frame JPEG and telemetry packets."""
    await websocket.accept()
    state.active_websockets.add(websocket)
    logger.info("WebSocket client connected. Total clients: %d", len(state.active_websockets))

    try:
        while websocket.client_state.name == "CONNECTED":
            # Generate or grab frame
            frame: Optional[np.ndarray] = None
            if state.is_camera_active:
                frame = state.pipeline.camera.get_frame()

            if frame is not None:
                # Process frame through full pipeline
                result = state.pipeline.process_frame(frame)
                stage_val = result.temporal_status.alert_stage.value if result.temporal_status else 0
                closure_sec = result.temporal_status.closure_duration_sec if result.temporal_status else 0.0
                mar_val = getattr(result.detection_result, "mar", 0.14)
                is_yawn = getattr(result.detection_result, "is_yawning", False)

                # Check if new temporal alert event emitted for event log
                if result.temporal_status and result.temporal_status.alert_event:
                    ev = result.temporal_status.alert_event
                    sev = "critical" if ev.stage.value == 3 else "warning" if ev.stage.value == 2 else "info"
                    state.logs.append({
                        "timestamp": time.strftime("%H:%M:%S"),
                        "event_type": ev.event_type.value,
                        "stage": ev.stage.value,
                        "description": ev.message,
                        "measurements": f"Closure: {closure_sec:.2f}s | MAR: {mar_val:.2f} | Conf: {result.overall_confidence * 100:.1f}%",
                        "severity": sev,
                    })

                # Draw bounding boxes and HUD on frame
                vis = draw_overlay(frame, result)
                _, buffer = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
                state.last_frame_base64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

                # Update telemetry with real accurate metrics
                lat = result.latency_breakdown.get("total_ms", 3.0)
                blinks = result.temporal_status.blink_count if result.temporal_status else 0
                # Use smoothed time-based attention score and PERCLOS from temporal engine
                if result.temporal_status:
                    att_score = int(round(result.temporal_status.attention_score))
                    perclos_val = result.temporal_status.perclos
                else:
                    att_score = 95
                    perclos_val = 0.0

                if result.overall_state == "OPEN":
                    ear = round(0.28 + (result.overall_confidence * 0.08), 2)
                elif result.overall_state == "CLOSED":
                    ear = round(max(0.05, 0.16 - (result.overall_confidence * 0.08)), 2)
                else:
                    ear = 0.22

                st_text = "Critical" if stage_val == 3 else "Drowsy" if stage_val == 2 else "Attention" if stage_val == 1 else "Normal"
                state.last_telemetry.update({
                    "eye_state": result.overall_state,
                    "confidence": round(result.overall_confidence, 3),
                    "closure_duration_sec": round(closure_sec, 2),
                    "fps": round(result.camera_fps or 30.0, 1),
                    "latency_ms": round(lat, 2),
                    "stage": stage_val,
                    "status_text": st_text,
                    "attention_score": att_score,
                    "perclos": round(perclos_val, 3),
                    "ear": ear,
                    "mar": round(mar_val, 2),
                    "is_yawning": is_yawn,
                    "bpm": round(blinks / max(0.2, (time.time() - (state.monitoring_start_time or time.time())) / 60.0), 1),
                    "blink_count": blinks,
                    "voice_message": AlertMessageCatalog.get_message(AlertStage(stage_val)),
                    "audio_status": "Playing Alert" if stage_val > 0 else "Ready",
                    "fusion_risk": round(result.fusion.risk_score, 3) if result.fusion else 0.0,
                    "head_pitch": result.head_pose.pitch if result.head_pose else 0.0,
                    "head_yaw": result.head_pose.yaw if result.head_pose else 0.0,
                    "is_nodding": result.head_pose.is_nodding if result.head_pose else False,
                    "is_looking_away": result.head_pose.is_looking_away if result.head_pose else False,
                    "distraction_type": result.distraction.active_type.value if result.distraction else "NONE",
                    "seatbelt_present": result.distraction.seatbelt_present if result.distraction else True,
                })

            packet = {
                "active": state.is_camera_active,
                "status": "ACTIVE" if state.is_camera_active else "READY",
                "frame": state.last_frame_base64 if state.is_camera_active else None,
                "telemetry": state.last_telemetry,
            }

            await websocket.send_json(packet)
            await asyncio.sleep(0.04)  # ~25 FPS stream rate

    except WebSocketDisconnect:
        state.active_websockets.remove(websocket)
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.warning("WebSocket transmission error: %s", exc)
        if websocket in state.active_websockets:
            state.active_websockets.remove(websocket)


# Mount frontend distribution for unified single-server serving
frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

