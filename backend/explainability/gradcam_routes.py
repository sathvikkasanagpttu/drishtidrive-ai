"""
Grad-CAM Explainability Developer Routes
========================================
Exposes developer diagnostic endpoints for visualizing eye classifier attention
heatmaps using Gradient-weighted Class Activation Mapping (Grad-CAM).
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, Optional
import cv2
import numpy as np
import torch
from fastapi import APIRouter
from pydantic import BaseModel

from backend.explainability.gradcam import GradCAM
from backend.training.models import DepthwiseSeparableCNN

logger = logging.getLogger("drishtidrive.api.gradcam")

router = APIRouter(prefix="/api/dev", tags=["developer-diagnostics"])

# Cache persistent GradCAM instance
_gradcam_instance: Optional[GradCAM] = None


def get_gradcam_engine() -> GradCAM:
    """Initialize or retrieve cached GradCAM engine on DepthwiseSeparableCNN."""
    global _gradcam_instance
    if _gradcam_instance is None:
        model = DepthwiseSeparableCNN(num_classes=2)
        model_path = "backend/models/depthwise_cnn.pt"
        if os.path.exists(model_path):
            try:
                ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict, strict=False)
            except Exception as exc:
                logger.warning("Loaded base weights with warning: %s", exc)
        model.eval()
        # Final conv block is the last DepthwiseSeparableBlock pointwise conv
        target_layer = model.blocks[-1].pointwise
        _gradcam_instance = GradCAM(model, target_layer)
    return _gradcam_instance


class GradCAMRequest(BaseModel):
    crop_b64: Optional[str] = None
    target_class: Optional[int] = None


@router.get("/gradcam")
def get_sample_gradcam(state: str = "closed") -> Dict[str, Any]:
    """
    Generate Grad-CAM explanation on a synthetic eye sample (closed or open).
    Demonstrates model feature focus for developer inspection.
    """
    engine = get_gradcam_engine()

    # Generate synthetic eye patch (64x64)
    patch = np.full((64, 64, 3), 120, dtype=np.uint8)
    if state.lower() == "closed":
        # Draw horizontal dark eyelid slit
        cv2.line(patch, (10, 32), (54, 32), (30, 30, 30), 4)
    else:
        # Draw open pupil and iris circle
        cv2.circle(patch, (32, 32), 16, (200, 200, 200), -1)
        cv2.circle(patch, (32, 32), 8, (20, 20, 20), -1)

    result = engine.explain_crop_b64(patch)
    result["sample_type"] = state
    return result


@router.post("/gradcam/explain")
def explain_custom_crop(req: GradCAMRequest) -> Dict[str, Any]:
    """
    Generate Grad-CAM explanation on user-submitted Base64 eye crop.
    """
    engine = get_gradcam_engine()

    if req.crop_b64:
        try:
            raw_bytes = base64.b64decode(req.crop_b64)
            np_arr = np.frombuffer(raw_bytes, np.uint8)
            crop_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if crop_bgr is None:
                raise ValueError("Could not decode image")
        except Exception:
            crop_bgr = np.full((64, 64, 3), 128, dtype=np.uint8)
    else:
        crop_bgr = np.full((64, 64, 3), 128, dtype=np.uint8)

    return engine.explain_crop_b64(crop_bgr, target_class=req.target_class)
