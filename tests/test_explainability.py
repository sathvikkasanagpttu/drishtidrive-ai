"""
Phase 5 Tests: Grad-CAM Explainability & Trip Safety Summary Engine
===================================================================
Tests Gradient-weighted Class Activation Mapping (Grad-CAM) saliency maps,
attention heatmaps, and post-trip safety heuristic summaries.
"""

import numpy as np
import pytest
import torch

from backend.explainability.gradcam import GradCAM
from backend.explainability.trip_summary import TripSummaryEngine
from backend.training.models import DepthwiseSeparableCNN


def test_gradcam_saliency_generation():
    """Verify Grad-CAM produces normalized 2D saliency maps and overlays."""
    model = DepthwiseSeparableCNN(num_classes=2)
    model.eval()
    target_layer = model.blocks[-1].pointwise

    gradcam = GradCAM(model, target_layer)

    # Test synthetic eye crop
    crop = np.full((64, 64, 3), 128, dtype=np.uint8)
    explanation = gradcam.explain_crop_b64(crop)

    assert "predicted_class" in explanation
    assert explanation["predicted_class"] in ("Closed", "Open")
    assert "confidence" in explanation
    assert 0.0 <= explanation["confidence"] <= 1.0
    assert "raw_crop_b64" in explanation
    assert "overlay_b64" in explanation
    assert "focus_intensity" in explanation

    gradcam.close()


def test_trip_summary_engine_heuristics():
    """Verify post-trip summary metrics and safety recommendation generation."""
    events = [
        {"stage": 1, "duration_sec": 1.1, "event_type": "EAR_EARLY_WARNING", "acknowledged": True},
        {"stage": 2, "duration_sec": 2.1, "event_type": "DROWSINESS_STAGE_2", "acknowledged": True},
        {"stage": 3, "duration_sec": 3.4, "event_type": "CRITICAL_CLOSURE", "acknowledged": False},
        {"stage": 1, "duration_sec": 0.9, "event_type": "DISTRACTION_PHONE", "acknowledged": True},
    ]

    report = TripSummaryEngine.compile_report(
        trip_id="trip_test_001",
        driver_id="driver_KA04",
        events=events,
        trip_duration_sec=7400.0,  # > 2 hours
        average_speed_kmh=58.0,
    )

    assert report["trip_id"] == "trip_test_001"
    assert report["total_alerts"] == 4
    assert report["stage_breakdown"][3] == 1
    assert report["longest_eye_closure_sec"] == 3.4
    assert report["distraction_events"] == 1
    assert report["risk_score"] > 30.0  # Stage 3 heavily weights risk

    # Verify recommendations contain fatigue and duration advisories
    recs = " ".join(report["recommendations"])
    assert "CRITICAL" in recs or "micro-sleep" in recs
    assert "2 hours" in recs
