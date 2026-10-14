"""Unit tests for multi-modal eye state decision fusion engine."""
from __future__ import annotations

import pytest
from backend.eye_state_fusion import fuse_eye_states
from backend.model_inference import EyeStatePrediction


def make_pred(state: str, conf: float) -> EyeStatePrediction:
    """Helper to construct synthetic EyeStatePrediction."""
    p_open = conf if state == "OPEN" else (1.0 - conf)
    p_closed = conf if state == "CLOSED" else (1.0 - conf)
    return EyeStatePrediction(
        state=state,
        confidence=conf,
        probabilities={"open": p_open, "closed": p_closed},
        latency_ms=1.0,
        model_name="synthetic",
    )


def test_clear_open_landmarks():
    """Verify clearly open landmarks and open CNN produce confident OPEN."""
    res = fuse_eye_states(
        mean_ear=0.32,
        mean_blink=0.05,
        left_cnn=make_pred("OPEN", 0.90),
        right_cnn=make_pred("OPEN", 0.92),
    )
    assert res.state == "OPEN"
    assert res.confidence >= 0.85
    assert res.landmark_driven


def test_clear_closed_landmarks():
    """Verify clearly closed landmarks produce confident CLOSED."""
    res = fuse_eye_states(
        mean_ear=0.10,
        mean_blink=0.85,
        left_cnn=make_pred("CLOSED", 0.88),
        right_cnn=make_pred("CLOSED", 0.90),
    )
    assert res.state == "CLOSED"
    assert res.confidence >= 0.85
    assert res.landmark_driven


def test_disagreement_strong_returns_uncertain():
    """Verify strong contradictions between landmarks and CNN result in UNCERTAIN."""
    # Contradiction A: Landmark clearly open (EAR 0.30), CNN firmly closed (0.92)
    res_a = fuse_eye_states(
        mean_ear=0.30,
        mean_blink=0.05,
        left_cnn=make_pred("CLOSED", 0.92),
        right_cnn=make_pred("CLOSED", 0.90),
    )
    assert res_a.state == "UNCERTAIN"
    assert "conflict" in res_a.reason.lower()

    # Contradiction B: Landmark clearly closed (EAR 0.08), CNN firmly open (0.92)
    res_b = fuse_eye_states(
        mean_ear=0.08,
        mean_blink=0.90,
        left_cnn=make_pred("OPEN", 0.92),
        right_cnn=make_pred("OPEN", 0.91),
    )
    assert res_b.state == "UNCERTAIN"
    assert "conflict" in res_b.reason.lower()


def test_borderline_ear_resolved_by_confident_cnn():
    """Verify borderline EAR (within 10% of threshold) can be resolved to OPEN by strong CNN."""
    # Threshold = 0.20; 10% band is [0.18, 0.20]. EAR = 0.19
    res = fuse_eye_states(
        mean_ear=0.19,
        mean_blink=0.20,
        left_cnn=make_pred("OPEN", 0.90),
        right_cnn=make_pred("OPEN", 0.88),
        ear_close_threshold=0.20,
    )
    assert res.state == "OPEN"
    assert "borderline" in res.reason.lower()


def test_cannot_override_clearly_closed_ear():
    """Verify high-confidence CNN OPEN CANNOT override a clearly closed physical EAR."""
    # EAR = 0.14 is way below 0.20 * 0.90 (0.18)
    res = fuse_eye_states(
        mean_ear=0.14,
        mean_blink=0.40,
        left_cnn=make_pred("OPEN", 0.86),
        right_cnn=make_pred("OPEN", 0.85),
        ear_close_threshold=0.20,
    )
    assert res.state == "CLOSED"
    assert "closure confirmed" in res.reason.lower()


def test_glasses_discount_cnn_in_favor_of_landmarks():
    """Verify eyeglasses reflections do not cause false closures when landmarks show open eye."""
    res = fuse_eye_states(
        mean_ear=0.28,
        mean_blink=0.08,
        left_cnn=make_pred("CLOSED", 0.88),  # Glare fooled CNN
        right_cnn=make_pred("CLOSED", 0.85),
        has_glasses=True,
    )
    assert res.state == "OPEN"


def test_one_eye_closed_unilateral():
    """Verify unilateral closure (winking / ptosis / microsleep) is caught as CLOSED."""
    res = fuse_eye_states(
        left_ear=0.08,
        right_ear=0.30,
        mean_ear=0.19,
        blink_left=0.85,
        blink_right=0.10,
    )
    assert res.state == "CLOSED"
    assert "unilateral" in res.reason.lower()


def test_head_pitched_down_elevates_threshold():
    """Verify forward head droop (nodding off) raises sensitivity to eye closure."""
    # EAR = 0.21 is normally open (> 0.20), but when head pitch > 16 deg, threshold is 0.22
    res_normal = fuse_eye_states(mean_ear=0.21, head_pitch=0.0, ear_close_threshold=0.20, ear_open_threshold=0.23)
    res_nodding = fuse_eye_states(mean_ear=0.21, head_pitch=20.0, ear_close_threshold=0.20, ear_open_threshold=0.23)

    assert res_nodding.state == "CLOSED"
    assert res_normal.state != "CLOSED"
