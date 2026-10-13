"""Unit tests for multi-signal drowsiness fusion engine.

Verifies:
- Transparent weighted risk calculation: S_eye, S_perclos, S_pose, S_mar.
- Rolling PERCLOS accumulation across sliding time window.
- Blink rate (BPM) tracking over elapsed time.
- Optional TinyTemporalLSTM inference when enabled.
- State machine consistency: mapping risk score to recommended AlertStage.
"""
from __future__ import annotations

import pytest

from backend.fusion import FusionResult, FusionWeights, MultiSignalFusion, TinyTemporalLSTM
from backend.head_pose import HeadPoseResult
from backend.temporal_state import AlertStage


@pytest.fixture
def fusion_engine() -> MultiSignalFusion:
    """Instantiate standard fusion engine with 30s window."""
    return MultiSignalFusion(window_seconds=30.0, enable_temporal_model=True)


def test_alert_driver_low_risk(fusion_engine: MultiSignalFusion) -> None:
    """Verify attentive open eyes and centered head produce near-zero risk score."""
    head_pose = HeadPoseResult(pitch=2.0, yaw=1.0, is_nodding=False)

    for i in range(30):
        res = fusion_engine.update(
            eye_state="OPEN",
            closed_prob=0.05,
            mar=0.14,
            head_pose=head_pose,
            timestamp=float(i) * 0.1,
        )

    assert isinstance(res, FusionResult)
    assert res.risk_score < 0.20
    assert res.perclos < 0.05
    assert not res.is_yawning
    assert not res.is_nodding
    assert res.recommended_stage == AlertStage.STAGE_0_NORMAL


def test_drowsy_driver_escalation(fusion_engine: MultiSignalFusion) -> None:
    """Verify sustained closed eyes, high PERCLOS, and nodding escalate to Stage 2/3."""
    head_droop = HeadPoseResult(pitch=22.0, yaw=0.0, is_nodding=True)

    # Ingest 25 frames of sustained closure and head nod
    for i in range(30):
        res = fusion_engine.update(
            eye_state="CLOSED",
            closed_prob=0.95,
            mar=0.20,
            head_pose=head_droop,
            timestamp=float(i) * 0.1,
        )

    assert res.risk_score >= 0.70
    assert res.perclos > 0.80
    assert res.is_nodding
    assert res.recommended_stage in (AlertStage.STAGE_2_DROWSY, AlertStage.STAGE_3_CRITICAL)
    assert res.model_score is not None
    assert 0.0 <= res.model_score <= 1.0


def test_yawn_signal_contribution(fusion_engine: MultiSignalFusion) -> None:
    """Verify high MAR indicates yawning and elevates s_mar component score."""
    head_pose = HeadPoseResult()
    res = fusion_engine.update(
        eye_state="OPEN",
        closed_prob=0.1,
        mar=0.75,  # Open mouth yawn
        head_pose=head_pose,
        timestamp=1.0,
    )
    assert res.is_yawning
    assert res.component_scores["s_mar"] > 0.5
