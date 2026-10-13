"""Multi-signal fusion engine for driver drowsiness risk assessment.

Fuses multiple behavioral and physiological fatigue signals:
1. Instantaneous Eye State & Closure Probability (S_eye)
2. Rolling Window PERCLOS (Percentage of Eyelid Closure, S_perclos)
3. Head Pose & Nodding Drops (S_pose)
4. Mouth Aspect Ratio (MAR) & Yawn Duration (S_mar)
5. Rolling Blink Rate (blinks per minute)

Provides both a transparent weighted formula and an optional lightweight
temporal sequence model (TinyTemporalLSTM) while maintaining the 3-stage
deterministic state machine as the final safety layer.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from backend.head_pose import HeadPoseResult
from backend.temporal_state import AlertStage


@dataclass
class FusionWeights:
    """Configurable weights for transparent linear risk fusion."""

    w_eye: float = 0.40
    w_perclos: float = 0.25
    w_pose: float = 0.20
    w_mar: float = 0.15

    def normalize(self) -> None:
        """Ensure weights strictly sum to 1.0."""
        total = self.w_eye + self.w_perclos + self.w_pose + self.w_mar
        if total > 0:
            self.w_eye /= total
            self.w_perclos /= total
            self.w_pose /= total
            self.w_mar /= total


@dataclass
class FusionResult:
    """Structured assessment combining fused signals and explainability metrics."""

    risk_score: float = 0.0           # 0.0 (fully alert) to 1.0 (critical drowsiness)
    perclos: float = 0.0              # 0.0 to 1.0 ratio
    blink_rate: float = 0.0           # Blinks per minute
    is_yawning: bool = False
    mar: float = 0.14
    is_nodding: bool = False
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    component_scores: Dict[str, float] = field(default_factory=dict)
    recommended_stage: AlertStage = AlertStage.STAGE_0_NORMAL
    model_score: Optional[float] = None


class TinyTemporalLSTM(nn.Module):
    """Ultra-lightweight 2-layer sequence model for temporal drowsiness modeling."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16) -> None:
        """Initialize tiny LSTM cell (latency <0.5ms on edge CPU)."""
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass over sequence (batch_size, seq_len, 4)."""
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.fc(last_step)


class MultiSignalFusion:
    """Engine aggregating multi-modal cues into calibrated drowsiness scores."""

    def __init__(
        self,
        weights: Optional[FusionWeights] = None,
        window_seconds: float = 30.0,
        enable_temporal_model: bool = False,
    ) -> None:
        """Initialize fusion engine with history window and optional LSTM."""
        self.weights = weights or FusionWeights()
        self.weights.normalize()
        self.window_seconds = window_seconds
        self.enable_temporal_model = enable_temporal_model

        # Rolling sample buffer: (timestamp, closed_prob, is_closed, mar, is_yawn, pitch, is_nod)
        self._history: Deque[Tuple[float, float, bool, float, bool, float, bool]] = deque()
        self._blink_timestamps: Deque[float] = deque()
        self._was_closed_last_frame = False

        self._temporal_model: Optional[TinyTemporalLSTM] = None
        if enable_temporal_model:
            self._temporal_model = TinyTemporalLSTM()
            self._temporal_model.eval()

    def update(
        self,
        eye_state: str,
        closed_prob: float,
        mar: float,
        head_pose: HeadPoseResult,
        timestamp: float,
    ) -> FusionResult:
        """Ingest single frame metrics and compute fused drowsiness evaluation."""
        is_closed = (eye_state == "CLOSED")
        is_yawn = (mar >= 0.65)
        is_nod = head_pose.is_nodding

        # 1. Blink transition detector (transition from open to closed to open)
        if is_closed and not self._was_closed_last_frame:
            self._blink_timestamps.append(timestamp)
        self._was_closed_last_frame = is_closed

        # 2. Append to rolling window buffer
        self._history.append((timestamp, closed_prob, is_closed, mar, is_yawn, head_pose.pitch, is_nod))

        # Evict records older than window_seconds
        cutoff = timestamp - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()

        # 3. Calculate PERCLOS over window
        total_samples = len(self._history)
        closed_samples = sum(1 for item in self._history if item[2])
        perclos = (closed_samples / total_samples) if total_samples > 0 else 0.0

        # Scale PERCLOS to [0, 1] risk score: <0.15 is normal, >=0.40 is high fatigue
        s_perclos = float(np.clip((perclos - 0.12) / 0.28, 0.0, 1.0))

        # 4. Instantaneous eye closure score
        s_eye = float(np.clip(closed_prob if is_closed else closed_prob * 0.5, 0.0, 1.0))

        # 5. Head nodding / droop score
        s_pose = 1.0 if is_nod else float(np.clip(max(0.0, head_pose.pitch) / 25.0, 0.0, 1.0))

        # 6. Yawn fatigue score
        s_mar = float(np.clip((mar - 0.25) / 0.45, 0.0, 1.0))

        # 7. Rolling blink rate (blinks per minute)
        elapsed_sec = max(5.0, timestamp - self._history[0][0]) if self._history else 30.0
        blink_rate_bpm = (len(self._blink_timestamps) / (elapsed_sec / 60.0))

        # 8. Compute weighted transparent fusion risk score
        weighted_score = (
            self.weights.w_eye * s_eye
            + self.weights.w_perclos * s_perclos
            + self.weights.w_pose * s_pose
            + self.weights.w_mar * s_mar
        )
        final_risk = float(np.clip(weighted_score, 0.0, 1.0))

        # 9. Optional lightweight temporal LSTM inference
        model_score = None
        if self._temporal_model is not None and len(self._history) >= 15:
            model_score = self._run_temporal_model()

        # 10. Recommended stage mapping from fusion risk
        if final_risk >= 0.75 or (s_eye >= 0.85 and s_pose >= 0.80):
            recommended = AlertStage.STAGE_3_CRITICAL
        elif final_risk >= 0.48 or (s_eye >= 0.60 and s_perclos >= 0.50):
            recommended = AlertStage.STAGE_2_DROWSY
        elif final_risk >= 0.25:
            recommended = AlertStage.STAGE_1_EARLY
        else:
            recommended = AlertStage.STAGE_0_NORMAL

        component_scores = {
            "s_eye": round(s_eye, 3),
            "s_perclos": round(s_perclos, 3),
            "s_pose": round(s_pose, 3),
            "s_mar": round(s_mar, 3),
        }

        return FusionResult(
            risk_score=round(final_risk, 3),
            perclos=round(perclos, 3),
            blink_rate=round(blink_rate_bpm, 1),
            is_yawning=is_yawn,
            mar=round(mar, 2),
            is_nodding=is_nod,
            head_pitch=round(head_pose.pitch, 1),
            head_yaw=round(head_pose.yaw, 1),
            component_scores=component_scores,
            recommended_stage=recommended,
            model_score=round(model_score, 3) if model_score is not None else None,
        )

    def _run_temporal_model(self) -> float:
        """Execute tiny LSTM on latest 20 sequence feature vectors."""
        seq_items = list(self._history)[-20:]
        feats = [
            [item[1], 1.0 if item[2] else 0.0, item[3], max(0.0, item[5]) / 30.0]
            for item in seq_items
        ]
        # Pad to 20 if shorter
        while len(feats) < 20:
            feats.insert(0, [0.0, 0.0, 0.14, 0.0])

        tensor_in = torch.tensor([feats], dtype=torch.float32)
        with torch.no_grad():
            pred = self._temporal_model(tensor_in)
            return float(pred.item())

    def reset(self) -> None:
        """Clear rolling history buffers."""
        self._history.clear()
        self._blink_timestamps.clear()
        self._was_closed_last_frame = False
