"""Eye-state perception fusion engine for DrishtiDrive AI.

Fuses MediaPipe landmark-based Eye Aspect Ratio (EAR), facial blendshapes,
deep CNN predictions, 3D head pose pitch, and environmental quality flags.

Rules:
1. Landmark EAR and blendshape blink scores are primary ground truth.
2. CNN acts as secondary confirmation; cannot override physically closed EAR.
3. Strong signal conflicts result in UNCERTAIN (never silently default to OPEN).
4. Head down-pitch (nodding) increases sensitivity to eye closure.
5. Eyeglass reflections discount CNN texture in favor of landmark geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from backend.model_inference import EyeStatePrediction


@dataclass
class EyeFusionResult:
    """Consolidated state decision with audit reasoning."""

    state: str  # "OPEN", "CLOSED", "UNCERTAIN"
    confidence: float
    reason: str
    mean_ear: float
    mean_blink: float
    landmark_driven: bool


def fuse_eye_states(
    left_ear: Optional[float] = None,
    right_ear: Optional[float] = None,
    mean_ear: Optional[float] = None,
    blink_left: Optional[float] = None,
    blink_right: Optional[float] = None,
    mean_blink: Optional[float] = None,
    left_cnn: Optional[EyeStatePrediction] = None,
    right_cnn: Optional[EyeStatePrediction] = None,
    head_pitch: float = 0.0,
    has_glasses: bool = False,
    is_poor_quality: bool = False,
    ear_close_threshold: float = 0.20,
    ear_open_threshold: float = 0.23,
    blink_close_threshold: float = 0.55,
    prev_closed: bool = False,
) -> EyeFusionResult:
    """Synthesize multi-modal sensory inputs into definitive driver eye state."""
    # 1. Resolve aggregate landmark EAR and blink coefficients
    if mean_ear is None:
        valid_ears = [e for e in (left_ear, right_ear) if e is not None and e > 0.0]
        eff_ear = float(sum(valid_ears) / len(valid_ears)) if valid_ears else None
    else:
        eff_ear = mean_ear

    if mean_blink is None:
        valid_blinks = [b for b in (blink_left, blink_right) if b is not None]
        eff_blink = float(sum(valid_blinks) / len(valid_blinks)) if valid_blinks else 0.0
    else:
        eff_blink = mean_blink

    has_landmarks = eff_ear is not None

    # Adapt close threshold if driver is pitching head down (nodding off)
    effective_close_th = ear_close_threshold
    if head_pitch > 16.0:  # Downward nodding droop
        effective_close_th = max(effective_close_th, 0.22)

    # 2. Extract CNN consensus
    cnn_states = [p.state for p in (left_cnn, right_cnn) if p is not None]
    cnn_confs = [p.confidence for p in (left_cnn, right_cnn) if p is not None]
    avg_cnn_conf = float(sum(cnn_confs) / len(cnn_confs)) if cnn_confs else 0.0

    cnn_all_open = len(cnn_states) > 0 and all(s == "OPEN" for s in cnn_states)
    cnn_all_closed = len(cnn_states) > 0 and all(s == "CLOSED" for s in cnn_states)

    # 3. Handle primary landmark-driven pathway
    if has_landmarks:
        ear_val = eff_ear  # type: ignore

        # Check for unilateral closure (one eye closed)
        if left_ear is not None and right_ear is not None:
            left_closed = left_ear < effective_close_th or (blink_left is not None and blink_left > blink_close_threshold)
            right_closed = right_ear < effective_close_th or (blink_right is not None and blink_right > blink_close_threshold)
            if left_closed != right_closed:
                # Monocular unilateral closure
                closed_side = "left" if left_closed else "right"
                return EyeFusionResult(
                    state="CLOSED",
                    confidence=0.78,
                    reason=f"Unilateral eyelid closure confirmed ({closed_side} eye closed).",
                    mean_ear=round(ear_val, 4),
                    mean_blink=round(eff_blink, 4),
                    landmark_driven=True,
                )

        # Primary rule: landmarks indicate CLOSED
        landmarks_say_closed = (ear_val < effective_close_th) or (eff_blink > blink_close_threshold)
        if prev_closed:
            # Hysteresis exit: requires EAR >= open_th AND blink <= 0.40
            landmarks_say_closed = not (ear_val >= ear_open_threshold and eff_blink <= 0.40)

        if landmarks_say_closed:
            # Rule: If landmarks say CLOSED, result is CLOSED
            # UNLESS CNN is confidently OPEN AND EAR is within 10% of threshold
            within_10_percent = ear_val >= (effective_close_th * 0.90)
            if cnn_all_open and avg_cnn_conf >= 0.85 and within_10_percent and not has_glasses:
                return EyeFusionResult(
                    state="OPEN",
                    confidence=round(avg_cnn_conf * 0.85, 3),
                    reason="Borderline EAR resolved to OPEN by high-confidence CNN consensus.",
                    mean_ear=round(ear_val, 4),
                    mean_blink=round(eff_blink, 4),
                    landmark_driven=True,
                )

            # Strong conflict check: if CNN says wide OPEN with high confidence but EAR is deeply closed
            if cnn_all_open and avg_cnn_conf >= 0.88 and ear_val < (effective_close_th * 0.70):
                # Strong conflict between deep closed EAR and confident CNN OPEN -> UNCERTAIN
                return EyeFusionResult(
                    state="UNCERTAIN",
                    confidence=0.50,
                    reason="Sensory conflict: landmark EAR shows deep closure but CNN predicts OPEN.",
                    mean_ear=round(ear_val, 4),
                    mean_blink=round(eff_blink, 4),
                    landmark_driven=True,
                )

            # Never allow single CNN OPEN to override clearly closed EAR
            conf = max(0.85, 1.0 - (ear_val / max(0.01, effective_close_th)) * 0.3)
            return EyeFusionResult(
                state="CLOSED",
                confidence=round(min(0.99, conf), 3),
                reason=f"Physical closure confirmed by landmarks (EAR {ear_val:.3f} < {effective_close_th:.3f}).",
                mean_ear=round(ear_val, 4),
                mean_blink=round(eff_blink, 4),
                landmark_driven=True,
            )

        # Primary rule: landmarks indicate OPEN
        if ear_val >= ear_open_threshold and eff_blink < blink_close_threshold:
            # Check for strong conflict: CNN firmly closed but landmark EAR clearly open
            if cnn_all_closed and avg_cnn_conf >= 0.85 and not has_glasses:
                # Strong conflict -> UNCERTAIN, do not default silently to OPEN
                return EyeFusionResult(
                    state="UNCERTAIN",
                    confidence=0.50,
                    reason="Sensory conflict: landmarks show open aperture but CNN votes CLOSED.",
                    mean_ear=round(ear_val, 4),
                    mean_blink=round(eff_blink, 4),
                    landmark_driven=True,
                )

            # Normal open consensus
            open_conf = 0.92 if (cnn_all_open or has_glasses) else 0.82
            return EyeFusionResult(
                state="OPEN",
                confidence=round(open_conf, 3),
                reason=f"Aperture verified open by landmarks (EAR {ear_val:.3f} >= {ear_open_threshold:.3f}).",
                mean_ear=round(ear_val, 4),
                mean_blink=round(eff_blink, 4),
                landmark_driven=True,
            )

        # Intermediate boundary band (effective_close_th <= ear_val < ear_open_threshold)
        if cnn_all_closed and avg_cnn_conf >= 0.70:
            return EyeFusionResult(
                state="CLOSED",
                confidence=round(avg_cnn_conf, 3),
                reason="Boundary EAR resolved to CLOSED by CNN vote.",
                mean_ear=round(ear_val, 4),
                mean_blink=round(eff_blink, 4),
                landmark_driven=True,
            )
        if cnn_all_open and avg_cnn_conf >= 0.70:
            return EyeFusionResult(
                state="OPEN",
                confidence=round(avg_cnn_conf, 3),
                reason="Boundary EAR resolved to OPEN by CNN vote.",
                mean_ear=round(ear_val, 4),
                mean_blink=round(eff_blink, 4),
                landmark_driven=True,
            )
        return EyeFusionResult(
            state="UNCERTAIN",
            confidence=0.50,
            reason="Aperture in boundary hysteresis band without definitive CNN consensus.",
            mean_ear=round(ear_val, 4),
            mean_blink=round(eff_blink, 4),
            landmark_driven=True,
        )

    # 4. Fallback pathway (landmarks unavailable)
    if not cnn_states:
        return EyeFusionResult(
            state="UNCERTAIN",
            confidence=0.0,
            reason="No sensory observations available (neither landmarks nor CNN crops).",
            mean_ear=0.28,
            mean_blink=0.0,
            landmark_driven=False,
        )

    if cnn_all_closed and avg_cnn_conf >= 0.65:
        return EyeFusionResult(
            state="CLOSED",
            confidence=round(avg_cnn_conf, 3),
            reason="Fallback CNN consensus: both eyes closed.",
            mean_ear=0.10,
            mean_blink=0.80,
            landmark_driven=False,
        )
    if cnn_all_open and avg_cnn_conf >= 0.65:
        return EyeFusionResult(
            state="OPEN",
            confidence=round(avg_cnn_conf, 3),
            reason="Fallback CNN consensus: both eyes open.",
            mean_ear=0.28,
            mean_blink=0.10,
            landmark_driven=False,
        )

    # Any asymmetric or ambiguous predictions below confidence threshold return UNCERTAIN
    return EyeFusionResult(
        state="UNCERTAIN",
        confidence=round(avg_cnn_conf, 3),
        reason="Fallback CNN predictions below confidence threshold or discordant.",
        mean_ear=0.20,
        mean_blink=0.50,
        landmark_driven=False,
    )
