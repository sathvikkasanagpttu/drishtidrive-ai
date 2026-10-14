"""Per-driver baseline calibration service.

Runs a guided 10-second calibration window ("keep eyes open normally", then
"close your eyes for 3 seconds") to measure driver-specific physiological baselines:
- Open eye EAR baseline & closed eye EAR baseline
- Adaptive closure & hysteresis thresholds
- Natural spontaneous blink rate (BPM)
- Neutral head pose orientation (pitch/yaw/roll)

Persists profile strictly locally in driver_profile.json (zero cloud transmission).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime
from enum import Enum
import json
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


class CalibrationState(str, Enum):
    """Lifecycle status of the driver calibration process."""

    IDLE = "IDLE"
    CALIBRATING = "CALIBRATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class DriverProfile:
    """Calibrated individual driver baseline parameters."""

    driver_id: str = "default_driver"
    baseline_open_conf: float = 0.90
    open_ear_baseline: float = 0.28
    closed_ear_baseline: float = 0.14
    ear_close_threshold: float = 0.196  # open_ear_baseline * 0.70
    ear_open_threshold: float = 0.224   # open_ear_baseline * 0.80 (hysteresis)
    blink_close_threshold: float = 0.55
    blink_open_threshold: float = 0.40
    baseline_blink_rate: float = 18.0
    neutral_pitch: float = 0.0
    neutral_yaw: float = 0.0
    neutral_roll: float = 0.0
    adapted_stage1_sec: float = 0.8
    adapted_stage2_sec: float = 1.8
    adapted_stage3_sec: float = 3.0
    sample_count: int = 0
    calibrated_at: str = ""


class DriverCalibrationService:
    """Manages guided driver calibration sessions and dynamic threshold adaptation."""

    def __init__(
        self,
        profile_path: Path = Path("driver_profile.json"),
        calibration_duration_sec: float = 10.0,
    ) -> None:
        """Initialize calibration manager and load existing local profile if present."""
        self.profile_path = profile_path
        self.calibration_duration_sec = calibration_duration_sec
        self.state = CalibrationState.IDLE
        self.start_timestamp: Optional[float] = None
        self.samples_conf: List[float] = []
        self.samples_pitch: List[float] = []
        self.samples_yaw: List[float] = []
        self.samples_roll: List[float] = []
        self.open_ear_samples: List[float] = []
        self.closed_ear_samples: List[float] = []
        self.open_blink_samples: List[float] = []
        self.closed_blink_samples: List[float] = []
        self.blink_count: int = 0
        self.last_measured_ear: float = 0.28
        self.last_measured_blink: float = 0.0
        self._current_driver_id = "default_driver"
        self.profile: Optional[DriverProfile] = self._load_profile()

    def start_calibration(self, driver_id: str = "default_driver") -> None:
        """Initiate a fresh guided 10-second calibration session."""
        self.state = CalibrationState.CALIBRATING
        self.start_timestamp = None
        self.samples_conf.clear()
        self.samples_pitch.clear()
        self.samples_yaw.clear()
        self.samples_roll.clear()
        self.open_ear_samples.clear()
        self.closed_ear_samples.clear()
        self.open_blink_samples.clear()
        self.closed_blink_samples.clear()
        self.blink_count = 0
        self._current_driver_id = driver_id

    def get_phase(self, current_timestamp: float) -> Tuple[str, str]:
        """Return active guided phase ('OPEN_PHASE' or 'CLOSED_PHASE') and prompt instruction."""
        if self.state != CalibrationState.CALIBRATING:
            if self.state == CalibrationState.COMPLETED:
                return "COMPLETED", "Calibration complete! Custom thresholds active."
            return "IDLE", "Calibration idle."

        if self.start_timestamp is None:
            open_sec = max(1, int(self.calibration_duration_sec * 0.70))
            return "OPEN_PHASE", f"Keep eyes OPEN normally looking ahead ({open_sec}s)"

        elapsed = round(max(0.0, current_timestamp - self.start_timestamp), 4)
        open_phase_sec = round(self.calibration_duration_sec * 0.70, 4)
        if elapsed < open_phase_sec:
            rem = max(0, int(open_phase_sec - elapsed) + 1)
            return "OPEN_PHASE", f"Keep eyes OPEN normally looking ahead ({rem}s)"
        rem = max(0, int(self.calibration_duration_sec - elapsed) + 1)
        return "CLOSED_PHASE", f"Now CLOSE your eyes gently ({rem}s)"

    def update(
        self,
        eye_state: str,
        confidence: float,
        pitch: float,
        yaw: float,
        roll: float,
        is_blink: bool,
        timestamp: float,
        mean_ear: Optional[float] = None,
        mean_blink: Optional[float] = None,
    ) -> CalibrationState:
        """Ingest frame metrics during active calibration window."""
        if mean_ear is not None:
            self.last_measured_ear = mean_ear
        if mean_blink is not None:
            self.last_measured_blink = mean_blink

        if self.state != CalibrationState.CALIBRATING:
            return self.state

        if self.start_timestamp is None:
            self.start_timestamp = timestamp

        elapsed = round(max(0.0, timestamp - self.start_timestamp), 4)
        phase, _ = self.get_phase(timestamp)

        # Ingest metrics according to guided phase
        if phase == "OPEN_PHASE":
            self.samples_conf.append(confidence)
            self.samples_pitch.append(pitch)
            self.samples_yaw.append(yaw)
            self.samples_roll.append(roll)
            if mean_ear is not None and mean_ear > 0.05:
                self.open_ear_samples.append(mean_ear)
            if mean_blink is not None:
                self.open_blink_samples.append(mean_blink)
        elif phase == "CLOSED_PHASE":
            if mean_ear is not None and mean_ear > 0.01:
                self.closed_ear_samples.append(mean_ear)
            if mean_blink is not None:
                self.closed_blink_samples.append(mean_blink)

        if is_blink:
            self.blink_count += 1

        if elapsed >= self.calibration_duration_sec:
            self._finalize_calibration()

        return self.state

    def is_eye_closed_calibrated(
        self,
        mean_ear: float,
        mean_blink: float = 0.0,
        prev_closed: bool = False,
    ) -> bool:
        """Evaluate eye closure state against calibrated profile using hysteresis."""
        if self.profile is None:
            # Fallback uncalibrated thresholds
            close_thresh = 0.18
            open_thresh = 0.22
            blink_close = 0.55
            blink_open = 0.40
        else:
            close_thresh = self.profile.ear_close_threshold
            open_thresh = self.profile.ear_open_threshold
            blink_close = self.profile.blink_close_threshold
            blink_open = self.profile.blink_open_threshold

        if prev_closed:
            # Hysteresis exit: both EAR must rise above open_thresh AND blink below blink_open
            return not (mean_ear >= open_thresh and mean_blink <= blink_open)
        # Hysteresis enter: EAR drops below close_thresh OR blink rises above blink_close
        return (mean_ear < close_thresh) or (mean_blink > blink_close)

    def get_progress(self, current_timestamp: float) -> float:
        """Return calibration progress percentage (0.0 to 100.0)."""
        if self.state != CalibrationState.CALIBRATING or self.start_timestamp is None:
            return 100.0 if self.state == CalibrationState.COMPLETED else 0.0
        elapsed = max(0.0, current_timestamp - self.start_timestamp)
        return min(100.0, round((elapsed / self.calibration_duration_sec) * 100.0, 1))

    def _finalize_calibration(self) -> None:
        """Compute baseline parameters and save profile."""
        min_samples = max(5, int(self.calibration_duration_sec * 3))
        if len(self.samples_conf) < min_samples:
            self.state = CalibrationState.FAILED
            return

        open_ear = float(np.mean(self.open_ear_samples)) if self.open_ear_samples else 0.28
        closed_ear = float(np.mean(self.closed_ear_samples)) if self.closed_ear_samples else (open_ear * 0.50)

        # Ensure sensible physical range
        open_ear = max(0.20, min(0.42, open_ear))
        closed_ear = max(0.06, min(open_ear * 0.75, closed_ear))

        # Close threshold at 70% of open baseline; open threshold with +10-15% hysteresis band
        close_ear_th = round(open_ear * 0.70, 3)
        open_ear_th = round(open_ear * 0.80, 3)

        duration_min = self.calibration_duration_sec / 60.0
        blink_rate = self.blink_count / duration_min if duration_min > 0 else 18.0

        self.profile = DriverProfile(
            driver_id=self._current_driver_id,
            baseline_open_conf=round(float(np.mean(self.samples_conf)), 3),
            open_ear_baseline=round(open_ear, 3),
            closed_ear_baseline=round(closed_ear, 3),
            ear_close_threshold=close_ear_th,
            ear_open_threshold=open_ear_th,
            blink_close_threshold=0.55,
            blink_open_threshold=0.40,
            baseline_blink_rate=round(blink_rate, 1),
            neutral_pitch=round(float(np.mean(self.samples_pitch)), 1),
            neutral_yaw=round(float(np.mean(self.samples_yaw)), 1),
            neutral_roll=round(float(np.mean(self.samples_roll)), 1),
            adapted_stage1_sec=0.8,
            adapted_stage2_sec=1.8,
            adapted_stage3_sec=3.0,
            sample_count=len(self.samples_conf) + len(self.closed_ear_samples),
            calibrated_at=datetime.datetime.now().isoformat(),
        )
        self._save_profile()
        self.state = CalibrationState.COMPLETED

    def _save_profile(self) -> None:
        """Store calibrated profile locally on filesystem."""
        if self.profile is not None:
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.profile), f, indent=2)

    def _load_profile(self) -> Optional[DriverProfile]:
        """Load profile from local filesystem if present."""
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return DriverProfile(**data)
            except Exception:
                return None
        return None

    def reset(self) -> None:
        """Clear calibration and delete local profile file."""
        self.state = CalibrationState.IDLE
        self.profile = None
        if self.profile_path.exists():
            self.profile_path.unlink(missing_ok=True)
