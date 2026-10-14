"""In-memory mock audio backend for DrishtiDrive AI testing and headless verification.

Provides deterministic recording of tones, speech messages, and stop calls
without requiring audio hardware.
"""
from __future__ import annotations

import time
from typing import List, Tuple

from backend.audio_backend import AudioBackendStatus, BaseAudioBackend
from backend.temporal_state import AlertStage


class MockAudioBackend(BaseAudioBackend):
    """Deterministic in-memory audio mock for unit and integration testing."""

    def __init__(self, simulate_device_failure: bool = False) -> None:
        """Initialize mock backend with telemetry tracking."""
        self.simulate_device_failure = simulate_device_failure
        self.status = AudioBackendStatus(
            device_available=not simulate_device_failure,
            engine_type="mock",
        )
        self.played_tones: List[Tuple[AlertStage, float]] = []
        self.spoken_messages: List[Tuple[str, AlertStage, str]] = []
        self.stop_calls: int = 0

    def reset(self) -> None:
        """Clear recorded tones, messages, and calls."""
        self.played_tones.clear()
        self.spoken_messages.clear()
        self.stop_calls = 0

    def play_tone(self, stage: AlertStage, volume: float) -> bool:
        """Record played tone in mock history."""
        if not self.status.device_available:
            return False
        self.played_tones.append((stage, volume))
        self.status.last_played_stage = stage
        self.status.last_played_time = time.time()
        return True

    def speak(
        self,
        text: str,
        stage: AlertStage,
        language: str = "en",
        volume: float = 0.8,
    ) -> bool:
        """Record spoken message in mock history."""
        if not self.status.device_available:
            return False
        self.spoken_messages.append((text, stage, language))
        self.status.last_played_stage = stage
        self.status.last_played_time = time.time()
        return True

    def stop(self) -> None:
        """Record stop call."""
        self.stop_calls += 1
        self.status.is_playing = False

    def get_status(self) -> AudioBackendStatus:
        """Return mock audio status."""
        return self.status
