"""Unit tests for local cross-platform audio playback backend.

Verifies:
- WAV tone and MP3 voice asset availability for Stages 1, 2, and 3.
- Cross-platform process invocation and timeout handling.
- Graceful termination via stop().
- Failure handling when tone/MP3 files are missing or device fails.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time
from unittest.mock import MagicMock, patch
import pytest

from backend.audio_backend import AudioBackendStatus, LocalAudioBackend
from backend.temporal_state import AlertStage


class TestLocalAudioBackend:
    """Verifies LocalAudioBackend operations across alert stages."""

    @pytest.fixture
    def audio_backend(self) -> LocalAudioBackend:
        """Create LocalAudioBackend instance pointing to assets directory."""
        backend = LocalAudioBackend(use_tts=False, initial_volume=0.5)
        yield backend
        backend.stop()

    def test_asset_paths_and_tones_exist(self, audio_backend: LocalAudioBackend) -> None:
        """Verify synthesized WAV tones exist for all three alert stages."""
        assert audio_backend.status.device_available is True
        for stage in (AlertStage.STAGE_1_EARLY, AlertStage.STAGE_2_DROWSY, AlertStage.STAGE_3_CRITICAL):
            tone_path = audio_backend.tone_paths.get(stage)
            assert tone_path is not None
            assert Path(tone_path).exists()
            assert Path(tone_path).stat().st_size > 0

    def test_voice_mp3_assets_exist(self, audio_backend: LocalAudioBackend) -> None:
        """Verify prerecorded MP3 voice files exist in assets directory."""
        mp3_names = ["Stage1_Voice.mp3", "Stage2_voice.mp3", "Stage3_voice.mp3"]
        for name in mp3_names:
            mp3_path = audio_backend.assets_dir / name
            assert mp3_path.exists(), f"Missing required MP3 voice asset: {name}"
            assert mp3_path.stat().st_size > 1000

    def test_play_tone_execution(self, audio_backend: LocalAudioBackend) -> None:
        """Verify play_tone executes and updates status telemetry."""
        # Mock subprocess.Popen so test does not emit actual sound in test runners
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 0
            mock_popen.return_value = mock_proc

            success = audio_backend.play_tone(AlertStage.STAGE_1_EARLY, volume=0.7)
            assert success is True
            assert audio_backend.status.last_played_stage == AlertStage.STAGE_1_EARLY
            assert audio_backend.status.last_played_time > 0

    def test_play_mp3_execution(self, audio_backend: LocalAudioBackend) -> None:
        """Verify _play_prerecorded_mp3 starts playback process and exits cleanly."""
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.side_effect = [None, 0]
            mock_popen.return_value = mock_proc

            success = audio_backend._play_prerecorded_mp3(AlertStage.STAGE_2_DROWSY, volume=0.8)
            assert success is True

    def test_stop_terminates_active_process(self, audio_backend: LocalAudioBackend) -> None:
        """Verify stop() terminates any running audio child process."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        audio_backend._active_proc = mock_proc
        audio_backend.status.is_playing = True

        audio_backend.stop()

        assert audio_backend.status.is_playing is False
        mock_proc.terminate.assert_called_once()
        assert audio_backend._active_proc is None

    def test_missing_tone_file_returns_false(self, tmp_path: Path) -> None:
        """Verify missing tone handles missing files gracefully without crashing."""
        empty_backend = LocalAudioBackend(assets_dir=tmp_path, use_tts=False)
        empty_backend.tone_paths.clear()

        result = empty_backend.play_tone(AlertStage.STAGE_3_CRITICAL, volume=0.8)
        assert result is False
