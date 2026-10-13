"""
Phase 4 Tests: Fleet, Speed Context, Response Check, and Multilingual Catalog
=============================================================================
Tests GPS speed scaling, stationary siren gating, Stage 2 response challenge,
encrypted offline SQLite event logging, and Indian language alert localizations.
"""

import time
import pytest

from backend.context.gps_service import GPSService
from backend.context.response_check import DriverResponseChecker, ResponseStatus
from backend.fleet.fleet_store import FleetStore, FleetCrypto
from backend.fleet.fleet_sync import FleetSyncAdapter
from backend.alert_catalog import AlertMessageCatalog, PIPER_VOICES
from backend.temporal_state import AlertStage
from backend.alert_manager import AlertManager
from backend.audio_backend import MockAudioBackend


def test_gps_service_speed_scaling_and_siren_gating():
    """Verify velocity-adaptive scaling and stationary siren suppression."""
    gps = GPSService(break_interval_sec=100.0)

    # 1. Parked / stationary (< 5 km/h)
    ctx_idle = gps.update_telemetry(speed_kmh=0.0)
    assert ctx_idle.is_moving is False
    assert ctx_idle.siren_allowed is False
    assert ctx_idle.threshold_scale == 1.25

    # 2. City traffic (30 km/h)
    ctx_city = gps.update_telemetry(speed_kmh=30.0)
    assert ctx_city.is_moving is True
    assert ctx_city.siren_allowed is True
    assert ctx_city.threshold_scale == 1.00

    # 3. Highway driving (75 km/h) -> tightened thresholds
    ctx_hwy = gps.update_telemetry(speed_kmh=75.0)
    assert ctx_hwy.threshold_scale == 0.80

    # 4. Expressway driving (105 km/h) -> aggressive scaling
    ctx_exp = gps.update_telemetry(speed_kmh=105.0)
    assert ctx_exp.threshold_scale == 0.70


def test_gps_service_nmea_sentence_parsing():
    """Verify parsing of standard $GPRMC sentences."""
    gps = GPSService()
    # 48.0 knots = 88.896 km/h, Active fix
    nmea_sample = "$GPRMC,123519,A,4807.038,N,01131.000,E,48.0,084.4,230394,003.1,W*6A"
    ctx = gps.parse_nmea(nmea_sample)

    assert ctx is not None
    assert round(ctx.speed_kmh, 1) == 88.9
    assert ctx.is_moving is True
    assert ctx.siren_allowed is True
    assert ctx.latitude is not None and ctx.latitude > 0


def test_driver_response_challenge_and_escalation():
    """Verify Stage 2 attentiveness challenge and escalation upon timeout."""
    checker = DriverResponseChecker(timeout_sec=2.0)
    t0 = 1000.0

    # Entry into Stage 2 triggers challenge
    stage = checker.evaluate(AlertStage.STAGE_2_DROWSY, timestamp=t0)
    assert stage == AlertStage.STAGE_2_DROWSY
    assert checker.status == ResponseStatus.WAITING_ACKNOWLEDGEMENT
    assert round(checker.get_remaining_time(timestamp=t0 + 0.5), 1) == 1.5

    # If within 2.0s, remains in Stage 2
    stage = checker.evaluate(AlertStage.STAGE_2_DROWSY, timestamp=t0 + 1.0)
    assert stage == AlertStage.STAGE_2_DROWSY

    # At 2.1s without acknowledgement -> Escalates to Stage 3!
    escalated = checker.evaluate(AlertStage.STAGE_2_DROWSY, timestamp=t0 + 2.1)
    assert escalated == AlertStage.STAGE_3_CRITICAL
    assert checker.status == ResponseStatus.EXPIRED_ESCALATED


def test_driver_response_successful_acknowledgement():
    """Verify manual button or UI acknowledgement resolves the challenge."""
    checker = DriverResponseChecker(timeout_sec=3.0)
    t0 = 2000.0

    checker.evaluate(AlertStage.STAGE_2_DROWSY, timestamp=t0)
    assert checker.status == ResponseStatus.WAITING_ACKNOWLEDGEMENT

    # Driver taps acknowledge button at t0 + 1.0s
    acked = checker.acknowledge(source="steering_button", timestamp=t0 + 1.0)
    assert acked is True
    assert checker.status == ResponseStatus.ACKNOWLEDGED

    # Subsequent evaluations do not escalate
    stage = checker.evaluate(AlertStage.STAGE_2_DROWSY, timestamp=t0 + 4.0)
    assert stage == AlertStage.STAGE_2_DROWSY


def test_multilingual_alert_catalog_coverage():
    """Verify comprehensive multilingual catalog coverage across all 7 Indian languages."""
    expected_langs = ["en", "hi", "mr", "te", "ta", "kn", "bn"]
    for lang in expected_langs:
        assert AlertMessageCatalog.is_language_supported(lang) is True
        assert lang in PIPER_VOICES

        for stage in (AlertStage.STAGE_0_NORMAL, AlertStage.STAGE_1_EARLY, AlertStage.STAGE_2_DROWSY, AlertStage.STAGE_3_CRITICAL):
            msg = AlertMessageCatalog.get_message(stage, language=lang)
            title = AlertMessageCatalog.get_title(stage, language=lang)
            assert len(msg.strip()) > 0
            assert len(title.strip()) > 0


def test_encrypted_fleet_store_and_sync(tmp_path):
    """Verify local encrypted storage and offline sync adapter."""
    db_file = str(tmp_path / "fleet_test.db")
    store = FleetStore(db_path=db_file)

    trip_id = store.start_trip(driver_id="driver_KA01")
    evt_id = store.record_event(
        trip_id=trip_id,
        event_type="DROWSINESS_STAGE_2",
        stage=2,
        duration_sec=2.4,
        speed_kmh=64.0,
        acknowledged=True,
        metadata={"ear": 0.18, "perclos": 0.32},
    )

    events = store.get_trip_events(trip_id=trip_id)
    assert len(events) == 1
    assert events[0]["event_id"] == evt_id
    assert events[0]["acknowledged"] is True
    assert events[0]["metadata"]["ear"] == 0.18

    # Verify risk calculation
    risk = store.calculate_driver_risk("driver_KA01")
    assert risk == 10.0  # Stage 2 event = 10 pts

    # Verify offline sync adapter in local mode
    sync_adapter = FleetSyncAdapter(store=store)
    sync_res = sync_adapter.sync_pending()
    assert sync_res["synced_count"] == 1
    assert sync_res["status"] == "SYNCED_LOCAL"


def test_stationary_siren_suppression_in_alert_manager():
    """Verify that Stage 3 siren tone is suppressed when vehicle is stationary."""
    mock_audio = MockAudioBackend()
    gps = GPSService()
    alert_mgr = AlertManager(audio_backend=mock_audio, gps_service=gps)

    # 1. Stationary vehicle (< 5 km/h) -> Siren suppressed
    gps.update_telemetry(speed_kmh=0.0)
    alert_mgr.trigger_alert(AlertStage.STAGE_3_CRITICAL, reason="Critical closure")
    time.sleep(0.15)  # Allow background worker to process queue

    # With stationary vehicle, tone played was downgraded to STAGE_2_DROWSY
    assert len(mock_audio.played_tones) > 0
    assert mock_audio.played_tones[-1][0] == AlertStage.STAGE_2_DROWSY

    # 2. Moving vehicle (60 km/h) -> Full siren allowed
    gps.update_telemetry(speed_kmh=60.0)
    # Clear cooldown for test
    alert_mgr._last_stage_time[AlertStage.STAGE_3_CRITICAL] = -999.0
    alert_mgr.trigger_alert(AlertStage.STAGE_3_CRITICAL, reason="Highway closure")
    time.sleep(0.15)

    assert mock_audio.played_tones[-1][0] == AlertStage.STAGE_3_CRITICAL
    alert_mgr.stop()
