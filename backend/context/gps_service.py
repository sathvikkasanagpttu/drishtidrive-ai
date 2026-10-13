"""
GPS & Speed Context Module
==========================
Parses vehicle telemetry from serial NMEA strings or REST API payloads:
1. Speed-scaled thresholds (tightened at highway speeds, relaxed when stopped)
2. Stationary Stage 3 siren gating (suppresses siren when vehicle speed <= 5 km/h)
3. Continuous driving trip timer and break reminder alerts
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger("drishtidrive.context.gps")


@dataclass
class TelemetryContext:
    """Current driving and GPS context state."""
    speed_kmh: float
    is_moving: bool
    siren_allowed: bool
    threshold_scale: float
    trip_duration_sec: float
    break_recommended: bool
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class GPSService:
    """
    Manages vehicle speed telemetry, adaptive speed scaling,
    siren suppression, and trip fatigue duration tracking.
    """

    MOVING_SPEED_CUTOFF = 5.0        # km/h
    HIGHWAY_SPEED_CUTOFF = 60.0      # km/h
    EXPRESS_SPEED_CUTOFF = 90.0      # km/h

    def __init__(
        self,
        break_interval_sec: float = 7200.0,  # 2 hours
        stationary_reset_sec: float = 900.0,  # 15 minutes
    ):
        self.break_interval_sec = break_interval_sec
        self.stationary_reset_sec = stationary_reset_sec

        self._speed_kmh: float = 0.0
        self._lat: Optional[float] = None
        self._lon: Optional[float] = None

        self._trip_start_time: Optional[float] = None
        self._last_moving_time: Optional[float] = None
        self._last_update_time: float = time.time()

    def update_telemetry(
        self,
        speed_kmh: float,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> TelemetryContext:
        """Update vehicle telemetry and refresh trip timer."""
        now = timestamp if timestamp is not None else time.time()
        self._speed_kmh = max(0.0, float(speed_kmh))
        self._last_update_time = now
        if latitude is not None and longitude is not None:
            self._lat = float(latitude)
            self._lon = float(longitude)

        is_moving = self._speed_kmh > self.MOVING_SPEED_CUTOFF

        if is_moving:
            if self._trip_start_time is None:
                self._trip_start_time = now
            self._last_moving_time = now
        else:
            # If vehicle has been stationary longer than reset window, reset trip
            if self._last_moving_time and (now - self._last_moving_time > self.stationary_reset_sec):
                self._trip_start_time = None
                self._last_moving_time = None

        return self.get_context(now)

    def parse_nmea(self, sentence: str) -> Optional[TelemetryContext]:
        """
        Parse standard NMEA sentence ($GPRMC or $GNRMC) for speed and coordinates.
        Speed is reported in knots in field 7.
        """
        try:
            parts = sentence.strip().split(",")
            if len(parts) < 8 or parts[0] not in ("$GPRMC", "$GNRMC"):
                return None

            # Valid fix status ('A' = active, 'V' = void)
            if parts[2] != "A":
                return None

            # Knots to km/h conversion (1 knot = 1.852 km/h)
            knots = float(parts[7]) if parts[7] else 0.0
            speed_kmh = knots * 1.852

            # Latitude & Longitude decoding if present
            lat, lon = self._decode_nmea_coords(parts[3], parts[4], parts[5], parts[6])
            return self.update_telemetry(speed_kmh=speed_kmh, latitude=lat, longitude=lon)
        except Exception as exc:
            logger.debug("Failed parsing NMEA sentence: %s", exc)
            return None

    def get_threshold_scaling(self, speed_kmh: float) -> float:
        """
        Compute multiplier for alert thresholds based on vehicle velocity.
        - Parked / Idle (<5 km/h): 1.25x (relaxed)
        - Urban (5-60 km/h): 1.00x (nominal)
        - Highway (60-90 km/h): 0.80x (tightened)
        - Expressway (>90 km/h): 0.70x (aggressive)
        """
        if speed_kmh <= self.MOVING_SPEED_CUTOFF:
            return 1.25
        elif speed_kmh <= self.HIGHWAY_SPEED_CUTOFF:
            return 1.00
        elif speed_kmh <= self.EXPRESS_SPEED_CUTOFF:
            return 0.80
        return 0.70

    def get_context(self, current_time: Optional[float] = None) -> TelemetryContext:
        """Return instantaneous context snapshot."""
        now = current_time if current_time is not None else time.time()
        is_moving = self._speed_kmh > self.MOVING_SPEED_CUTOFF

        trip_dur = 0.0
        if self._trip_start_time is not None:
            trip_dur = max(0.0, now - self._trip_start_time)

        break_rec = trip_dur >= self.break_interval_sec
        scale = self.get_threshold_scaling(self._speed_kmh)

        return TelemetryContext(
            speed_kmh=round(self._speed_kmh, 1),
            is_moving=is_moving,
            siren_allowed=is_moving,  # Only play Stage 3 siren while moving
            threshold_scale=scale,
            trip_duration_sec=round(trip_dur, 1),
            break_recommended=break_rec,
            latitude=self._lat,
            longitude=self._lon,
        )

    def reset_trip(self) -> None:
        """Explicitly reset trip continuous driving counter."""
        self._trip_start_time = None
        self._last_moving_time = None

    @staticmethod
    def _decode_nmea_coords(
        lat_str: str, lat_dir: str, lon_str: str, lon_dir: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Convert NMEA ddmm.mmmm format to decimal degrees."""
        try:
            if not lat_str or not lon_str:
                return None, None
            lat_deg = float(lat_str[:2]) + float(lat_str[2:]) / 60.0
            if lat_dir == "S":
                lat_deg = -lat_deg
            lon_deg = float(lon_str[:3]) + float(lon_str[3:]) / 60.0
            if lon_dir == "W":
                lon_deg = -lon_deg
            return round(lat_deg, 6), round(lon_deg, 6)
        except Exception:
            return None, None
