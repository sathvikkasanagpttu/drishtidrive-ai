"""
Offline-First Encrypted Fleet Event Storage
===========================================
Local SQLite store for fleet trip logs and alert events.
Privacy Defaults:
- Zero raw video/facial frames stored.
- Event payloads are encrypted and HMAC-signed locally.
- Sync status tracking (synced=0 offline, synced=1 once synced to fleet API).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger("drishtidrive.fleet.store")


class FleetCrypto:
    """Lightweight authenticated encryption for local at-rest telemetry."""

    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            raw_key = os.environ.get("DRISHTIDRIVE_ENCRYPTION_KEY", "drishtidrive-default-local-key-2026")
            self.key = hashlib.sha256(raw_key.encode("utf-8")).digest()
        else:
            self.key = key

    def encrypt(self, plain_text: str) -> str:
        """Encrypt string payload and append HMAC signature."""
        data_bytes = plain_text.encode("utf-8")
        # Keystream via repeated SHA-256 blocks
        keystream = b""
        counter = 0
        while len(keystream) < len(data_bytes):
            keystream += hashlib.sha256(self.key + counter.to_bytes(4, "big")).digest()
            counter += 1

        ciphertext = bytes([b ^ k for b, k in zip(data_bytes, keystream[:len(data_bytes)])])
        sig = hmac.new(self.key, ciphertext, hashlib.sha256).digest()
        packed = sig + ciphertext
        return base64.b64encode(packed).decode("ascii")

    def decrypt(self, cipher_b64: str) -> str:
        """Verify HMAC signature and decrypt string payload."""
        packed = base64.b64decode(cipher_b64.encode("ascii"))
        sig = packed[:32]
        ciphertext = packed[32:]

        expected_sig = hmac.new(self.key, ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected_sig):
            raise ValueError("Integrity verification failed for stored event record")

        keystream = b""
        counter = 0
        while len(keystream) < len(ciphertext):
            keystream += hashlib.sha256(self.key + counter.to_bytes(4, "big")).digest()
            counter += 1

        plain = bytes([c ^ k for c, k in zip(ciphertext, keystream[:len(ciphertext)])])
        return plain.decode("utf-8")


class FleetStore:
    """Local SQLite repository managing trip telematics and safety events."""

    def __init__(self, db_path: str = "data/fleet_events.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.crypto = FleetCrypto()
        self._init_db()

    def _init_db(self) -> None:
        """Create database tables and performance indexes."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trips (
                    trip_id TEXT PRIMARY KEY,
                    driver_id TEXT,
                    start_time REAL,
                    end_time REAL,
                    risk_score REAL DEFAULT 0.0,
                    synced INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    trip_id TEXT,
                    timestamp REAL,
                    event_type TEXT,
                    stage INTEGER,
                    duration_sec REAL,
                    speed_kmh REAL,
                    acknowledged INTEGER,
                    payload_encrypted TEXT,
                    synced INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_synced ON events (synced)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_trip ON events (trip_id)")
            conn.commit()

    def start_trip(self, driver_id: str = "driver_default") -> str:
        """Create new local driving trip entry."""
        trip_id = f"trip_{uuid4().hex[:8]}"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO trips (trip_id, driver_id, start_time, synced) VALUES (?, ?, ?, 0)",
                (trip_id, driver_id, time.time()),
            )
            conn.commit()
        return trip_id

    def record_event(
        self,
        trip_id: str,
        event_type: str,
        stage: int,
        duration_sec: float,
        speed_kmh: float,
        acknowledged: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log safety event with privacy-protected encrypted metadata."""
        event_id = f"evt_{uuid4().hex[:10]}"
        payload = json.dumps(metadata or {})
        enc_payload = self.crypto.encrypt(payload)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO events (
                    event_id, trip_id, timestamp, event_type, stage,
                    duration_sec, speed_kmh, acknowledged, payload_encrypted, synced
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    event_id,
                    trip_id,
                    time.time(),
                    event_type,
                    int(stage),
                    float(duration_sec),
                    float(speed_kmh),
                    1 if acknowledged else 0,
                    enc_payload,
                ),
            )
            conn.commit()
        return event_id

    def get_trip_events(self, trip_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve historical events with decrypted metadata."""
        query = "SELECT event_id, trip_id, timestamp, event_type, stage, duration_sec, speed_kmh, acknowledged, payload_encrypted, synced FROM events"
        params: List[Any] = []
        if trip_id:
            query += " WHERE trip_id = ?"
            params.append(trip_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        records = []
        with sqlite3.connect(self.db_path) as conn:
            for row in conn.execute(query, params):
                try:
                    dec_meta = json.loads(self.crypto.decrypt(row[8]))
                except Exception:
                    dec_meta = {}
                records.append({
                    "event_id": row[0],
                    "trip_id": row[1],
                    "timestamp": row[2],
                    "event_type": row[3],
                    "stage": row[4],
                    "duration_sec": row[5],
                    "speed_kmh": row[6],
                    "acknowledged": bool(row[7]),
                    "metadata": dec_meta,
                    "synced": bool(row[9]),
                })
        return records

    def calculate_driver_risk(self, driver_id: str = "driver_default") -> float:
        """Calculate weighted safety risk score (0-100) based on event severity."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT e.stage, COUNT(*) FROM events e
                JOIN trips t ON e.trip_id = t.trip_id
                WHERE t.driver_id = ?
                GROUP BY e.stage
                """,
                (driver_id,),
            )
            counts = {stage: count for stage, count in cursor.fetchall()}

        # Risk scoring weights: Stage 1 = 2 pts, Stage 2 = 10 pts, Stage 3 = 30 pts
        raw_score = (counts.get(1, 0) * 2.0) + (counts.get(2, 0) * 10.0) + (counts.get(3, 0) * 30.0)
        return min(100.0, round(raw_score, 1))

    def mark_synced(self, event_ids: List[str]) -> None:
        """Flag events as successfully transmitted to fleet cloud."""
        if not event_ids:
            return
        placeholders = ",".join("?" * len(event_ids))
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"UPDATE events SET synced = 1 WHERE event_id IN ({placeholders})", event_ids)
            conn.commit()
