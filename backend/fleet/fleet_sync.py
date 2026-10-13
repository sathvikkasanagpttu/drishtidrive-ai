"""
Fleet Telemetry Sync Adapter
============================
Transmits offline encrypted event records to remote fleet API over HTTP or MQTT.
Retains records locally if connectivity is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import urllib.request
import json

from backend.fleet.fleet_store import FleetStore

logger = logging.getLogger("drishtidrive.fleet.sync")


class FleetSyncAdapter:
    """
    Manages queued event synchronization to cloud fleet management servers.
    """

    def __init__(
        self,
        store: FleetStore,
        endpoint_url: Optional[str] = None,
        timeout_sec: float = 3.0,
    ):
        self.store = store
        self.endpoint_url = endpoint_url
        self.timeout_sec = timeout_sec

    def sync_pending(self, batch_size: int = 50) -> Dict[str, Any]:
        """
        Attempt to synchronize un-synced events.

        Returns:
            Dictionary containing synced_count, failed_count, and network_status.
        """
        events = self._get_unsynced_events(limit=batch_size)
        if not events:
            return {"synced_count": 0, "failed_count": 0, "status": "UP_TO_DATE"}

        if not self.endpoint_url:
            # Local-only mode or simulated fleet cloud
            event_ids = [e["event_id"] for e in events]
            self.store.mark_synced(event_ids)
            logger.info("Local/simulated sync completed for %d events", len(event_ids))
            return {
                "synced_count": len(event_ids),
                "failed_count": 0,
                "status": "SYNCED_LOCAL",
            }

        try:
            payload = json.dumps({"events": events}).encode("utf-8")
            req = urllib.request.Request(
                self.endpoint_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                if 200 <= resp.status < 300:
                    event_ids = [e["event_id"] for e in events]
                    self.store.mark_synced(event_ids)
                    return {
                        "synced_count": len(event_ids),
                        "failed_count": 0,
                        "status": "SUCCESS",
                    }
                else:
                    return {
                        "synced_count": 0,
                        "failed_count": len(events),
                        "status": f"HTTP_ERROR_{resp.status}",
                    }

        except Exception as exc:
            logger.debug("Fleet sync failed (offline or network error): %s", exc)
            return {
                "synced_count": 0,
                "failed_count": len(events),
                "status": "OFFLINE",
            }

    def _get_unsynced_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch pending records with synced = 0."""
        import sqlite3
        events = []
        with sqlite3.connect(self.store.db_path) as conn:
            query = """
                SELECT event_id, trip_id, timestamp, event_type, stage,
                       duration_sec, speed_kmh, acknowledged, payload_encrypted
                FROM events WHERE synced = 0 ORDER BY timestamp ASC LIMIT ?
            """
            for row in conn.execute(query, (limit,)):
                events.append({
                    "event_id": row[0],
                    "trip_id": row[1],
                    "timestamp": row[2],
                    "event_type": row[3],
                    "stage": row[4],
                    "duration_sec": row[5],
                    "speed_kmh": row[6],
                    "acknowledged": bool(row[7]),
                    "payload_encrypted": row[8],
                })
        return events
