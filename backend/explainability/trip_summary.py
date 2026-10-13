"""
Trip Safety Summary & Recommendation Engine
============================================
Compiles comprehensive post-trip driver safety reports:
- Aggregates alert event frequencies and peak closure durations
- Computes weighted fatigue risk score (0-100)
- Generates rule-based offline safety recommendations (kept strictly
  out of the real-time alerting loop)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("drishtidrive.explainability.trip_summary")


class TripSummaryEngine:
    """
    Generates actionable safety insights and recommendations from trip logs.
    """

    @classmethod
    def compile_report(
        cls,
        trip_id: str,
        driver_id: str,
        events: List[Dict[str, Any]],
        trip_duration_sec: float = 3600.0,
        average_speed_kmh: float = 55.0,
    ) -> Dict[str, Any]:
        """
        Compile structured trip report.

        Args:
            trip_id: Unique trip identifier.
            driver_id: Driver account / vehicle ID.
            events: Historical event records from FleetStore.
            trip_duration_sec: Total continuous driving duration in seconds.
            average_speed_kmh: Mean vehicle speed.
        """
        stage_counts = {1: 0, 2: 0, 3: 0}
        longest_closure = 0.0
        distraction_count = 0
        acknowledged_count = 0

        for evt in events:
            stg = evt.get("stage", 0)
            if stg in stage_counts:
                stage_counts[stg] += 1
            dur = evt.get("duration_sec", 0.0)
            if dur > longest_closure:
                longest_closure = dur
            if "DISTRACTION" in evt.get("event_type", ""):
                distraction_count += 1
            if evt.get("acknowledged", False):
                acknowledged_count += 1

        total_alerts = sum(stage_counts.values())

        # Calculate safety risk score (0-100)
        risk_score = min(
            100.0,
            round((stage_counts[1] * 2.0) + (stage_counts[2] * 8.0) + (stage_counts[3] * 25.0), 1),
        )

        risk_category = "EXCELLENT" if risk_score < 15 else (
            "MODERATE" if risk_score < 45 else "HIGH_RISK"
        )

        recommendations = cls._generate_recommendations(
            stage_counts=stage_counts,
            longest_closure=longest_closure,
            trip_duration_sec=trip_duration_sec,
            distraction_count=distraction_count,
            risk_score=risk_score,
        )

        return {
            "trip_id": trip_id,
            "driver_id": driver_id,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "trip_duration_min": round(trip_duration_sec / 60.0, 1),
            "average_speed_kmh": round(average_speed_kmh, 1),
            "total_alerts": total_alerts,
            "stage_breakdown": stage_counts,
            "longest_eye_closure_sec": round(longest_closure, 2),
            "distraction_events": distraction_count,
            "response_acknowledgements": acknowledged_count,
            "risk_score": risk_score,
            "risk_category": risk_category,
            "recommendations": recommendations,
        }

    @staticmethod
    def _generate_recommendations(
        stage_counts: Dict[int, int],
        longest_closure: float,
        trip_duration_sec: float,
        distraction_count: float,
        risk_score: float,
    ) -> List[str]:
        """Produce offline safety recommendations based on telemetry heuristics."""
        recs: List[str] = []

        if stage_counts[3] > 0 or longest_closure >= 2.5:
            recs.append(
                "CRITICAL: Sustained micro-sleep (>2.5s) detected. Driver must take an immediate 30-minute rest stop at the nearest secure parking bay."
            )

        if trip_duration_sec >= 7200:
            recs.append(
                "Continuous driving exceeded 2 hours. Scheduled 15-minute breaks are legally mandated for long-haul commercial vehicle operators."
            )

        if stage_counts[2] >= 3:
            recs.append(
                "Repeated drowsiness warnings observed. Ensure cab ventilation is open and avoid continuous night-shift driving without rest."
            )

        if distraction_count >= 2:
            recs.append(
                "Multiple mobile device distractions or off-road gaze episodes recorded. Keep mobile phone mounted and do not operate while vehicle is in motion."
            )

        if risk_score < 15 and not recs:
            recs.append(
                "Exemplary attentiveness maintained throughout the trip. Safe driving parameters observed."
            )

        return recs
