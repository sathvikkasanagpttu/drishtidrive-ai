"""
Driver Response Verification Check Module
=========================================
At Alert Stage 2 (Drowsiness Warning), demands driver interaction (button press,
voice confirmation, or touchscreen acknowledgement) within an N-second window
(default: 5.0s). If unanswered, escalates directly to Stage 3 (Critical).
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Optional

from backend.temporal_state import AlertStage

logger = logging.getLogger("drishtidrive.context.response")


class ResponseStatus(str, Enum):
    IDLE = "IDLE"
    WAITING_ACKNOWLEDGEMENT = "WAITING_ACKNOWLEDGEMENT"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    EXPIRED_ESCALATED = "EXPIRED_ESCALATED"


class DriverResponseChecker:
    """
    Supervises Stage 2 driver attentiveness challenge and escalation timers.
    """

    def __init__(self, timeout_sec: float = 5.0):
        self.timeout_sec = timeout_sec
        self.status = ResponseStatus.IDLE
        self._challenge_start: Optional[float] = None
        self._last_ack_time: Optional[float] = None
        self._last_ack_source: Optional[str] = None

    def evaluate(
        self,
        current_stage: AlertStage,
        timestamp: Optional[float] = None,
    ) -> AlertStage:
        """
        Evaluate stage escalation based on driver response window.

        Args:
            current_stage: Proposed or active alert stage.
            timestamp: Monotonic or epoch timestamp.

        Returns:
            AlertStage.STAGE_3_CRITICAL if response timed out, else current_stage.
        """
        now = timestamp if timestamp is not None else time.time()

        if current_stage == AlertStage.STAGE_2_DROWSY:
            if self.status in (ResponseStatus.IDLE, ResponseStatus.ACKNOWLEDGED):
                # Start new challenge if not already in flight
                if self.status == ResponseStatus.IDLE:
                    self._start_challenge(now)
            elif self.status == ResponseStatus.WAITING_ACKNOWLEDGEMENT:
                assert self._challenge_start is not None
                elapsed = now - self._challenge_start
                if elapsed >= self.timeout_sec:
                    logger.warning(
                        "Driver failed Stage 2 response check (%.1fs elapsed). Escalating to Stage 3!",
                        elapsed,
                    )
                    self.status = ResponseStatus.EXPIRED_ESCALATED
                    return AlertStage.STAGE_3_CRITICAL
            elif self.status == ResponseStatus.EXPIRED_ESCALATED:
                return AlertStage.STAGE_3_CRITICAL

        elif current_stage < AlertStage.STAGE_2_DROWSY:
            # Stage cleared or downgraded; reset challenge state
            self.reset()

        return current_stage

    def acknowledge(self, source: str = "button", timestamp: Optional[float] = None) -> bool:
        """
        Record driver acknowledgement (button press, voice command, or UI tap).

        Returns:
            True if acknowledgement was active and accepted, False if no challenge was pending.
        """
        now = timestamp if timestamp is not None else time.time()
        if self.status in (ResponseStatus.WAITING_ACKNOWLEDGEMENT, ResponseStatus.EXPIRED_ESCALATED):
            logger.info("Driver response confirmed via %s", source)
            self.status = ResponseStatus.ACKNOWLEDGED
            self._last_ack_time = now
            self._last_ack_source = source
            return True
        return False

    def get_remaining_time(self, timestamp: Optional[float] = None) -> float:
        """Seconds remaining before unresponsiveness triggers Stage 3 escalation."""
        if self.status != ResponseStatus.WAITING_ACKNOWLEDGEMENT or self._challenge_start is None:
            return 0.0
        now = timestamp if timestamp is not None else time.time()
        return max(0.0, self.timeout_sec - (now - self._challenge_start))

    def reset(self) -> None:
        """Clear active challenge."""
        self.status = ResponseStatus.IDLE
        self._challenge_start = None

    def _start_challenge(self, start_time: float) -> None:
        """Initiate timeout countdown."""
        self.status = ResponseStatus.WAITING_ACKNOWLEDGEMENT
        self._challenge_start = start_time
        logger.info("Initiated Stage 2 driver response check (timeout: %.1fs)", self.timeout_sec)
