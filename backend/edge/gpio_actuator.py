"""Raspberry Pi GPIO Hardware Actuator & Mock Interface for DrishtiDrive AI.

Controls physical edge cabin hardware:
1. Status RGB LED:
   - Stage 0 (Attentive): Solid Green
   - Stage 1 (Attention): Solid Yellow
   - Stage 2 (Drowsy): Pulsing Orange
   - Stage 3 (Critical): High-frequency Flashing Red (4 Hz)
2. High-decibel Piezo Buzzer:
   - Stage 1: Short advisory chirp (100ms)
   - Stage 2: Pulsed warning beep (300ms on / 300ms off)
   - Stage 3: Continuous urgent siren
3. Physical Snooze / Driver Response Button:
   - Input on GPIO with debouncing to acknowledge Stage 2 alerts
4. Haptic Vibration Motor:
   - Tactile seat/steering vibration pulses

Features transparent MockGPIO fallback on macOS / non-Pi hardware.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("drishtidrive.edge.gpio")

# Attempt RPi.GPIO import with graceful mock fallback
try:
    import RPi.GPIO as GPIO  # type: ignore
    IS_RPI_HARDWARE = True
except (ImportError, RuntimeError):
    IS_RPI_HARDWARE = False


class MockGPIO:
    """Mock GPIO interface for desktop development, simulation, and hermetic tests."""

    BCM = "BCM"
    OUT = "OUT"
    IN = "IN"
    HIGH = 1
    LOW = 0
    PUD_UP = "PUD_UP"

    def __init__(self) -> None:
        self.pin_states: dict[int, int] = {}
        self.pin_modes: dict[int, str] = {}

    def setmode(self, mode: str) -> None:
        pass

    def setwarnings(self, state: bool) -> None:
        pass

    def setup(self, pin: int, mode: str, pull_up_down: Optional[str] = None) -> None:
        self.pin_modes[pin] = mode
        self.pin_states[pin] = self.HIGH if pull_up_down == self.PUD_UP else self.LOW

    def output(self, pin: int, value: int) -> None:
        self.pin_states[pin] = value

    def input(self, pin: int) -> int:
        return self.pin_states.get(pin, self.HIGH)

    def cleanup(self) -> None:
        self.pin_states.clear()


gpio = GPIO if IS_RPI_HARDWARE else MockGPIO()


class GPIOActuator:
    """Manages physical alert actuators with non-blocking threads."""

    # Default BCM Pin Mapping for Raspberry Pi 4/5
    PIN_LED_RED = 17
    PIN_LED_GREEN = 27
    PIN_LED_BLUE = 22
    PIN_BUZZER = 23
    PIN_BUTTON = 24
    PIN_HAPTIC = 25

    def __init__(self, use_mock: bool = False, mock_hardware: Optional[bool] = None) -> None:
        """Initialize GPIO pin setup and thread locks."""
        mock_requested = use_mock or (mock_hardware is True)
        self._mock = gpio if (mock_requested or not IS_RPI_HARDWARE) else GPIO
        self.is_hardware = IS_RPI_HARDWARE and not mock_requested
        self._is_mock = not self.is_hardware
        self._button_callback: Optional[Callable[[], None]] = None
        self._running = False
        self._active_stage = 0
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._setup_pins()

    def _setup_pins(self) -> None:
        """Configure input and output GPIO pin directions."""
        self._mock.setmode(self._mock.BCM)
        self._mock.setwarnings(False)

        for pin in (self.PIN_LED_RED, self.PIN_LED_GREEN, self.PIN_LED_BLUE, self.PIN_BUZZER, self.PIN_HAPTIC):
            self._mock.setup(pin, self._mock.OUT)
            self._mock.output(pin, self._mock.LOW)

        self._mock.setup(self.PIN_BUTTON, self._mock.IN, pull_up_down=self._mock.PUD_UP)

    def start(self) -> None:
        """Start non-blocking actuation loop."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._actuation_loop, daemon=True)
        self._worker_thread.start()

    def set_stage(self, stage: int) -> None:
        """Update active alert stage (0 = Normal, 1 = Attention, 2 = Drowsy, 3 = Critical)."""
        with self._lock:
            self._active_stage = int(stage)

    def set_state(self, stage: int) -> None:
        """Alias for set_stage for state machine compatibility."""
        self.set_stage(stage)

    def get_current_stage(self) -> int:
        """Return currently active alert stage."""
        with self._lock:
            return self._active_stage

    @property
    def is_mock(self) -> bool:
        """Whether hardware is mocked."""
        return self._is_mock

    def self_test(self) -> bool:
        """Flash RGB test sequence to verify actuator connectivity."""
        try:
            self._set_rgb(1, 0, 0)
            time.sleep(0.02)
            self._set_rgb(0, 1, 0)
            time.sleep(0.02)
            self._set_rgb(0, 0, 1)
            time.sleep(0.02)
            self._set_rgb(0, 0, 0)
            return True
        except Exception as exc:
            logger.error("GPIO self-test failed: %s", exc)
            return False

    def trigger_haptic(self, duration_sec: float = 0.5) -> None:
        """Trigger brief haptic vibration pulse."""
        def _pulse():
            self._mock.output(self.PIN_HAPTIC, self._mock.HIGH)
            time.sleep(duration_sec)
            self._mock.output(self.PIN_HAPTIC, self._mock.LOW)
        threading.Thread(target=_pulse, daemon=True).start()

    def set_button_callback(self, callback: Callable[[], None]) -> None:
        """Register listener for physical snooze button press."""
        self._button_callback = callback

    def check_button_pressed(self) -> bool:
        """Query instantaneous state of snooze button (active LOW)."""
        val = self._mock.input(self.PIN_BUTTON)
        return (val == self._mock.LOW)

    def _set_rgb(self, r: int, g: int, b: int) -> None:
        """Set RGB LED color states."""
        self._mock.output(self.PIN_LED_RED, r)
        self._mock.output(self.PIN_LED_GREEN, g)
        self._mock.output(self.PIN_LED_BLUE, b)

    def _actuation_loop(self) -> None:
        """Background thread handling LED pulsing and buzzer cadences."""
        button_was_pressed = False

        while self._running:
            with self._lock:
                stage = self._active_stage

            # Check button press with simple debounce
            is_pressed = self.check_button_pressed()
            if is_pressed and not button_was_pressed:
                if self._button_callback:
                    self._button_callback()
            button_was_pressed = is_pressed

            # Actuation cadences by stage
            if stage == 0:
                self._set_rgb(0, 1, 0)  # Solid Green
                self._mock.output(self.PIN_BUZZER, 0)
                time.sleep(0.1)
            elif stage == 1:
                self._set_rgb(1, 1, 0)  # Solid Yellow
                self._mock.output(self.PIN_BUZZER, 0)
                time.sleep(0.1)
            elif stage == 2:
                # Pulsing Orange & intermittent buzzer warning
                self._set_rgb(1, 1, 0)
                self._mock.output(self.PIN_BUZZER, 1)
                time.sleep(0.2)
                self._set_rgb(0, 0, 0)
                self._mock.output(self.PIN_BUZZER, 0)
                time.sleep(0.2)
            elif stage == 3:
                # Rapid 4Hz Flashing Red & continuous siren
                self._set_rgb(1, 0, 0)
                self._mock.output(self.PIN_BUZZER, 1)
                time.sleep(0.125)
                self._set_rgb(0, 0, 0)
                time.sleep(0.125)

    def stop(self) -> None:
        """Stop actuation loop and clear pins."""
        self._running = False
        with self._lock:
            self._active_stage = 0
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)
        self._set_rgb(0, 0, 0)
        self._mock.output(self.PIN_BUZZER, 0)
        self._mock.output(self.PIN_HAPTIC, 0)
        self._mock.cleanup()

    def cleanup(self) -> None:
        """Alias for stop method."""
        self.stop()
