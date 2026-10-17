"""Eye-gesture input for the ``eyes`` clutch: per-eye open/closed -> closed durations.

Aria Gen 2's on-device eye tracking says, per eye, whether it currently sees
the eye. The flags are ``EyeGaze.vergence.left_blink`` / ``right_blink``, but
on the glasses they read 1 while the eye is open and 0 while it is closed
(they match ``left_gaze_valid`` / ``right_gaze_valid``). Measured on one
wearer: ordinary blinks close both eyes together for at most ~0.4 s, and a
deliberate wink keeps one eye closed while the other reads open, apart from
brief blips.

``EyeTracker`` debounces those samples and reports how long each eye, and both
together, have been closed. ``TeleopMapper`` turns that into commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Callable

SIDES = ("left", "right")


@dataclass
class EyeState:
    left_closed_s: float  # how long the left eye has read closed; 0 while open
    right_closed_s: float
    both_closed_s: float  # how long both have been closed at the same time
    received_at: float  # host clock of the newest sample


class EyeTracker:
    """Called from the SDK's eye-gaze thread; read with ``state()`` from the main loop."""

    def __init__(self, reopen_s: float = 0.03, clock: Callable[[], float] = time.monotonic):
        # An eye counts as reopened after reading open this long: two samples in a row
        # at 30 Hz. Longer stretches every blink (the longest measured was 0.43 s with
        # this setting, against a 0.5 s pause threshold).
        self.reopen_s = reopen_s
        self._clock = clock
        self._lock = threading.Lock()
        self._closed_since: dict[str, float | None] = {s: None for s in SIDES}
        self._open_since: dict[str, float | None] = {s: None for s in SIDES}
        self._both_since: float | None = None
        self._t: float | None = None
        self._received_at = -math.inf
        self.samples = 0

    def on_sample(self, t_ns: int, left_open: bool | None, right_open: bool | None) -> None:
        """One eye-tracking result; ``None`` for an eye means "unknown", which changes nothing."""
        t = t_ns * 1e-9
        with self._lock:
            if self._t is not None and t <= self._t:
                return
            for side, is_open in zip(SIDES, (left_open, right_open)):
                if is_open is not None:
                    self._update_eye(side, is_open, t)
            if all(self._closed_since[s] is not None for s in SIDES):
                if self._both_since is None:
                    self._both_since = max(self._closed_since[s] for s in SIDES)
            else:
                self._both_since = None
            self._t = t
            self._received_at = self._clock()
            self.samples += 1

    def _update_eye(self, side: str, is_open: bool, t: float) -> None:
        if not is_open:
            self._open_since[side] = None
            if self._closed_since[side] is None:
                self._closed_since[side] = t
        elif self._closed_since[side] is not None:
            if self._open_since[side] is None:
                self._open_since[side] = t
            if t - self._open_since[side] >= self.reopen_s:
                self._closed_since[side] = self._open_since[side] = None

    def state(self) -> EyeState | None:
        with self._lock:
            if self._t is None:
                return None
            t = self._t

            def closed_for(since: float | None) -> float:
                return 0.0 if since is None else t - since

            return EyeState(
                closed_for(self._closed_since["left"]),
                closed_for(self._closed_since["right"]),
                closed_for(self._both_since),
                self._received_at,
            )
