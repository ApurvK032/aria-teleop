"""One Euro filters (Casiez et al., CHI 2012) for positions and rotations.

A speed-adaptive low-pass: heavy smoothing while the hand is still (removes
jitter), light smoothing while it moves fast (keeps lag down).
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import from_rotvec, rotvec


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One Euro filter for a vector; the cutoff adapts to the norm of its velocity."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self._x is None:
            self._x, self._dx, self._t = x.copy(), np.zeros_like(x), t
            return self._x.copy()
        dt = t - self._t
        if dt <= 0.0:
            return self._x.copy()
        self._t = t
        self._dx = self._dx + _alpha(self.d_cutoff, dt) * ((x - self._x) / dt - self._dx)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        self._x = self._x + _alpha(cutoff, dt) * (x - self._x)
        return self._x.copy()


class OneEuroRotationFilter:
    """One Euro filter on SO(3): smooths in the tangent space of the last estimate."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._R: np.ndarray | None = None
        self._w = np.zeros(3)
        self._t: float | None = None

    def __call__(self, R: np.ndarray, t: float) -> np.ndarray:
        R = np.asarray(R, dtype=float)
        if self._R is None:
            self._R, self._w, self._t = R.copy(), np.zeros(3), t
            return self._R.copy()
        dt = t - self._t
        if dt <= 0.0:
            return self._R.copy()
        self._t = t
        delta = rotvec(self._R.T @ R)
        self._w = self._w + _alpha(self.d_cutoff, dt) * (delta / dt - self._w)
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._w))
        self._R = self._R @ from_rotvec(_alpha(cutoff, dt) * delta)
        return self._R.copy()
