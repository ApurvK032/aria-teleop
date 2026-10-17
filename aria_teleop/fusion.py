"""Head-motion compensation: device-frame hand tracking -> room-frame hands.

Aria reports hands in the glasses' own (device) frame, so turning your head
moves them. Composing with the on-device VIO head pose at the same timestamp
gives the hand in the VIO odometry frame, called the "room" frame here:

    T_room_wrist = T_room_device(t) @ T_device_wrist(t)

Callbacks arrive on SDK threads, so everything here is guarded by one lock and
does only a few small matrix products.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
import threading
import time
from typing import Callable

import numpy as np

from .geometry import interpolate_transform

GRAVITY_DEFAULT = np.array([0.0, 0.0, -9.81])


@dataclass
class HandObservation:
    """One hand as reported by the glasses, in the device frame."""

    T_device_wrist: np.ndarray
    landmarks_device: np.ndarray | None  # (21, 3), Aria landmark order
    confidence: float


@dataclass
class HandState:
    """One hand in the room frame."""

    T_room_wrist: np.ndarray
    landmarks_room: np.ndarray | None
    confidence: float


@dataclass
class HandFrame:
    t_ns: int  # device time of the hand estimate
    received_at: float  # host clock when the frame was fused
    hands: dict[str, HandState]  # "left" / "right"; a missing key means not tracked
    T_room_head: np.ndarray  # head pose (CPF when calibration is known, else device frame)
    gravity_room: np.ndarray
    session: object  # VIO session; poses from different sessions are not comparable


class PoseBuffer:
    """Recent head poses, looked up by device timestamp with interpolation."""

    def __init__(self, max_len: int = 4000, max_extrapolation_ns: int = 50_000_000):
        self.max_len = max_len
        self.max_extrapolation_ns = max_extrapolation_ns
        self.session: object = None
        self._t: list[int] = []
        self._T: list[np.ndarray] = []

    def add(self, t_ns: int, T: np.ndarray, session: object = None) -> None:
        if session != self.session:
            self._t.clear()
            self._T.clear()
            self.session = session
        if self._t and t_ns <= self._t[-1]:
            return
        self._t.append(t_ns)
        self._T.append(np.array(T, dtype=float))
        if len(self._t) > 2 * self.max_len:
            del self._t[: self.max_len]
            del self._T[: self.max_len]

    def lookup(self, t_ns: int) -> np.ndarray | None:
        """Pose at ``t_ns``; holds the newest pose for a short gap past the end."""
        if not self._t or t_ns < self._t[0]:
            return None
        if t_ns >= self._t[-1]:
            if t_ns - self._t[-1] > self.max_extrapolation_ns:
                return None
            return self._T[-1].copy()
        i = bisect.bisect_right(self._t, t_ns)
        t0, t1 = self._t[i - 1], self._t[i]
        return interpolate_transform(self._T[i - 1], self._T[i], (t_ns - t0) / (t1 - t0))


class HandFuser:
    """Combines the hand and head-pose streams into room-frame ``HandFrame``s."""

    def __init__(
        self,
        max_extrapolation_ms: float = 50.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._lock = threading.Lock()
        self._poses = PoseBuffer(max_extrapolation_ns=int(max_extrapolation_ms * 1e6))
        self._gravity = GRAVITY_DEFAULT.copy()
        self._T_device_head = np.eye(4)
        self._latest: HandFrame | None = None
        self._clock = clock
        self.hand_frames = 0
        self.head_poses = 0
        self.frames_without_pose = 0
        self.has_head_calibration = False

    def set_device_to_head(self, T_device_head: np.ndarray) -> None:
        """Device -> CPF transform from the glasses' calibration (used for heading)."""
        with self._lock:
            self._T_device_head = np.array(T_device_head, dtype=float)
            self.has_head_calibration = True

    def on_head_pose(
        self,
        t_ns: int,
        T_room_device: np.ndarray,
        session: object = None,
        gravity: np.ndarray | None = None,
    ) -> None:
        with self._lock:
            self._poses.add(t_ns, T_room_device, session)
            if gravity is not None and np.linalg.norm(gravity) > 1.0:
                self._gravity = np.asarray(gravity, dtype=float).reshape(3)
            self.head_poses += 1

    def on_hands(self, t_ns: int, observations: dict[str, HandObservation]) -> None:
        with self._lock:
            T_room_device = self._poses.lookup(t_ns)
            if T_room_device is None:
                self.frames_without_pose += 1
                return
            R, p = T_room_device[:3, :3], T_room_device[:3, 3]
            hands = {}
            for side, obs in observations.items():
                landmarks = None
                if obs.landmarks_device is not None:
                    landmarks = obs.landmarks_device @ R.T + p
                hands[side] = HandState(T_room_device @ obs.T_device_wrist, landmarks, obs.confidence)
            self._latest = HandFrame(
                t_ns=t_ns,
                received_at=self._clock(),
                hands=hands,
                T_room_head=T_room_device @ self._T_device_head,
                gravity_room=self._gravity.copy(),
                session=self._poses.session,
            )
            self.hand_frames += 1

    def latest(self) -> HandFrame | None:
        with self._lock:
            return self._latest
