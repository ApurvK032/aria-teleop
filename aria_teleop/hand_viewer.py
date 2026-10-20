"""Optional Rerun window that shows the tracked hands live (``--hand-viewer``).

Draws what the teleop actually sees: both hands after head-motion compensation,
in the room frame, plus the head pose and time plots of confidence and pinch
distance. Turning your head should not move the hands here. Hand frames are
logged from the main loop at most ``rate_hz`` times a second; cameras are never
logged, so the viewer's memory stays small (capped by ``memory_limit``).
"""

from __future__ import annotations

import numpy as np

from .eyes import EyeState
from .fusion import HandFrame
from .mapping import pinch_distance

# projectaria_tools hand_tracking.kHandJointConnections, as landmark index pairs
# (kept here so the viewer also works with the mock, without projectaria_tools).
BONES = [
    (5, 17), (17, 18), (18, 19), (19, 4),
    (5, 14), (14, 15), (15, 16), (16, 3),
    (5, 11), (11, 12), (12, 13), (13, 2),
    (5, 8), (8, 9), (9, 10), (10, 1),
    (5, 6), (6, 7), (7, 0),
    (6, 8), (8, 11), (11, 14), (14, 17),
]  # fmt: skip
COLORS = {"left": (80, 160, 255), "right": (255, 140, 40)}


class HandViewer:
    def __init__(self, memory_limit: str = "1GB", rate_hz: float = 30.0, spawn: bool = True):
        import rerun as rr  # comes with the [aria] extra

        self.rr = rr
        rr.init("aria_teleop_hands")
        if spawn:
            rr.spawn(memory_limit=memory_limit)
        rr.log("room", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # gravity is -z in the room frame
        self.period = 1.0 / rate_hz
        self._last_t_ns: int | None = None
        self._last_log = -np.inf

    def log(self, frame: HandFrame | None, now: float, eyes: EyeState | None = None) -> None:
        if frame is None or frame.t_ns == self._last_t_ns or now - self._last_log < self.period:
            return
        rr = self.rr
        self._last_t_ns, self._last_log = frame.t_ns, now
        rr.set_time("device_time", duration=frame.t_ns * 1e-9)
        if eyes is not None:
            for name in ("left_closed_s", "right_closed_s", "both_closed_s"):
                rr.log(f"plots/eyes/{name}", rr.Scalars(getattr(eyes, name)))
        T = frame.T_room_head
        rr.log("room/head", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]), rr.TransformAxes3D(0.1))
        for side, color in COLORS.items():
            hand = frame.hands.get(side)
            if hand is None or hand.landmarks_room is None:
                rr.log(f"room/{side}", rr.Clear(recursive=True))
                continue
            pts = np.asarray(hand.landmarks_room)
            bones = [[pts[a], pts[b]] for a, b in BONES if max(a, b) < len(pts)]
            rr.log(f"room/{side}/landmarks", rr.Points3D(pts, radii=0.004, colors=color))
            rr.log(f"room/{side}/bones", rr.LineStrips3D(bones, radii=0.0015, colors=color))
            W = hand.T_room_wrist
            rr.log(f"room/{side}/wrist", rr.Transform3D(translation=W[:3, 3], mat3x3=W[:3, :3]), rr.TransformAxes3D(0.05))
            rr.log(f"plots/confidence/{side}", rr.Scalars(hand.confidence))
            d = pinch_distance(hand)
            if d is not None:
                rr.log(f"plots/pinch_cm/{side}", rr.Scalars(d * 100))
