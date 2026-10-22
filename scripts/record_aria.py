"""Record a short session from Aria Gen 2 glasses: hands, head pose, eyes, and a few RGB frames.

    python scripts/record_aria.py --device-ip <GLASSES_IP>      # 30 s, prints what to do when
    python scripts/record_aria.py --source mock                 # no glasses: to try the plots

Writes ``recordings/<source>_<time>.npz`` for ``scripts/plot_recording.py``. The hand
landmarks are also projected into each saved RGB frame with the glasses' own camera
calibration. RGB frames show whatever the glasses saw: check them before you publish.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import threading
import time

import numpy as np

from aria_teleop.fusion import HandFuser
from aria_teleop.sources import NUM_LANDMARKS, AriaSource, MockSource

SIDES = ("left", "right")
CUES = [  # (seconds from start, what to do); the plots use these segments
    (0.0, "hold your RIGHT hand still in front of you and turn your head slowly left and right"),
    (8.0, "now move your right hand around slowly, keeping it in view; pinch a few times"),
    (17.0, "wink LEFT (hold ~1 s), relax, wink RIGHT (~1 s), then close both eyes for ~1 s"),
    (25.0, "hold up both hands in front of you, fingers spread"),
]
MAX_RGB_WIDTH = 960  # saved frames are downscaled to this


class Recorder:
    """Taps the fuser's inputs, so it records exactly what the teleop sees."""

    def __init__(self, fuser: HandFuser, rgb_every_s: float):
        self.t0 = time.monotonic()
        self.lock = threading.Lock()
        self.hands: list[tuple[int, float, dict]] = []
        self.heads: list[tuple[int, np.ndarray]] = []
        self.eyes: list[tuple[int, float, float, float]] = []
        self.rgb: list[tuple[int, float, np.ndarray]] = []
        self.calib = None
        self.rgb_every_s = rgb_every_s
        self._last_rgb = -np.inf
        self._rgb_note = False
        on_hands, on_head, on_eye = fuser.on_hands, fuser.on_head_pose, fuser.eyes.on_sample

        def hands(t_ns, observations):
            with self.lock:
                self.hands.append((t_ns, self.now(), observations))
            on_hands(t_ns, observations)

        def head(t_ns, T, session=None, gravity=None):
            with self.lock:
                self.heads.append((t_ns, np.array(T, dtype=float)))
            on_head(t_ns, T, session=session, gravity=gravity)

        def eye(t_ns, left_open, right_open):
            with self.lock:
                self.eyes.append((t_ns, self.now(), _flag(left_open), _flag(right_open)))
            on_eye(t_ns, left_open, right_open)

        fuser.on_hands, fuser.on_head_pose, fuser.eyes.on_sample = hands, head, eye

    def now(self) -> float:
        return time.monotonic() - self.t0

    def on_rgb(self, *args) -> None:
        """RGB callback; tolerant of how the SDK passes the image and its record."""
        if self.now() - self._last_rgb < self.rgb_every_s:
            return
        image = next((a for a in args if isinstance(a, np.ndarray)), None)
        if image is None:
            image = next((a.to_numpy_array() for a in args if hasattr(a, "to_numpy_array")), None)
        record = next((a for a in args if hasattr(a, "capture_timestamp_ns")), None)
        if image is None or record is None:
            if not self._rgb_note:
                self._rgb_note = True
                print(f"RGB callback: unexpected arguments {[type(a).__name__ for a in args]}; skipping frames")
            return
        self._last_rgb = self.now()
        with self.lock:
            self.rgb.append((int(record.capture_timestamp_ns), self._last_rgb, np.array(image)))


def _flag(v) -> float:
    return np.nan if v is None else float(bool(v))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", choices=["aria", "mock"], default="aria")
    p.add_argument("--device-ip", default="")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--rgb-every", type=float, default=0.5, help="keep one RGB frame per this many seconds")
    p.add_argument("--out-dir", default="recordings")
    args = p.parse_args()

    fuser = HandFuser()
    rec = Recorder(fuser, args.rgb_every)
    if args.source == "mock":
        source = MockSource(fuser)
    else:
        source = AriaSource(fuser, device_ip=args.device_ip, eyes=True, on_rgb=rec.on_rgb)
        on_calib = source._on_calib

        def calib(c):
            rec.calib = c
            on_calib(c)

        source._on_calib = calib  # read when start() registers the callback
    source.start()
    rec.t0 = time.monotonic()
    print(f"recording {args.seconds:.0f} s from {args.source}")
    try:
        cues = list(CUES)
        while rec.now() < args.seconds:
            if cues and rec.now() >= cues[0][0]:
                print(f"[{rec.now():5.1f}s] {cues.pop(0)[1]}")
            time.sleep(0.05)
    finally:
        source.stop()
    out = save(rec, args)
    print(f"saved {out}: {len(rec.hands)} hand frames, {len(rec.heads)} head poses, "
          f"{len(rec.eyes)} eye samples, {len(rec.rgb)} RGB frames")


def save(rec: Recorder, args) -> Path:
    n = len(rec.hands)
    data: dict[str, np.ndarray] = {
        "hand_t_ns": np.array([h[0] for h in rec.hands], dtype=np.int64),
        "hand_host_s": np.array([h[1] for h in rec.hands]),
        "head_t_ns": np.array([h[0] for h in rec.heads], dtype=np.int64),
        "head_T_room_device": np.array([h[1] for h in rec.heads]).reshape(-1, 4, 4),
        "eye_t_ns": np.array([e[0] for e in rec.eyes], dtype=np.int64),
        "eye_host_s": np.array([e[1] for e in rec.eyes]),
        "eye_left_open": np.array([e[2] for e in rec.eyes]),
        "eye_right_open": np.array([e[3] for e in rec.eyes]),
        "cue_s": np.array([c[0] for c in CUES]),
        "cue_text": np.array([c[1] for c in CUES]),
        "source": np.array(args.source),
    }
    for side in SIDES:
        wrist = np.full((n, 4, 4), np.nan)
        landmarks = np.full((n, NUM_LANDMARKS, 3), np.nan)
        conf = np.full(n, np.nan)
        for i, (_, _, obs) in enumerate(rec.hands):
            o = obs.get(side)
            if o is None:
                continue
            wrist[i] = o.T_device_wrist
            if o.landmarks_device is not None:
                landmarks[i] = o.landmarks_device
            conf[i] = o.confidence
        data[f"{side}_T_device_wrist"], data[f"{side}_landmarks_device"], data[f"{side}_confidence"] = wrist, landmarks, conf

    if rec.rgb:
        frames, pixels = [], []
        for t_ns, host_s, image in rec.rgb:
            scale = min(1.0, MAX_RGB_WIDTH / image.shape[1])
            small = _downscale(image, scale)
            frames.append(small)
            pixels.append(_project_hands(rec, data, t_ns, image.shape, scale))
        data["rgb_t_ns"] = np.array([r[0] for r in rec.rgb], dtype=np.int64)
        data["rgb_host_s"] = np.array([r[1] for r in rec.rgb])
        data["rgb_frames"] = np.array(frames)
        data["rgb_hand_px"] = np.array(pixels)  # (frames, 2 sides, 21, 2), NaN where not visible

    out = Path(args.out_dir) / f"{args.source}_{datetime.now():%Y%m%d_%H%M%S}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)
    return out


def _downscale(image: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return image
    from PIL import Image

    h, w = image.shape[:2]
    return np.asarray(Image.fromarray(image).resize((round(w * scale), round(h * scale)), Image.BILINEAR))


def _project_hands(rec: Recorder, data: dict, t_ns: int, shape: tuple, scale: float) -> np.ndarray:
    """Landmarks of the hand frame nearest ``t_ns`` (within 50 ms), as pixels in the saved frame."""
    px = np.full((len(SIDES), NUM_LANDMARKS, 2), np.nan)
    if rec.calib is None or not len(data["hand_t_ns"]):
        return px
    i = int(np.argmin(np.abs(data["hand_t_ns"] - t_ns)))
    if abs(int(data["hand_t_ns"][i]) - t_ns) > 50_000_000:
        return px
    cam = rec.calib.get_camera_calib("camera-rgb")
    T_camera_device = np.linalg.inv(cam.get_transform_device_camera().to_matrix())
    calib_w = cam.get_image_size()[0]
    ratio = shape[1] / calib_w * scale  # the stream may be smaller than the calibrated image
    for s, side in enumerate(SIDES):
        for k, p_device in enumerate(data[f"{side}_landmarks_device"][i]):
            if not np.all(np.isfinite(p_device)):
                continue
            p_cam = T_camera_device[:3, :3] @ p_device + T_camera_device[:3, 3]
            uv = cam.project(p_cam)
            if uv is not None:
                px[s, k] = np.asarray(uv).reshape(2) * ratio
    return px


if __name__ == "__main__":
    main()
