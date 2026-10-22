"""Make the README figures from a recording made with ``scripts/record_aria.py``.

    python scripts/plot_recording.py recordings/aria_<time>.npz            # -> docs/media/

Replays the recording through the project's own head-motion compensation
(``PoseBuffer``) and eye clutch (``EyeTracker`` + ``TeleopMapper``), so the figures
show exactly what the teleop computes.

- ``aria_rgb_hands.jpg``: an RGB frame from the glasses with the tracked hands drawn on it
- ``head_compensation.png``: wrist motion in the glasses' frame vs the room frame while the head turns
- ``hand_path_3d.png``: the right hand in the room frame while it moves
- ``eye_clutch.png``: eye closures and the clutch state they produce
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from aria_teleop.eyes import EyeTracker  # noqa: E402
from aria_teleop.fusion import PoseBuffer  # noqa: E402
from aria_teleop.hand_viewer import BONES  # noqa: E402
from aria_teleop.mapping import Status, TeleopConfig, TeleopMapper  # noqa: E402

# Validated categorical slots (light surface), assigned in fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HAND_COLORS = {"left": BLUE, "right": ORANGE}
HOME_P, HOME_R = np.array([0.3, 0.0, 0.2]), np.eye(3)  # sim.HOME_POSITION / HOME_ROTATION (avoids importing MuJoCo)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10, "lines.linewidth": 2.0,
    "legend.frameon": False,
})  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("recording")
    p.add_argument("--out-dir", default="docs/media")
    args = p.parse_args()
    d = dict(np.load(args.recording, allow_pickle=False))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = 0.0
    cues = d["cue_s"]
    room = to_room_frame(d)

    made = [
        rgb_overlay(d, cues, out / "aria_rgb_hands.jpg"),
        head_compensation(d, room, cues, t0, out / "head_compensation.png"),
        hand_path_3d(d, room, cues, out / "hand_path_3d.png"),
        eye_clutch(d, cues, out / "eye_clutch.png"),
    ]
    for path in made:
        if path is not None:
            print(path)


def segment(t: np.ndarray, cues: np.ndarray, k: int) -> np.ndarray:
    end = cues[k + 1] if k + 1 < len(cues) else np.inf
    return (t >= cues[k]) & (t < end)


def to_room_frame(d: dict) -> dict:
    """Wrist poses and landmarks in the room frame, composed with the head pose at each hand timestamp."""
    poses = PoseBuffer(max_len=len(d["head_t_ns"]) + 1)
    for t_ns, T in zip(d["head_t_ns"], d["head_T_room_device"]):
        poses.add(int(t_ns), T)
    n = len(d["hand_t_ns"])
    out = {"T_room_device": np.full((n, 4, 4), np.nan)}
    for side in ("left", "right"):
        out[f"{side}_wrist"] = np.full((n, 3), np.nan)
        out[f"{side}_landmarks"] = np.full((n, 21, 3), np.nan)
    for i, t_ns in enumerate(d["hand_t_ns"]):
        T = poses.lookup(int(t_ns))
        if T is None:
            continue
        out["T_room_device"][i] = T
        for side in ("left", "right"):
            W = d[f"{side}_T_device_wrist"][i]
            if np.isfinite(W).all():
                out[f"{side}_wrist"][i] = (T @ W)[:3, 3]
            L = d[f"{side}_landmarks_device"][i]
            out[f"{side}_landmarks"][i] = L @ T[:3, :3].T + T[:3, 3]
    return out


def rgb_overlay(d: dict, cues: np.ndarray, path: Path) -> Path | None:
    if "rgb_frames" not in d:
        return None
    px = d["rgb_hand_px"]  # (frames, sides, 21, 2)
    visible = np.isfinite(px).all(axis=-1).sum(axis=(1, 2))
    in_last = segment(d["rgb_host_s"], cues, len(cues) - 1)
    score = visible + 100 * in_last  # prefer the both-hands segment, then the most landmarks
    i = int(np.argmax(score))
    if visible[i] == 0:
        return None
    image = d["rgb_frames"][i]
    h = image.shape[0]
    upright = Image.fromarray(np.rot90(image, -1).copy())  # Aria's RGB sensor is mounted sideways
    draw = ImageDraw.Draw(upright)
    width = max(2, upright.width // 240)
    for s, side in enumerate(("left", "right")):
        uv = px[i, s]
        xy = np.stack([h - 1 - uv[:, 1], uv[:, 0]], axis=1)  # pixel coords after the clockwise rotation
        color = HAND_COLORS[side]
        for a, b in BONES:
            if np.isfinite(xy[[a, b]]).all():
                draw.line([tuple(xy[a]), tuple(xy[b])], fill=color, width=width)
        r = width + 1
        for x, y in xy[np.isfinite(xy).all(axis=1)]:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="white", width=1)
    upright.save(path, quality=90)
    return path


def head_compensation(d: dict, room: dict, cues: np.ndarray, t0: float, path: Path) -> Path | None:
    t = d["hand_host_s"]
    sel = segment(t, cues, 0) & np.isfinite(room["right_wrist"]).all(axis=1)
    if sel.sum() < 10:
        return None
    t = t[sel]
    p_device = d["right_T_device_wrist"][sel][:, :3, 3]
    p_room = room["right_wrist"][sel]
    disp_device = np.linalg.norm(p_device - np.median(p_device, axis=0), axis=1) * 100
    disp_room = np.linalg.norm(p_room - np.median(p_room, axis=0), axis=1) * 100
    fwd = room["T_room_device"][sel][:, :3, 2]  # device z = looking direction
    yaw = np.degrees(np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0])))
    yaw -= np.median(yaw)

    fig, (ax_yaw, ax) = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True, height_ratios=[1, 2.2],
                                     layout="constrained")
    ax_yaw.plot(t, yaw, color=INK_2, linewidth=1.5)
    ax_yaw.set_ylabel("head yaw (°)")
    ax_yaw.set_title("Turning your head doesn't move the robot")
    ax.plot(t, disp_device, color=ORANGE)
    ax.plot(t, disp_room, color=BLUE)
    for y, label in ((disp_device, "glasses' frame (raw)"), (disp_room, "room frame (compensated)")):
        ax.annotate(f"{label}\n{np.sqrt(np.mean(y ** 2)):.1f} cm RMS", (t[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_ylabel("wrist movement (cm)")
    ax.set_xlabel("time (s)  ·  hand held still while the head turns")
    ax.set_ylim(bottom=0)
    ax.set_xlim(t[0], t[-1] + 0.32 * (t[-1] - t[0]))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def hand_path_3d(d: dict, room: dict, cues: np.ndarray, path: Path) -> Path | None:
    t = d["hand_host_s"]
    sel = segment(t, cues, 1) & np.isfinite(room["right_wrist"]).all(axis=1) & (d["right_confidence"] >= 0.5)
    if sel.sum() < 10:
        return None
    idx = np.flatnonzero(sel)
    p = room["right_wrist"][idx] * 100
    # Drop single-frame glitches, as the teleop does: points far from their neighbours' median.
    k = 5
    padded = np.pad(p, ((k // 2, k // 2), (0, 0)), mode="edge")
    local = np.median(np.stack([padded[i : i + len(p)] for i in range(k)]), axis=0)
    keep = np.linalg.norm(p - local, axis=1) < 3.0
    idx, p = idx[keep], p[keep]
    fig = plt.figure(figsize=(6.4, 5.2), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color=ORANGE, linewidth=1.6)
    for j in idx[:: max(1, len(idx) // 5)]:  # a few hand snapshots along the path
        L = room["right_landmarks"][j] * 100
        for a, b in BONES:
            if np.isfinite(L[[a, b]]).all():
                ax.plot(*L[[a, b]].T, color=INK_2, linewidth=1.0)
    ax.set_title("Right hand in the room frame (cm)")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(SURFACE)
        axis.pane.set_edgecolor(GRID)
    ax.set_xlabel("x (forward)")
    ax.set_ylabel("y (left)")
    ax.set_zlabel("z (up)")
    ax.set_box_aspect(np.ptp(p, axis=0) + 1e-6)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def eye_clutch(d: dict, cues: np.ndarray, path: Path) -> Path | None:
    host = d["eye_host_s"]
    if len(host) < 10:
        return None
    clock = [0.0]
    tracker = EyeTracker(clock=lambda: clock[0])
    mapper = TeleopMapper(TeleopConfig(clutch="eyes"), HOME_P, HOME_R)
    mapper.toggle_clutch()  # armed, as after SPACE
    states = []
    for t_ns, h, left, right in zip(d["eye_t_ns"], host, d["eye_left_open"], d["eye_right_open"]):
        clock[0] = h
        tracker.on_sample(int(t_ns), None if np.isnan(left) else bool(left), None if np.isnan(right) else bool(right))
        cmd = mapper.update(None, h, tracker.state())
        states.append("paused" if cmd.status is Status.PAUSED else "engaged" if mapper.engaged else "armed")
    start = cues[2] - 1.0 if len(cues) > 2 else host[0]
    end = cues[3] + 0.5 if len(cues) > 3 else host[-1]
    sel = (host >= start) & (host <= end)
    if sel.sum() < 10:
        return None

    fig, ax = plt.subplots(figsize=(8, 2.9), layout="constrained")
    rows = [("left eye closed", d["eye_left_open"] == 0, BLUE, 2),
            ("right eye closed", d["eye_right_open"] == 0, ORANGE, 1)]
    for label, closed, color, y in rows:
        for a, b in _runs(host[sel], closed[sel]):
            ax.broken_barh([(a, b - a)], (y - 0.3, 0.6), color=color)
    state_colors = {"engaged": AQUA, "paused": INK_2, "armed": GRID}
    for name in ("armed", "engaged", "paused"):
        mask = np.array([s == name for s in states])[sel]
        for a, b in _runs(host[sel], mask):
            ax.broken_barh([(a, b - a)], (-0.3, 0.6), color=state_colors[name])
            if b - a > 0.6:
                ax.text((a + b) / 2, 0, name, ha="center", va="center", fontsize=9,
                        color=INK if name != "paused" else SURFACE)
    ax.set_yticks([2, 1, 0], [r[0] for r in rows] + ["clutch"])
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(start, end)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("time (s)")
    ax.set_title("Eye-gesture clutch")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _runs(t: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """(start, end) times of each run of True in ``mask``."""
    runs, start = [], None
    for ti, m in zip(t, mask):
        if m and start is None:
            start = ti
        elif not m and start is not None:
            runs.append((start, ti))
            start = None
    if start is not None:
        runs.append((start, t[-1]))
    return runs


if __name__ == "__main__":
    main()
