"""Render the README's MuJoCo demo animation from the scripted mock hand.

Runs the same pipeline as ``python -m aria_teleop --source mock`` (head-motion
compensation, clutch, filtering, IK) on a virtual clock, so it is deterministic
and needs no glasses and no display. Renders offscreen with EGL.

    python scripts/render_sim_demo.py                     # -> docs/media/sim_demo.gif
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from aria_teleop.fusion import HandFuser  # noqa: E402
from aria_teleop.mapping import PoseSmoother, TeleopConfig, TeleopMapper  # noqa: E402
from aria_teleop.sim import HOME_POSITION, HOME_ROTATION, WidowXSim  # noqa: E402
from aria_teleop.sources import MockSource  # noqa: E402

TRAIL_S = 2.0  # how much of the gripper's recent path to draw


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="docs/media/sim_demo.gif")
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--size", default="560x400", help="WIDTHxHEIGHT")
    args = p.parse_args()
    width, height = map(int, args.size.split("x"))

    clock = [0.0]
    fuser = HandFuser(clock=lambda: clock[0])
    mock = MockSource(fuser, dropout=0.0, glitch=0.0)
    sim = WidowXSim(PoseSmoother())
    mapper = TeleopMapper(TeleopConfig(), HOME_POSITION, HOME_ROTATION)

    renderer = mujoco.Renderer(sim.model, height, width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.27, 0.0, 0.17]
    cam.distance, cam.azimuth, cam.elevation = 0.78, 140.0, -20.0

    dt = sim.timestep
    next_head = next_hand = next_frame = 0.0
    trail: list[tuple[float, np.ndarray]] = []
    frames: list[Image.Image] = []
    engaged = False
    t = 0.0
    while t < args.seconds:
        if t >= next_head:
            fuser.on_head_pose(int(t * 1e9), mock.head_pose(t), session=0)
            next_head += 1.0 / mock.head_rate_hz
        if t >= next_hand:
            fuser.on_hands(int(t * 1e9), mock.observe(t))
            fuser.eyes.on_sample(int(t * 1e9), True, True)
            next_hand += 1.0 / mock.hand_rate_hz
        if not engaged and t >= 1.0:
            mapper.set_clutch(True)
            engaged = True
        cmd = mapper.update(fuser.latest(), t, fuser.eyes.state())
        sim.step(cmd)
        if t >= next_frame:
            p_ee, _ = sim.ee_pose()
            trail = [(ts, pt) for ts, pt in trail if t - ts <= TRAIL_S] + [(t, p_ee.copy())]
            sim.show_target(cmd)
            mujoco.mj_forward(sim.model, sim.data)
            renderer.update_scene(sim.data, cam)
            _draw_trail(renderer.scene, trail, t)
            frames.append(Image.fromarray(renderer.render()))
            next_frame += 1.0 / args.fps
        t += dt
        clock[0] = t
    renderer.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # One palette for the whole animation, from a strip of frames, so colours don't flicker.
    picks = frames[:: max(1, len(frames) // 6)]
    strip = Image.new("RGB", (width, height * len(picks)))
    for i, f in enumerate(picks):
        strip.paste(f, (0, i * height))
    palette = strip.quantize(colors=224, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(out, save_all=True, append_images=quantized[1:], duration=round(1000 / args.fps), loop=0,
                      optimize=True)
    print(f"{out}: {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB")


def _draw_trail(scene: mujoco.MjvScene, trail: list[tuple[float, np.ndarray]], now: float) -> None:
    """Small spheres along the gripper's recent path, fading with age."""
    for ts, pt in trail[:-1]:
        if scene.ngeom >= scene.maxgeom:
            break
        alpha = 0.35 + 0.65 * (1.0 - (now - ts) / TRAIL_S)
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.0055, 0, 0]),
                            pt, np.eye(3).flatten(), np.array([1.0, 0.55, 0.05, alpha], dtype=np.float32))
        scene.ngeom += 1


if __name__ == "__main__":
    main()
