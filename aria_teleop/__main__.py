"""Aria Gen 2 hand tracking -> WidowX AI in MuJoCo.

    python -m aria_teleop --source mock     # no glasses: scripted hand motion
    python -m aria_teleop --source aria     # live glasses over USB

In the viewer, press SPACE to engage / release the clutch (or use --clutch pinch).
With --clutch eyes, SPACE arms; then wink your left eye to engage and your right
eye to release. Closing both eyes pauses the robot.
"""

from __future__ import annotations

import argparse

from .fusion import HandFuser
from .mapping import PoseSmoother, TeleopConfig, TeleopMapper
from .sources import AriaSource, MockSource


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m aria_teleop", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["mock", "aria"], default="mock")
    p.add_argument("--hand", choices=["right", "left"], default="right", help="hand that drives the arm")
    p.add_argument("--clutch", choices=["key", "pinch", "eyes"], default="key",
                   help="key: SPACE in the viewer toggles; pinch: pinch the other hand to engage (deadman); "
                        "eyes: SPACE arms, left wink engages, right wink releases, both eyes closed pauses")
    p.add_argument("--clutch-key", default=" ", help="viewer key that toggles (or, with eyes, arms) the clutch")
    p.add_argument("--wink-s", type=float, default=0.6, help="eyes clutch: how long a wink must be held, s")
    p.add_argument("--no-speak", action="store_true", help="eyes clutch: don't announce changes on the glasses")
    p.add_argument("--scale", type=float, default=1.0, help="robot metres per hand metre")
    p.add_argument("--no-rotation", action="store_true", help="follow position only, keep gripper orientation")
    p.add_argument("--yaw-offset", type=float, default=0.0,
                   help="degrees; 180 if you face the robot, or to fix a sideways mapping")
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--pinch-closed", type=float, default=0.02, help="thumb-index distance (m) for closed gripper")
    p.add_argument("--pinch-open", type=float, default=0.08, help="thumb-index distance (m) for open gripper")
    p.add_argument("--max-speed", type=float, default=0.6, help="end-effector speed cap, m/s")
    p.add_argument("--hand-viewer", action="store_true",
                   help="open a Rerun window with the tracked hands, head pose, confidence and pinch")
    p.add_argument("--hand-viewer-memory", default="1GB", help="memory limit for the Rerun hand viewer")

    aria = p.add_argument_group("aria source")
    aria.add_argument("--profile", default="profile9", help="streaming profile (profile9 = hands at 30 Hz)")
    aria.add_argument("--profile-json", default="", help="custom streaming profile file; overrides --profile")
    aria.add_argument("--interface", choices=AriaSource.INTERFACES, default="usb")
    aria.add_argument("--device-ip", default="", help="connect to the glasses by IP instead of USB")
    aria.add_argument("--no-start-streaming", action="store_true",
                      help="only run the receiver (streaming already started with aria_gen2)")

    test = p.add_argument_group("testing")
    test.add_argument("--headless", action="store_true", help="no viewer")
    test.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    test.add_argument("--auto-engage", type=float, default=None, metavar="SECONDS",
                      help="engage the clutch automatically after this many seconds")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from .sim import HOME_POSITION, HOME_ROTATION, WidowXSim, run  # needs mujoco + trossen_arm_mujoco

    cfg = TeleopConfig(
        hand=args.hand,
        clutch=args.clutch,
        scale=args.scale,
        use_rotation=not args.no_rotation,
        yaw_offset_deg=args.yaw_offset,
        min_confidence=args.min_confidence,
        pinch_closed_m=args.pinch_closed,
        pinch_open_m=args.pinch_open,
        wink_s=args.wink_s,
    )
    fuser = HandFuser()
    if args.source == "mock":
        source = MockSource(fuser, hand=args.hand, clutch_pinch=args.clutch == "pinch")
    else:
        source = AriaSource(
            fuser,
            profile=args.profile,
            profile_json=args.profile_json,
            interface=args.interface,
            device_ip=args.device_ip,
            start_streaming=not args.no_start_streaming,
            eyes=args.clutch == "eyes",
        )
    sim = WidowXSim(PoseSmoother(max_speed=args.max_speed))
    mapper = TeleopMapper(cfg, HOME_POSITION, HOME_ROTATION)
    hand_viewer = None
    if args.hand_viewer:
        from .hand_viewer import HandViewer

        hand_viewer = HandViewer(memory_limit=args.hand_viewer_memory)  # needs rerun-sdk ([aria] extra)

    print(__doc__)
    try:
        try:
            source.start()
        except RuntimeError as e:
            if args.source != "aria":
                raise
            raise SystemExit(
                f"Aria: {e}\nCheck the glasses are on and plugged in, then run `aria_gen2 device list` and `aria_doctor`."
            ) from e
        stats = run(
            sim,
            mapper,
            fuser,
            headless=args.headless,
            duration=args.duration,
            auto_engage_after=args.auto_engage,
            clutch_key=ord(args.clutch_key.upper()[0]),
            hand_viewer=hand_viewer,
            announce=source.say if args.clutch == "eyes" and not args.no_speak else None,
        )
    except KeyboardInterrupt:
        stats = None
    finally:
        source.stop()
    if stats and stats.get("tracking_steps"):
        print(
            f"tracking error: mean {stats['mean_error_m'] * 1000:.1f} mm, max {stats['max_error_m'] * 1000:.1f} mm; "
            f"fell behind real time {stats['behind']}x"
        )


if __name__ == "__main__":
    main()
