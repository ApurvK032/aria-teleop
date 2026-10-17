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
    p.add_argument("--rotation-scale", type=float, default=1.0,
                   help="gripper turn per wrist turn (below 1: calmer, less range per engage)")
    p.add_argument("--rotation-deadband", type=float, default=0.0,
                   help="degrees of wrist turn from the engage pose to ignore (soft, no jump)")
    p.add_argument("--pivot-offset", type=float, default=0.0,
                   help="m; turn the gripper about a point this far behind its tip (0.156 = its last wrist joint)")
    p.add_argument("--max-angular-speed", type=float, default=4.0, help="gripper turn-speed cap, rad/s")
    p.add_argument("--yaw-offset", type=float, default=0.0,
                   help="degrees; 180 if you face the robot, or to fix a sideways mapping")
    p.add_argument("--min-confidence", type=float, default=0.3)
    p.add_argument("--pinch-closed", type=float, default=0.02, help="thumb-index distance (m) for closed gripper")
    p.add_argument("--pinch-open", type=float, default=0.08, help="thumb-index distance (m) for open gripper")
    p.add_argument("--max-speed", type=float, default=0.6, help="end-effector speed cap, m/s")
    p.add_argument("--hand-viewer", action="store_true",
                   help="open a Rerun window with the tracked hands, head pose, confidence and pinch")
    p.add_argument("--hand-viewer-memory", default="1GB", help="memory limit for the Rerun hand viewer")

    sim = p.add_argument_group("simulated arm")
    sim.add_argument("--joint-speed-limit", type=_speeds, default=None, metavar="RAD_S",
                     help="warn when a joint moves faster than this: one value, or six comma-separated for "
                          "joint_0..5 (use your arm's velocity_max from TrossenArmDriver.get_joint_limits())")
    sim.add_argument("--cap-joint-speed", action="store_true",
                     help="also hold the IK to --joint-speed-limit, so the sim moves the way the real arm could")
    sim.add_argument("--joint-limit-margin", type=float, default=5.0,
                     help="warn when a joint gets this many degrees from the end of its range")

    arm = p.add_argument_group("arm (joint pipeline shared by the preview and the real arm)")
    arm.add_argument("--arm", choices=["sim", "preview", "dry-run", "real"], default="sim",
                     help="sim: the simulator's own IK (default); preview: the real-arm pipeline driving the "
                          "MuJoCo arm; dry-run: connect to the real arm and compute commands, never move it; "
                          "real: drive the real arm (needs --enable-motion)")
    arm.add_argument("--robot-ip", default="", help="the real arm's IP address (dry-run, real)")
    arm.add_argument("--enable-motion", action="store_true", help="required with --arm real")
    arm.add_argument("--arm-rate", type=float, default=200.0, help="joint control rate, Hz")
    arm.add_argument("--arm-max-velocity", type=_speeds, default=[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], metavar="RAD_S",
                     help="joint speed caps (one value, or six); the real arm's own limits also apply")
    arm.add_argument("--arm-velocity-fraction", type=float, default=0.5,
                     help="use at most this fraction of the velocity_max the real arm reports")

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
    args = p.parse_args(argv)
    if args.cap_joint_speed and args.joint_speed_limit is None:
        p.error("--cap-joint-speed needs --joint-speed-limit")
    if args.arm in ("dry-run", "real") and not args.robot_ip:
        p.error(f"--arm {args.arm} needs --robot-ip")
    if args.arm == "real" and not args.enable_motion:
        p.error("--arm real moves the physical arm: add --enable-motion (try --arm dry-run first)")
    return args


def _speeds(text: str) -> list[float]:
    values = [float(v) for v in text.split(",")]
    if len(values) not in (1, 6) or min(values) <= 0:
        raise argparse.ArgumentTypeError("give one positive speed, or six comma-separated (joint_0..joint_5)")
    return values


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from .sim import HOME_POSITION, HOME_ROTATION, WidowXSim, run  # needs mujoco + trossen_arm_mujoco

    cfg = TeleopConfig(
        hand=args.hand,
        clutch=args.clutch,
        scale=args.scale,
        use_rotation=not args.no_rotation,
        rotation_scale=args.rotation_scale,
        rotation_deadband_deg=args.rotation_deadband,
        pivot_offset_m=args.pivot_offset,
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
    sim = WidowXSim(
        PoseSmoother(max_speed=args.max_speed, max_angular_speed=args.max_angular_speed),
        joint_speed_limit=args.joint_speed_limit,
        cap_joint_speed=args.cap_joint_speed,
    )
    mapper = TeleopMapper(cfg, HOME_POSITION, HOME_ROTATION)
    hand_viewer = None
    if args.hand_viewer:
        from .hand_viewer import HandViewer

        hand_viewer = HandViewer(memory_limit=args.hand_viewer_memory)  # needs rerun-sdk ([aria] extra)
    arm = None
    if args.arm != "sim":
        from .arm import ArmController, ArmKinematics, SimArm, TrossenArm

        backend = SimArm(sim) if args.arm == "preview" else TrossenArm(args.robot_ip, read_only=args.arm == "dry-run")
        speeds = args.arm_max_velocity * (6 if len(args.arm_max_velocity) == 1 else 1)
        arm = ArmController(
            backend,
            ArmKinematics(sim.model),
            PoseSmoother(max_speed=args.max_speed, max_angular_speed=args.max_angular_speed),
            (HOME_POSITION, HOME_ROTATION),
            rate_hz=args.arm_rate,
            max_velocity=speeds,
            velocity_fraction=args.arm_velocity_fraction,
            sim=sim,
        )

    print(__doc__)
    from .arm import ArmFault

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
            joint_limit_margin_deg=args.joint_limit_margin,
            arm=arm,
        )
    except KeyboardInterrupt:
        stats = None
    except ArmFault as e:
        moving = arm is not None and arm.started and not arm.backend.read_only
        print(f"ARM FAULT: {e}." + (" Holding, then returning to rest." if moving else ""))
        stats = None
    finally:
        try:
            if arm is not None:
                arm.stop()  # hold, then move slowly to rest, then disconnect
        finally:
            source.stop()
    if stats and stats.get("tracking_steps"):
        print(
            f"tracking error: mean {stats['mean_error_m'] * 1000:.1f} mm, max {stats['max_error_m'] * 1000:.1f} mm; "
            f"fell behind real time {stats['behind']}x"
        )
    if stats and stats["joint_speed_peak"].any():
        from .sim import JOINT_NAMES

        peaks = ", ".join(f"{name} {v:.2f}" for name, v in zip(JOINT_NAMES, stats["joint_speed_peak"]))
        print(f"peak joint speed, rad/s: {peaks}; warnings: {stats['speed_warnings']} speed, "
              f"{stats['limit_warnings']} near a range end")


if __name__ == "__main__":
    main()
