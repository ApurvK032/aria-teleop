"""Bring the WidowX AI slowly back to its folded rest pose (all joints at zero), then disconnect.

    python scripts/return_to_rest.py --robot-ip 192.168.1.3

For recovering after an interrupted run. If the arm already reads within 0.05 rad
of rest it doesn't move. This moves the arm: stay at the power cut-off.
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--robot-ip", required=True)
    p.add_argument("--seconds", type=float, default=4.0, help="duration of the move (slower than teleop's 3 s)")
    args = p.parse_args()

    import trossen_arm as ta  # the driver's major.minor must match the arm's firmware

    d = ta.TrossenArmDriver()
    d.configure(ta.Model.wxai_v0, ta.StandardEndEffector.wxai_v0_follower, args.robot_ip, False, 5.0)
    try:
        q = [round(x, 3) for x in d.get_arm_positions()]
        print("arm positions now:", q)
        if max(abs(x) for x in q) < 0.05:
            print("already at rest; not moving")
            return
        print(f"moving to rest over {args.seconds:.0f} s...")
        d.set_arm_modes(ta.Mode.position)
        d.set_arm_positions([0.0] * 6, args.seconds, True)
        print("at rest:", [round(x, 3) for x in d.get_arm_positions()])
    finally:
        d.cleanup()


if __name__ == "__main__":
    main()
