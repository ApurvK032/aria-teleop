"""Show what the glasses report for each eye, to see why a wink does or doesn't count.

    python scripts/eye_monitor.py --device-ip <GLASSES_IP> [--seconds 40]

Prints the per-eye open/closed flags while an eye is closed, and a verdict after
each left-eye closure, using the same thresholds as ``TeleopConfig``. No arm, no viewer.
"""

from __future__ import annotations

import argparse
import time

from aria_teleop.fusion import HandFuser
from aria_teleop.mapping import TeleopConfig
from aria_teleop.sources import AriaSource


def flag(v: bool | None) -> str:
    return "  ?   " if v is None else " open " if v else "CLOSED"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device-ip", default="")
    p.add_argument("--seconds", type=float, default=40.0)
    args = p.parse_args()
    cfg = TeleopConfig()
    wink_s, pause_s = cfg.wink_s, cfg.eyes_pause_s

    fuser = HandFuser()
    src = AriaSource(fuser, device_ip=args.device_ip, eyes=True)
    raw: dict[str, bool | None] = {}
    on_sample = fuser.eyes.on_sample

    def tap(t_ns, left_open, right_open):
        raw["L"], raw["R"] = left_open, right_open
        on_sample(t_ns, left_open, right_open)

    fuser.eyes.on_sample = tap
    src.start()
    print(f"watching eyes for {args.seconds:.0f} s: wink your LEFT eye (hold ~1 s, right eye open), a few times\n")
    t0 = time.monotonic()
    last_beat = 0.0
    closure = None  # the current left-eye closure
    try:
        while time.monotonic() - t0 < args.seconds:
            time.sleep(0.1)
            s = fuser.eyes.state()
            if s is None:
                continue
            t = time.monotonic() - t0
            if s.left_closed_s > 0 or s.right_closed_s > 0:
                if closure is None and s.left_closed_s > 0:
                    closure = {"max_l": 0.0, "max_both": 0.0, "fired": False, "cancelled": False}
                if closure is not None:
                    closure["max_l"] = max(closure["max_l"], s.left_closed_s)
                    closure["max_both"] = max(closure["max_both"], s.both_closed_s)
                    closure["cancelled"] |= s.both_closed_s >= pause_s
                    if s.left_closed_s >= wink_s and s.right_closed_s == 0.0 and not closure["cancelled"]:
                        closure["fired"] = True
                print(f"[{t:5.1f}s] L {flag(raw.get('L'))} R {flag(raw.get('R'))}  "
                      f"closed L {s.left_closed_s:.2f}  R {s.right_closed_s:.2f}  both {s.both_closed_s:.2f}")
                last_beat = t
            else:
                if closure is not None:
                    c = closure
                    if c["fired"]:
                        verdict = "COUNTS as a left wink (would engage)"
                    elif c["cancelled"]:
                        verdict = f"cancelled: both eyes read closed together {c['max_both']:.2f} s (>= {pause_s})"
                    elif c["max_l"] < wink_s:
                        verdict = f"too short: left eye read closed only {c['max_l']:.2f} s (needs {wink_s})"
                    else:
                        verdict = f"the right eye read closed whenever the left passed {wink_s} s"
                    print(f"  -> left closure {c['max_l']:.2f} s: {verdict}\n")
                    closure = None
                if t - last_beat >= 1.0:
                    print(f"[{t:5.1f}s] both open   (eye samples so far: {fuser.eyes.samples})")
                    last_beat = t
    except KeyboardInterrupt:
        pass
    finally:
        src.stop()


if __name__ == "__main__":
    main()
