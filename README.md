# aria-teleop

Drive a Trossen WidowX AI arm in MuJoCo with the hand tracking that runs on Aria Gen 2 glasses. Point your hand and the gripper follows; pinch your thumb and index finger and it closes.

```
Aria Gen 2 ──USB──▶ HandFuser ──▶ TeleopMapper ──▶ PoseSmoother ──▶ IK ──▶ WidowX AI (MuJoCo)
 hands 30 Hz         head-motion     gate, filter,     30 Hz → 500 Hz,
 head pose           compensation    clutch, mapping   speed cap
```

- **fusion.py**: converts hand poses from the glasses' frame into the room frame, using the on-device VIO head pose at the hand's timestamp. Turning your head doesn't move the robot.
- **mapping.py**: drops low-confidence frames and single-frame glitches, then applies One Euro filtering. Motion is relative and clutched: when you engage, the robot stays put, and after that it copies how your hand moves. Your thumb–index distance sets the gripper opening.
- **sim.py**: runs Trossen's follow-target scene and IK controller. The cube shows the target. It's grey when idle, green when tracking, and orange when holding because the hand was lost.
- **sources.py**: `AriaSource` streams live from the glasses through the Client SDK. `MockSource` is a scripted hand seen from a swaying head, so you can test without the glasses.

## Install (Ubuntu 22.04 / 24.04, Python 3.10–3.12)

```bash
python3 -m venv .venv && source .venv/bin/activate
git clone https://github.com/TrossenRobotics/trossen_arm_mujoco.git
pip install -e trossen_arm_mujoco
pip install -e ".[aria,test]"
```

Install `trossen_arm_mujoco` with `-e`. A regular install leaves out its `src/` module, which holds the IK controller.

One-time setup for the glasses (approve the pairing in the Companion App):

```bash
aria_doctor
aria_gen2 auth pair
aria_gen2 device list
```

## Run

Start with the mock, so you can check the sim on its own:

```bash
python -m aria_teleop --source mock
```

Then with the glasses plugged in over USB:

```bash
python -m aria_teleop --source aria
```

| Key in the viewer | Action |
|---|---|
| `SPACE` | engage / release the clutch |
| `H` | reset heading: "forward" is taken from where you face at the next engage |

Before engaging, stand behind the arm facing the way it points. If you face the robot instead, add `--yaw-offset 180`. Keep your hand where the glasses can see it.

`--clutch pinch` turns your other hand into a deadman switch: pinch it to drive, open it to stop.

`--clutch eyes` drives the clutch with the glasses' on-device eye tracking:

| Gesture | Action |
|---|---|
| `SPACE` in the viewer | arm / disarm. Disarmed, the robot never moves |
| wink your left eye (hold ~0.6 s, right eye open) | engage |
| wink your right eye | release |
| close both eyes for longer than a blink (0.5 s) | pause at once: the robot holds; when you open your eyes it re-anchors, so hand motion you made without looking is ignored |
| keep both eyes closed for 2 s | release |

Losing eye-tracking data also pauses. The glasses say "engaged", "paused", "released" and so on (`--no-speak` turns this off), and the cube turns blue while paused. Tune the wink time with `--wink-s`. Ordinary blinks were measured at up to ~0.4 s, so they don't pause the robot. A wink only counts if the other eye stays open; if both eyes close together for 0.5 s, the gesture is treated as closed eyes, not a wink.

A status line prints once a second:

```
[  12.0s] tracking clutch=on  hands 30.0Hz head   800Hz conf 0.93 pinch  4.5cm target (+0.320,-0.050,+0.210) err   4.1mm rejected 0 anchors 1
```

`--hand-viewer` opens a Rerun window with both tracked hands (after head-motion compensation, so turning your head shouldn't move them), the head pose, and live plots of confidence and pinch distance. You can read your pinch values there without engaging. The Rerun viewer takes about one CPU core, so on a slow laptop use it to check tracking and leave it off for teleop.

## First live run: things to check

The live Aria path has been tested against the real SDK types but not against real glasses yet. Watch the status line for these:

1. **`hands` around 30 Hz.** The default `--profile profile9` streams hand tracking at 30 Hz.
2. **`head` rate.** About 800 Hz means the high-frequency VIO pose is streaming. About 10 Hz means the profile doesn't include it and the code fell back to the 10 Hz VIO output. That still works, but head-motion compensation is coarser.
3. **`conf`.** The on-device confidence scale isn't documented. If frames are being dropped, lower `--min-confidence`.
4. **Direction.** If pushing your hand forward moves the arm sideways, set `--yaw-offset` (degrees, positive turns left), or press `H` and re-engage while facing the arm.
5. **`pinch`.** Read your own fully-closed and fully-open values, then pass them as `--pinch-closed` and `--pinch-open`.

## Useful flags

| Flag | Default | Meaning |
|---|---|---|
| `--hand` | `right` | hand that drives the arm |
| `--scale` | `1.0` | robot metres per hand metre (below 1 for precision) |
| `--no-rotation` | off | follow position only, keep the gripper orientation fixed |
| `--rotation-scale` | `1.0` | gripper turn per wrist turn; below 1 is calmer, with less range per engage (re-engage to turn further) |
| `--rotation-deadband` | `0` | degrees of wrist turn from the engage pose to ignore, so the wrist's own small twists while you move don't turn the gripper; it's soft, so nothing jumps past it |
| `--pivot-offset` | `0` | turn the gripper about a point this many metres behind its tip, the way your hand turns about your wrist; `0.156` is the arm's last wrist joint, `0` turns it about the fingertips |
| `--max-speed` | `0.6` | end-effector speed cap, m/s |
| `--max-angular-speed` | `4.0` | gripper turn-speed cap, rad/s |
| `--joint-speed-limit` | none | sim: warn when a joint moves faster than this, rad/s (one value, or six comma-separated for joint_0..5) |
| `--cap-joint-speed` | off | sim: also hold the IK to `--joint-speed-limit` |
| `--joint-limit-margin` | `5` | sim: warn when a joint gets this many degrees from the end of its range |
| `--interface` | `usb` | `wifi_sta` / `wifi_sap` add latency; use USB for teleop |
| `--no-start-streaming` | off | only run the receiver, if you started streaming with `aria_gen2 streaming start` |
| `--hand-viewer` | off | Rerun window with the tracked hands; `--hand-viewer-memory` caps its memory (default `1GB`) |
| `--clutch eyes --wink-s 0.6` | | eye-gesture clutch, and how long a wink must be held |
| `--headless --duration 10 --auto-engage 1` | | run without the viewer, for testing |

The rest (filter cutoffs, glitch threshold, workspace box, reach limit) are in `TeleopConfig` in `aria_teleop/mapping.py`.

If the gripper turns too much or when you don't mean it, a starting point measured on one operator is `--rotation-scale 0.5 --rotation-deadband 5 --max-angular-speed 2`. Their wrist twisted by itself about 6–9 degrees while they moved their hand, and held-still jitter was about 0.5 degrees. Try `--pivot-offset 0.156` against `0` to see which feels more like your own hand.

### Simulated vs. real arm

The simulator lets joints move as fast as the IK asks; the real arm has per-joint speed limits and faults with `joint_limit_exceeded`. To see trouble before it reaches hardware, watch `joints 0.21/0.74rad/s limit j3 23deg` in the status line. It shows the peak shoulder/elbow and wrist joint speeds over the last second, and the joint closest to the end of its range. While you're driving the arm, a `WARNING joints:` line appears when a joint passes `--joint-speed-limit` or comes within `--joint-limit-margin` of its range end. The run ends with each joint's peak speed. Read the real limits from the arm with `TrossenArmDriver.get_joint_limits()` (`velocity_max`), then pass them as `--joint-speed-limit`. Add `--cap-joint-speed` to make the sim move only as fast as the real arm could; the arm then lags fast hand motion instead of whipping. With `--pivot-offset 0` (turning about the fingertips), wrist turns swing the shoulder and elbow several times harder than with `0.156`.

## Real arm

`--arm` chooses where joint commands come from and go to:

| `--arm` | What happens |
|---|---|
| `sim` (default) | The simulator's own IK drives the MuJoCo arm, as before |
| `preview` | The real-arm pipeline drives the MuJoCo arm: use it to see what the real arm will be sent |
| `dry-run` | Connects to the real arm, reads its state and limits, and computes every command, but never switches the joints to position mode or sends anything; the viewer shows the commands |
| `real` | Drives the real arm. Needs `--enable-motion` |

The pipeline (`aria_teleop/arm.py`) runs at a fixed 200 Hz, separate from the 30 Hz hand stream, so input timing never reaches the motors:

```
target pose -> PoseSmoother -> IK (on a kinematic model) -> JointShaper -> safety gate -> arm
```

`JointShaper` follows the IK solution like a critically damped spring that matches the target's velocity. It caps every joint's velocity, acceleration and jerk, and never moves a joint faster than it can brake from before the target or the end of its range. On the mock it is 3–5x smoother than the simulator's own IK (RMS jerk), with about 2 mm more tracking error. Each command goes to the driver as a position goal 20 ms ahead, with velocity and acceleration feedforward.

Safety:

- the driver's major.minor must match the arm's firmware (`pip install -e ".[arm]"` installs a 1.8.x driver), or nothing starts;
- an unreachable arm fails within 3 s with a message, instead of hanging;
- start-up is a slow (3 s) blocking move from wherever the arm is to the teleop home pose; exit, Ctrl+C or any fault holds the arm and then moves it slowly to rest (all zeros);
- a gate independent of the shaper rejects any command that is non-finite, outside a joint's range, or a larger step than the speed caps allow;
- if the arm stays more than 0.12 rad from its command for 50 ms, teleop stops (hold, then rest);
- joint speeds are capped at `--arm-max-velocity` (default 1.0 rad/s for the shoulder/elbow joints, 2.0 for the wrist) and at `--arm-velocity-fraction` (0.5) of the `velocity_max` each joint reports.

Clutch release, closed eyes and a lost hand all hold the target, and the shaper brings the joints to a smooth stop. First session with the arm, power cut-off within reach:

```bash
python -m aria_teleop --source aria --clutch eyes --arm preview      # check the motion in MuJoCo
python -m aria_teleop --source aria --clutch eyes --arm dry-run --robot-ip 192.168.1.2
python -m aria_teleop --source aria --clutch eyes --arm real --enable-motion --robot-ip 192.168.1.2
```

## Tests

```bash
pytest
```

These cover the geometry and filters, the head-motion compensation, the mapper (clutch, heading, glitches, tracking loss, workspace, pinch), and the full mock-to-MuJoCo loop. The sim tests are skipped when MuJoCo or `trossen_arm_mujoco` isn't installed.

## Troubleshooting

- **`mujoco.FatalError: gladLoadGL error`**: run `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` (Trossen's documented fix).
- **`GLIBCXX_3.4.31 not found` on Ubuntu 22.04**: install `g++-13` from the `ubuntu-toolchain-r/test` PPA.
- **`No devices found`**: check the glasses are on and plugged in, then run `aria_gen2 device list` and `aria_doctor`.
