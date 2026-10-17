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

A status line prints once a second:

```
[  12.0s] tracking clutch=on  hands 30.0Hz head   800Hz conf 0.93 pinch  4.5cm target (+0.320,-0.050,+0.210) err   4.1mm rejected 0 anchors 1
```

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
| `--max-speed` | `0.6` | end-effector speed cap, m/s |
| `--interface` | `usb` | `wifi_sta` / `wifi_sap` add latency; use USB for teleop |
| `--no-start-streaming` | off | only run the receiver, if you started streaming with `aria_gen2 streaming start` |
| `--headless --duration 10 --auto-engage 1` | | run without the viewer, for testing |

The rest (filter cutoffs, glitch threshold, workspace box, reach limit) are in `TeleopConfig` in `aria_teleop/mapping.py`.

## Tests

```bash
pytest
```

These cover the geometry and filters, the head-motion compensation, the mapper (clutch, heading, glitches, tracking loss, workspace, pinch), and the full mock-to-MuJoCo loop. The sim tests are skipped when MuJoCo or `trossen_arm_mujoco` isn't installed.

## Troubleshooting

- **`mujoco.FatalError: gladLoadGL error`**: run `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` (Trossen's documented fix).
- **`GLIBCXX_3.4.31 not found` on Ubuntu 22.04**: install `g++-13` from the `ubuntu-toolchain-r/test` PPA.
- **`No devices found`**: check the glasses are on and plugged in, then run `aria_gen2 device list` and `aria_doctor`.
