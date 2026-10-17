# Handoff: first real-arm test on the 5090 workstation

For Claude Code on the lab workstation. Read all of this before running anything. Each
step says whether it moves the arm.

## Rules from the user (follow exactly)

- **Never move the arm without the user's explicit OK in this session**, given right
  before the run, with the user at the arm and the controller's power cut-off in reach.
  Only `--arm real --enable-motion` (and the driver's own motion calls) can move it.
- Don't change the mapping or safety logic (`aria_teleop/mapping.py`) without asking
  first.
- **Don't commit until the user asks. Never push without asking.** Work on branch
  `wrist-calibration`, not `main`.
- Keep RAM use low: no camera decoding. The Rerun viewer (`--hand-viewer`) is capped at
  1 GB and costs about one CPU core.
- Run anything long or interactive in a terminal the user can see and stop with Ctrl+C.
- The user runs anything that needs `sudo` or changes system or network settings (for
  example `aria_doctor`'s fixes, IP configuration). Give them the command.

## What this is and where it stands

`aria-teleop` drives a Trossen WidowX AI arm from Meta Aria Gen 2 glasses: on-device
hand tracking at 30 Hz plus ~800 Hz VIO head pose. Read `README.md` for the
architecture.

Verified live on the user's laptop with the real glasses (in MuJoCo):

- Hands 30 Hz, head pose ~800 Hz (high-frequency VIO), confidence reads ~1.00.
- Eye-gesture clutch `--clutch eyes`: SPACE arms; left wink (≥ 0.6 s) engages; right
  wink releases; both eyes closed ≥ 0.5 s pauses (re-anchors, no jump); ≥ 2 s releases.
  The glasses speak the state changes.
- The user's preferred feel: `--rotation-scale 0.75 --rotation-deadband 5
  --max-angular-speed 2 --pivot-offset 0.156` and `--pinch-open 0.10`.

Built but **never run on hardware**: the real-arm pipeline in `aria_teleop/arm.py`.

```
target pose -> PoseSmoother -> IK (kinematic model) -> JointShaper -> safety gate -> arm
```

- It runs at a fixed 200 Hz, separate from the 30 Hz hand stream.
- `JointShaper` caps each joint's velocity, acceleration and jerk, and brakes before the
  target and the range ends.
- Each command goes to the driver as a position goal 20 ms ahead, with velocity and
  acceleration feedforward.
- `--arm preview | dry-run | real`; the default `sim` is the old path. Details are in the
  README section "Real arm".
- 66 tests pass, including a fake-driver test of the Trossen backend's call order.
- The user found their earlier Quest-3 teleop (a different project) jerky on the real
  arm. Smoothness is the main thing to judge.

## 1. Check the machine (no arm motion)

```bash
lsb_release -ds; uname -r; python3 --version; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

- Python must be 3.10–3.12. If `python3` is 3.13 or newer, use `python3.12` (or 3.11 /
  3.10) for the venv.
- On a desktop with only the 5090 there's no PRIME offload: drop the
  `__NV_PRIME_RENDER_OFFLOAD`/`__GLX_VENDOR_LIBRARY_NAME` variables the laptop needed.
- Put the PC on AC power (if it's a laptop) and set Performance mode. The sim fell
  behind real time on battery.

## 2. Install (no arm motion)

```bash
git clone -b wrist-calibration https://github.com/ApurvK032/aria-teleop.git && cd aria-teleop
```

```bash
python3 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip
```

```bash
git clone https://github.com/TrossenRobotics/trossen_arm_mujoco.git ../trossen_arm_mujoco && pip install -e ../trossen_arm_mujoco
```

```bash
pip install -e ".[aria,test,arm]" && pip install "trossen-arm==1.8.6" && pip check && pytest -q
```

- `trossen_arm_mujoco` **must** be installed with `-e`; a normal install drops its
  `src/` IK module.
- The arm's controller firmware is **1.8.3**, and the driver's major.minor must match it.
  The code refuses to start otherwise. `[arm]` allows any 1.8.x; the second command pins
  **1.8.6**, the version the user's hardware has run before. (`trossen_arm_mujoco` pulls
  in 1.11 first; that's fine to replace, because the sim doesn't import the driver.)
  Check with `pip show trossen-arm | grep Version`.
- Expect `66 passed`.
- MuJoCo GL troubleshooting: `gladLoadGL error` →
  `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`; `GLIBCXX_3.4.31 not found`
  on 22.04 → install `g++-13` from `ppa:ubuntu-toolchain-r/test` (the user runs that).

Mock sanity check (no arm, no glasses):

```bash
python -m aria_teleop --source mock --headless --duration 8 --auto-engage 1
```

```bash
python -m aria_teleop --source mock --headless --duration 8 --auto-engage 1 --arm preview
```

Expect about 5 mm mean tracking error for the first and about 7–9 mm for the preview,
`fell behind real time 0x`, and no `WARNING`/`ARM FAULT` lines.

## 3. Glasses (no arm motion)

Plug the glasses **directly into a port on the machine**, not a hub or dock. On the
laptop, a USB hub port caused repeated disconnects (`error -71` in `journalctl -k`).

Pairing is per computer, so this machine needs it once:

1. The user runs `aria_doctor` in their own terminal. It asks y/n and uses sudo to add
   udev rules, NetworkManager profiles for the glasses' USB network, and firewall accept
   rules. It must end with all checks ok.
2. Run `aria_gen2 auth pair`. The user approves it in the Aria Companion App on their
   phone.
3. Run `aria_gen2 device list`. It prints `<serial> @ <ip>`; use that IP as
   `--device-ip`. The glasses' serial is `1M0YDC6HC70227`. USB discovery is flaky:
   `device list` / `device status` sometimes say "No devices found" even when the link
   is up (check `ip -br addr`, `ip neigh`), so always pass `--device-ip`.

Then a live check in MuJoCo, with the user wearing the glasses:

```bash
python -m aria_teleop --source aria --device-ip <GLASSES_IP> --clutch eyes --pinch-open 0.10 --rotation-scale 0.75 --rotation-deadband 5 --max-angular-speed 2 --pivot-offset 0.156
```

- Connecting takes about 20 s, because the SDK installs a fresh streaming certificate
  each time.
- The status line should show hands ~30 Hz, head ~800 Hz, and eyes ~30 Hz.
- Click the MuJoCo window and press SPACE (you'll see "armed"). Wink left to engage;
  close the window to stop.

## 4. Network to the arm (no arm motion)

- The arm controller is at **192.168.1.2**, with this PC at **192.168.1.10/24** on a
  wired port (the user's setup; confirm with them).
- If the PC isn't on that subnet, give the user the command; don't change it yourself.
  For example:
  `sudo nmcli connection add type ethernet ifname <NIC> con-name trossen-arm ipv4.method manual ipv4.addresses 192.168.1.10/24 ipv4.never-default yes`
  then `sudo nmcli connection up trossen-arm`. A second arm may be at `.3`, so don't
  take that address.
- Check:

```bash
ip -br addr; ping -c 3 -W 1 192.168.1.2; timeout 4 bash -c 'echo > /dev/tcp/192.168.1.2/50001' && echo "controller port open"
```

## 5. Dry run (connects to the arm, **no motion**)

Start with the arm **at rest** (folded, as it powers up). Both the dry run and the
snippet below call the driver's `configure()`, which connects to the controller. The
user's earlier project avoided even that in its no-motion check, so don't run either
while the arm is holding a raised pose.

First, read the arm's own limits. This is read-only and sends no motion command:

```bash
python - <<'EOF'
import trossen_arm as ta
d = ta.TrossenArmDriver()
d.configure(ta.Model.wxai_v0, ta.StandardEndEffector.wxai_v0_follower, "192.168.1.2", False)
print("firmware", d.get_controller_version(), "| positions", [round(x, 3) for x in d.get_all_positions()])
for i, lim in enumerate(d.get_joint_limits()):
    print(f"joint {i}: range [{lim.position_min:.3f}, {lim.position_max:.3f}]  velocity_max {lim.velocity_max:.3f}")
d.cleanup()
EOF
```

Record these `velocity_max` values for the user. The pipeline caps each joint at half of
them (`--arm-velocity-fraction 0.5`).

Then the dry run:

```bash
python -m aria_teleop --source aria --device-ip <GLASSES_IP> --clutch eyes --pinch-open 0.10 --rotation-scale 0.75 --rotation-deadband 5 --max-angular-speed 2 --pivot-offset 0.156 --arm dry-run --robot-ip 192.168.1.2
```

- This never switches joint modes or sends a command. The viewer shows the arm where
  the commands would put it, and commands are computed as if the arm were at home.
- An unreachable arm fails within 3 s:
  `ARM FAULT: can't reach the arm controller at 192.168.1.2:50001`.
- Check this line:
  `arm (dry-run): connecting (no motion)... WidowX AI at 192.168.1.2, firmware 1.8.x, driver 1.8.x; joint speed limits [...] rad/s`.
  The limits shown are min(`--arm-max-velocity`, 0.5 × each joint's `velocity_max`).
- Engage and move a little, then confirm the viewer arm moves smoothly and the status
  line shows sensible `joints .../...rad/s` and `limit jN ..deg` values.
- Close the window to stop. The real arm must not have moved at any point.

## 6. First real run (**moves the arm**: ask the user first)

Before running, confirm all of these with the user:

- The arm is firmly mounted and the workspace is clear.
- The user is at the power cut-off.
- The user knows: on start the arm moves **slowly (3 s) from where it is to the teleop
  home pose**. Nothing else moves until they engage: SPACE in the MuJoCo window, then a
  left wink.
- **Stops:**
  - Closing the window or Ctrl+C holds the arm, then moves it slowly to rest (all
    zeros, folded).
  - A right wink, closed eyes or a lost hand holds the arm, decelerating smoothly.
  - Faults stop teleop (hold, then rest): the arm more than 0.12 rad from its command
    for 50 ms, a command outside a joint's range, or too large a step.

First run with conservative caps:

```bash
python -m aria_teleop --source aria --device-ip <GLASSES_IP> --clutch eyes --pinch-open 0.10 --rotation-scale 0.75 --rotation-deadband 5 --max-angular-speed 2 --pivot-offset 0.156 --arm real --enable-motion --robot-ip 192.168.1.2 --arm-max-velocity 0.5 --scale 0.5
```

Watch for:

- `ARM FAULT: ...`: stop and read it out to the user before retrying.
- `WARNING joints: ...`: a joint near a range end or over `--joint-speed-limit`.
- Smoothness. Ask the user whether the motion is smooth, laggy or jerky. If it's jerky,
  don't raise any limits; capture the terminal output.
- At home the elbow (`joint_2`) sits only ~8° from its range end, so `limit j2 8deg` is
  expected there.

`--arm-max-velocity` takes one value for all six joints (`0.5` caps the wrist at 0.5
too) or six comma-separated values. The default is `1,1,1,2,2,2`.

If it's smooth, raise one thing at a time, with the user's OK each time:

1. `--arm-max-velocity 0.5,0.5,0.5,1,1,1`
2. drop the flag to use the default `1,1,1,2,2,2`
3. `--scale 0.75`
4. `--scale 1.0` (no flag)

`--arm-velocity-fraction` (default 0.5 of the arm's `velocity_max`) always applies on
top.

## Not verified yet (tell the user if any of these look wrong)

- How Trossen's controller interpolates a stream of 20 ms goals with feedforward. The
  sim models it as a cubic to each goal, and real behaviour may differ.
- Gripper mapping: 0 = closed to 0.040 m open (`GRIPPER_OPEN_M` in `arm.py`). Check
  direction and range at low speed.
- Home pose: `HOME_POSITION (0.3, 0, 0.2)` m, gripper level, pointing forward
  (q ≈ [0, 0.585, 0.148, 0.437, 0, 0]). Rest is all zeros.
- Default acceleration/jerk caps: 4/240 for joints 0–2, 8/480 for joints 3–5 (rad/s²,
  rad/s³).

## Known quirks from the laptop

- `python -m aria_teleop` runs the code in the current directory before `PYTHONPATH`.
  Run from the repo you mean.
- In ad-hoc SDK scripts, keep the `DeviceClient` referenced. If it's garbage-collected,
  the device connection closes ("(9) Connection not found" / "Failed to stop session").
  `AriaSource` already does this.
- The eye flags are misnamed: `EyeGaze.vergence.left_blink` / `right_blink` read **1
  when the eye is open**.
- `profile9` also streams RGB+SLAM cameras (~1.5 MB/s). The receiver uses ~150 MB with
  decoding off. There's no camera-free built-in profile.
- If the glasses are left streaming after a crash:
  `aria_gen2 --device-ip <GLASSES_IP> streaming stop`.
