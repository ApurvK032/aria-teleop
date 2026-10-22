# Real-arm setup

Step-by-step bring-up of the glasses and a Trossen WidowX AI on a new Linux workstation, from a fresh
install to the first teleoperated motion. Each step says whether it can move the arm.

> **Safety.** Only `--arm real --enable-motion` (and `scripts/return_to_rest.py`) send motion
> commands. Before any of them: the arm is firmly mounted, the workspace is clear, and you are at the
> controller's power cut-off.

## 1. Workstation (no arm motion)

- Ubuntu 22.04 or 24.04, Python 3.10–3.12.
- Plug the glasses **straight into a USB port on the PC**, not a hub or dock. Hub ports caused repeated
  disconnects (`error -71` in `journalctl -k`), and one front-panel port dropped the glasses mid-run. If the
  stream freezes (hands, head and eyes all at 0 Hz), check `journalctl -k | grep -i usb` and try another port.
- Install as in the [README](../README.md#installation), including the `[arm]` extra, then pin the driver
  to the arm's firmware:

  ```bash
  pip install "trossen-arm==1.8.6"     # driver major.minor must match the controller firmware (1.8.x here)
  pip show trossen-arm | grep Version
  pytest -q
  ```

  `trossen_arm_mujoco` pulls in a newer driver; replacing it is fine, the simulator doesn't use it.

## 2. Glasses (no arm motion)

Pairing is per computer:

```bash
aria_doctor               # asks y/n, uses sudo: udev rules, NetworkManager profiles, firewall ports
aria_gen2 auth pair       # approve in the Aria companion app
aria_gen2 device list     # prints <serial> @ <ip>; use that IP as --device-ip
```

`aria_doctor`'s changes are machine-wide. Its NetworkManager profiles only match the glasses' own USB
interface, and its firewall rules last until reboot. If the glasses were plugged in before the udev rule
existed, replug them so their network interface is renamed and picks up the profile. USB discovery is
flaky, so always pass `--device-ip`; `aria_gen2 auth check` confirms the pairing.

Live check in MuJoCo, wearing the glasses:

```bash
python -m aria_teleop --source aria --device-ip <GLASSES_IP> --clutch eyes
```

Expect hands ~30 Hz, head ~800 Hz and eyes ~30 Hz in the status line. If winks don't register, run
`python scripts/eye_monitor.py --device-ip <GLASSES_IP>`: it prints what the glasses report for each eye
and why each wink did or didn't count.

## 3. Network to the arm (no arm motion)

The controller has a fixed address on its own subnet (192.168.1.x) and no DHCP server, so give the PC's
wired port a fixed address on that subnet (pick one no other device uses):

```bash
sudo nmcli connection add type ethernet ifname <NIC> con-name trossen-arm \
    ipv4.method manual ipv4.addresses 192.168.1.10/24 ipv4.never-default yes \
    ipv4.dad-timeout 3000 connection.autoconnect-priority 10
sudo nmcli connection up trossen-arm
```

`ipv4.dad-timeout` refuses the address if another device already has it; `never-default` keeps your
internet on its usual interface. If you don't know the controller's address, find what answers on the
link (a one-off ping sweep; nothing here moves the arm):

```bash
for i in $(seq 1 254); do ping -c1 -W1 -I <NIC> 192.168.1.$i >/dev/null && echo "192.168.1.$i" & done; wait
timeout 4 bash -c 'echo > /dev/tcp/<ROBOT_IP>/50001' && echo "controller port open"
```

## 4. Read the arm's limits (connects, no motion)

With the arm folded at rest, read its firmware, position and per-joint limits:

```bash
python - <<'EOF'
import trossen_arm as ta
d = ta.TrossenArmDriver()
d.configure(ta.Model.wxai_v0, ta.StandardEndEffector.wxai_v0_follower, "<ROBOT_IP>", False, 5.0)
try:
    print("firmware", d.get_controller_version(), "| positions", [round(x, 3) for x in d.get_all_positions()])
    for i, lim in enumerate(d.get_joint_limits()):
        print(f"joint {i}: [{lim.position_min:.3f}, {lim.position_max:.3f}] rad, velocity_max {lim.velocity_max:.3f}")
finally:
    d.cleanup()
EOF
```

The pipeline caps every joint at `--arm-velocity-fraction` (0.5) of its `velocity_max` and at
`--arm-max-velocity`, whichever is lower. Joints 1 and 2 reading a few mrad below zero at rest is normal.

## 5. Dry run (connects, no motion)

```bash
python -m aria_teleop --source aria --device-ip <GLASSES_IP> --clutch eyes --headless \
    --arm dry-run --robot-ip <ROBOT_IP> --arm-max-velocity 0.5 --scale 0.5
```

It never switches the joint modes or sends a command: commands are computed as if the arm were at home.
Check the start line (`WidowX AI at <ROBOT_IP>, firmware 1.8.x, driver 1.8.x; joint speed limits [...]`),
then press SPACE in the terminal, wink left, move a little, wink right. The arm must stay still throughout.

## 6. First real run (moves the arm)

```bash
python -m aria_teleop --source aria --device-ip <GLASSES_IP> --clutch eyes --headless \
    --arm real --enable-motion --robot-ip <ROBOT_IP> --arm-max-velocity 0.5 --scale 0.5 --duration 90
```

- On start the arm moves slowly (3 s) from rest to the home pose. Nothing else moves until you press SPACE
  in the terminal and wink left.
- **Stopping:** press **Ctrl+C once**. The arm holds, then returns slowly to rest. A second Ctrl+C can cut
  that return short. A right wink, closed eyes or a lost hand hold the arm; a fault (tracking error,
  out-of-range command, oversized step) holds it and returns it to rest; `--duration` does the same.
- If a run ends with the arm away from rest, `python scripts/return_to_rest.py --robot-ip <ROBOT_IP>`.

Raise one limit at a time, and only while the motion stays smooth:

1. `--arm-max-velocity 0.5,0.5,0.5,1,1,1` (wrist joints first: they hit the cap most)
2. drop the flag for the default `1,1,1,2,2,2`
3. `--scale 0.75`
4. `--scale 1.0` (drop the flag)

On controller firmware 1.8.x, streaming short position goals can become jerky as speeds rise. If it does,
step back to the last smooth level and try a lower `--arm-rate` before raising anything else.

## Known quirks

- `EyeGaze.vergence.left_blink` / `right_blink` read **1 while the eye is open**.
- The `profile9` streaming profile also sends the RGB and SLAM cameras; the receiver decodes none of them
  during teleop (~150 MB of memory).
- After a crash the glasses may keep streaming: `aria_gen2 --device-ip <GLASSES_IP> streaming stop`.
- In ad-hoc SDK scripts keep the `DeviceClient` referenced; if it is garbage-collected the device
  connection closes. `AriaSource` already does this.
