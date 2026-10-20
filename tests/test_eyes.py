"""Eye-gesture clutch: blinks are ignored, winks engage/release, closed eyes pause."""

import numpy as np

from aria_teleop.eyes import EyeTracker
from aria_teleop.fusion import HandFrame, HandFuser, HandState
from aria_teleop.geometry import make_transform
from aria_teleop.mapping import Status, TeleopConfig, TeleopMapper
from aria_teleop.sources import _R_ROOM_DEVICE_LEVEL, MockSource

HOME = np.array([0.3, 0.0, 0.2])
DT = 1 / 30  # Aria eye tracking and hand tracking both run at ~30 Hz
WRIST0 = np.array([0.4, -0.15, -0.4])


def samples(tracker, t0, n, left_open, right_open):
    t = t0
    for _ in range(n):
        t += DT
        tracker.on_sample(int(round(t * 1e9)), left_open, right_open)
    return t


# --- EyeTracker ---------------------------------------------------------------


def test_blink_is_short_and_ends_when_the_eyes_reopen():
    eyes = EyeTracker(clock=lambda: 0.0)
    t = samples(eyes, 0.0, 10, True, True)
    t = samples(eyes, t, 10, False, False)  # 0.33 s blink
    assert 0.25 < eyes.state().both_closed_s < 0.5
    samples(eyes, t, 2, True, True)
    s = eyes.state()
    assert s.left_closed_s == s.right_closed_s == s.both_closed_s == 0.0


def test_wink_closes_one_eye_only():
    eyes = EyeTracker(clock=lambda: 0.0)
    t = samples(eyes, 0.0, 5, True, True)
    samples(eyes, t, 25, False, True)
    s = eyes.state()
    assert s.left_closed_s > 0.75 and s.right_closed_s == 0.0 and s.both_closed_s == 0.0


def test_one_open_sample_inside_a_closure_is_ignored():
    eyes = EyeTracker(clock=lambda: 0.0)
    t = samples(eyes, 0.0, 12, False, True)
    t = samples(eyes, t, 1, True, True)
    samples(eyes, t, 12, False, True)
    assert eyes.state().left_closed_s > 0.75  # one closure of 25 samples (0.8 s), not two of 0.4 s


def test_unknown_and_out_of_order_samples_change_nothing():
    eyes = EyeTracker(clock=lambda: 0.0)
    t = samples(eyes, 0.0, 10, False, True)
    before = eyes.state().left_closed_s
    eyes.on_sample(int(t * 1e9) - 10**8, True, True)  # older than the last sample
    samples(eyes, t, 3, None, None)
    assert eyes.state().left_closed_s >= before
    assert eyes.samples == 13


# --- TeleopMapper with clutch="eyes" -------------------------------------------


def frame(t, wrist):
    lm = np.tile(wrist, (21, 1))
    lm[1] = lm[0] + [0.0, 0.05, 0.0]
    return HandFrame(
        t_ns=int(round(t * 1e9)),
        received_at=t,
        hands={"right": HandState(make_transform(np.eye(3), wrist), lm, 0.9)},
        T_room_head=make_transform(_R_ROOM_DEVICE_LEVEL, [0, 0, 0]),
        gravity_room=np.array([0.0, 0.0, -9.81]),
        session=0,
    )


class Rig:
    """A mapper with the eyes clutch, fed hand frames and eye samples at 30 Hz."""

    def __init__(self, armed=True, **cfg):
        self.t = 0.0
        self.eyes = EyeTracker(clock=lambda: self.t)
        self.m = TeleopMapper(TeleopConfig(clutch="eyes", **cfg), HOME, np.eye(3))
        if armed:
            self.m.toggle_clutch()  # SPACE
        self.wrist = WRIST0.copy()
        self.statuses = []

    def run(self, n, left_open=True, right_open=True, dx=0.0, eye_data=True):
        for _ in range(n):
            self.t += DT
            self.wrist = self.wrist + [dx, 0.0, 0.0]
            if eye_data:
                self.eyes.on_sample(int(round(self.t * 1e9)), left_open, right_open)
            self.cmd = self.m.update(frame(self.t, self.wrist), self.t, self.eyes.state())
            self.statuses.append(self.cmd.status)
        return self.cmd

    def wink(self, side):
        self.run(24, left_open=side != "left", right_open=side != "right")  # 0.8 s
        self.run(3)

    def engaged(self):
        self.run(5)
        self.wink("left")
        assert self.m.engaged
        self.run(15)  # settle
        return self


def test_left_wink_engages_only_when_armed():
    rig = Rig(armed=False)
    rig.run(5)
    rig.wink("left")
    assert not rig.m.engaged and rig.cmd.status is Status.IDLE
    rig.m.toggle_clutch()  # SPACE arms
    rig.wink("left")
    assert rig.m.engaged and rig.cmd.status is Status.TRACKING
    assert np.allclose(rig.cmd.position, HOME, atol=1e-3)  # engaging doesn't move the robot


def test_short_wink_does_nothing_and_right_wink_releases():
    rig = Rig()
    rig.run(5)
    rig.run(12, left_open=False)  # 0.4 s: shorter than wink_s
    rig.run(3)
    assert not rig.m.engaged
    rig.wink("left")
    rig.wink("right")
    assert not rig.m.engaged and rig.m.armed and rig.cmd.status is Status.IDLE


def test_mock_blinks_never_pause_the_robot():
    rig = Rig().engaged()
    mock = MockSource(HandFuser())
    start = len(rig.statuses)
    for _ in range(300):  # 10 s, blinking every 3.7 s for 0.3 s
        is_open = mock.eyes_open(rig.t + DT)
        rig.run(1, left_open=is_open, right_open=is_open, dx=0.0005)
    assert Status.PAUSED not in rig.statuses[start:]
    assert rig.cmd.position[0] > HOME[0] + 0.1  # kept following the hand


def test_closed_eyes_pause_and_resume_without_a_jump():
    rig = Rig().engaged()
    rig.run(14, left_open=False, right_open=False)  # < 0.5 s: still a blink
    assert rig.cmd.status is Status.TRACKING
    rig.run(4, left_open=False, right_open=False)
    assert rig.cmd.status is Status.PAUSED
    held = rig.cmd.position.copy()
    rig.run(20, left_open=False, right_open=False, dx=0.005)  # hand moves 10 cm, unseen
    assert np.allclose(rig.cmd.position, held)
    rig.run(10)  # eyes open: re-anchor where the robot is
    assert rig.cmd.status is Status.TRACKING
    assert np.allclose(rig.cmd.position, held, atol=2e-3)
    rig.run(20, dx=0.001)  # then follow new motion
    assert rig.cmd.position[0] > held[0] + 0.01


def test_long_closure_releases_and_opening_one_eye_first_is_not_a_wink():
    rig = Rig().engaged()
    rig.run(66, left_open=False, right_open=False)  # 2.2 s
    assert not rig.m.engaged and rig.m.armed
    rig.run(30, left_open=False, right_open=True)  # right opens first; left still closed 1 s
    assert not rig.m.engaged
    rig.run(3)
    rig.wink("left")  # a real wink still works afterwards
    assert rig.m.engaged


def test_lost_eye_tracking_pauses_until_it_returns():
    rig = Rig().engaged()
    rig.run(20, eye_data=False)  # 0.67 s without eye samples
    assert rig.cmd.status is Status.PAUSED
    rig.run(3)
    assert rig.cmd.status is Status.TRACKING


def test_disarming_releases_and_ignores_winks():
    rig = Rig().engaged()
    rig.m.toggle_clutch()  # SPACE disarms
    rig.run(1)
    assert not rig.m.engaged and not rig.m.armed and rig.cmd.status is Status.IDLE
    rig.wink("left")
    assert not rig.m.engaged
