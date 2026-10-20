"""The Rerun hand viewer logs mock frames without a window (catches Rerun API drift)."""

import pytest

pytest.importorskip("rerun")

from aria_teleop.fusion import HandFuser  # noqa: E402
from aria_teleop.hand_viewer import BONES, HandViewer  # noqa: E402
from aria_teleop.sources import NUM_LANDMARKS, MockSource  # noqa: E402


def mock_frames(n, **mock_kw):
    clock = [0.0]
    fuser = HandFuser(clock=lambda: clock[0])
    mock = MockSource(fuser, dropout=0.0, glitch=0.0, **mock_kw)
    for i in range(n):
        t = i / 30
        clock[0] = t
        fuser.on_head_pose(int(t * 1e9), mock.head_pose(t), session=0)
        fuser.on_hands(int(t * 1e9), mock.observe(t))
        yield fuser.latest(), t


def test_bones_index_aria_landmarks():
    assert len(BONES) == 23
    assert all(0 <= i < NUM_LANDMARKS for bone in BONES for i in bone)


def test_logs_both_hands_and_skips_repeats():
    viewer = HandViewer(spawn=False, rate_hz=1000.0)
    viewer.rr.memory_recording()
    logged = []
    for frame, t in mock_frames(10, clutch_pinch=True):  # clutch_pinch adds the left hand
        assert set(frame.hands) == {"left", "right"}
        viewer.log(frame, t)
        logged.append(viewer._last_t_ns)
        viewer.log(frame, t + 0.5)  # same frame again: ignored
        assert viewer._last_t_ns == logged[-1]
    assert len(set(logged)) == 10


def test_rate_limit_and_missing_hand():
    viewer = HandViewer(spawn=False, rate_hz=10.0)
    viewer.rr.memory_recording()
    seen = set()
    for frame, t in mock_frames(30):  # right hand only: the left one is cleared each frame
        viewer.log(frame, t)
        seen.add(viewer._last_t_ns)
    assert 8 <= len(seen) <= 11  # 1 s of 30 Hz frames at 10 Hz
    viewer.log(None, 99.0)  # no frame yet: nothing to draw
