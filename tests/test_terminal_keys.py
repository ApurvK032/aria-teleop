import os
import sys
import termios
import time

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("trossen_arm_mujoco")

from aria_teleop.sim import terminal_keys  # noqa: E402


def _wait_for(pressed: list, n: int, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while len(pressed) < n and time.monotonic() < deadline:
        time.sleep(0.01)


def test_key_presses_reach_the_handler_and_the_terminal_is_restored(monkeypatch):
    master, slave = os.openpty()
    stdin = os.fdopen(slave, "r")
    monkeypatch.setattr(sys, "stdin", stdin)
    before = termios.tcgetattr(slave)
    pressed = []
    try:
        with terminal_keys(pressed.append):
            assert not termios.tcgetattr(slave)[3] & termios.ICANON  # cbreak: keys arrive one at a time
            os.write(master, b" h")
            _wait_for(pressed, 2)
        assert pressed == [ord(" "), ord("H")]  # the viewer's key codes
        assert termios.tcgetattr(slave) == before
    finally:
        stdin.close()
        os.close(master)


def test_does_nothing_when_stdin_is_not_a_terminal(monkeypatch):
    r, w = os.pipe()
    stdin = os.fdopen(r, "r")
    monkeypatch.setattr(sys, "stdin", stdin)
    pressed = []
    try:
        with terminal_keys(pressed.append):
            os.write(w, b" ")
            time.sleep(0.2)
        assert pressed == []
    finally:
        stdin.close()
        os.close(w)
