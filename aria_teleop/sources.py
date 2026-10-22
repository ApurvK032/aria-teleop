"""Hand + head-pose sources feeding a ``HandFuser``.

``AriaSource`` streams from the glasses with the Aria Gen 2 Client SDK.
``MockSource`` fakes the same two streams so the whole pipeline can be tested
without hardware.
"""

from __future__ import annotations

from datetime import timedelta
import threading
import time
from typing import Callable

import numpy as np

from .fusion import HandFuser, HandObservation
from .geometry import from_rotvec, invert_transform, make_transform, rot_y, rot_z

NUM_LANDMARKS = 21


def _timedelta_ns(td: timedelta) -> int:
    return (td // timedelta(microseconds=1)) * 1000


class AriaSource:
    """Live on-device hand tracking + VIO from Aria Gen 2 glasses.

    Starts streaming on the glasses (unless ``start_streaming`` is False, e.g.
    when you started it with ``aria_gen2 streaming start``), then runs the
    SDK's HTTP receiver. Prefers the ~800 Hz high-frequency VIO pose and falls
    back to the 10 Hz VIO output if the profile does not stream it. With
    ``eyes=True`` it also feeds on-device eye tracking to ``fuser.eyes``.
    """

    INTERFACES = ("usb", "wifi_sta", "wifi_sap")

    def __init__(
        self,
        fuser: HandFuser,
        profile: str = "profile9",
        profile_json: str = "",
        interface: str = "usb",
        device_ip: str = "",
        start_streaming: bool = True,
        port: int = 6768,
        min_pose_quality: float = 0.5,
        eyes: bool = False,
        on_rgb: Callable | None = None,
    ):
        if interface not in self.INTERFACES:
            raise ValueError(f"interface must be one of {self.INTERFACES}")
        self.fuser = fuser
        self.profile = profile
        self.profile_json = profile_json
        self.interface = interface
        self.device_ip = device_ip
        self.start_streaming = start_streaming
        self.port = port
        self.min_pose_quality = min_pose_quality
        self.eyes = eyes
        self.on_rgb = on_rgb  # RGB camera frames, decoded; off for teleop to keep CPU and memory low
        self._client = None  # must outlive the device: a collected client closes its connection
        self._device = None
        self._receiver = None
        self._have_high_freq = False
        self._callback_errors: set[str] = set()
        self._speech: str | None = None
        self._speech_ready = threading.Event()

    def _connect(self):
        import aria.sdk_gen2 as sdk_gen2

        self._client = sdk_gen2.DeviceClient()
        self._client.set_client_config(sdk_gen2.DeviceClientConfig())
        target = sdk_gen2.DeviceTarget(ip=self.device_ip) if self.device_ip else sdk_gen2.DeviceTarget()
        return self._client.connect(target)

    def start(self) -> None:
        import aria.sdk_gen2 as sdk_gen2
        import aria.stream_receiver as receiver
        from projectaria_tools.core.sensor_data import TrackingQuality, VioStatus

        self._vio_valid = VioStatus.VALID
        self._vio_good = TrackingQuality.GOOD

        if self.start_streaming:
            self._device = self._connect()
            config = sdk_gen2.HttpStreamingConfig()
            if self.profile_json:
                config.profile_json = self.profile_json
            else:
                config.profile_name = self.profile
            config.streaming_interface = {
                "usb": sdk_gen2.StreamingInterface.USB_NCM,
                "wifi_sta": sdk_gen2.StreamingInterface.WIFI_STA,
                "wifi_sap": sdk_gen2.StreamingInterface.WIFI_SAP,
            }[self.interface]
            self._device.set_streaming_config(config)
            self._device.start_streaming()
            print(f"Aria: streaming started ({self.profile_json or self.profile}, {self.interface})")
            threading.Thread(target=self._speech_loop, daemon=True).start()

        server = sdk_gen2.HttpServerConfig()
        server.address = "0.0.0.0"  # 127.0.0.1 never sees the glasses' connection
        server.port = self.port
        self._receiver = receiver.StreamReceiver(enable_image_decoding=self.on_rgb is not None)
        self._receiver.set_server_config(server)
        self._receiver.register_hand_pose_callback(self._guard("hand", self._on_hands))
        self._receiver.register_vio_high_frequency_callback(self._guard("vio_hf", self._on_vio_high_freq))
        self._receiver.register_vio_callback(self._guard("vio", self._on_vio))
        self._receiver.register_device_calib_callback(self._guard("calib", self._on_calib))
        if self.eyes:
            self._receiver.register_eye_gaze_callback(self._guard("eyes", self._on_eyes))
        if self.on_rgb is not None:
            self._receiver.register_rgb_callback(self._guard("rgb", self.on_rgb))
        self._receiver.start_server()
        print(f"Aria: receiver listening on 0.0.0.0:{self.port}")

    def stop(self) -> None:
        if self._device is not None:
            device, self._device = self._device, None  # also stops the speech thread
            self._speech_ready.set()
            try:
                device.stop_streaming()
            except Exception as e:  # retry once on a fresh connection, then give up
                print(f"Aria: stop_streaming failed ({e}); retrying on a new connection")
                try:
                    self._connect().stop_streaming()
                except Exception as e2:
                    print(f"Aria: stop_streaming failed again: {e2}. Run `aria_gen2 streaming stop`.")
            time.sleep(1.0)
            self._client = None
        if self._receiver is not None:
            self._receiver.stop_server()
            self._receiver = None

    def say(self, text: str) -> None:
        """Speak through the glasses' speaker, without blocking; a newer message replaces an unsent one."""
        self._speech = text
        self._speech_ready.set()

    def _speech_loop(self) -> None:
        while True:
            self._speech_ready.wait()
            self._speech_ready.clear()
            device, text, self._speech = self._device, self._speech, None
            if device is None:
                return
            if text is None:
                continue
            try:
                device.render_tts(text)
            except Exception as e:
                if "tts" not in self._callback_errors:
                    self._callback_errors.add("tts")
                    print(f"Aria: text-to-speech failed (further errors suppressed): {e!r}")

    def _guard(self, name, fn):
        """SDK threads swallow or crash on exceptions; report each kind once instead."""

        def wrapped(*args):
            try:
                fn(*args)
            except Exception as e:
                if name not in self._callback_errors:
                    self._callback_errors.add(name)
                    print(f"Aria: error in {name} callback (further ones suppressed): {e!r}")

        return wrapped

    def _on_hands(self, result) -> None:
        observations = {}
        for side, hand in (("left", result.left_hand), ("right", result.right_hand)):
            if hand is None or hand.transform_device_wrist is None:
                continue
            landmarks = hand.landmark_positions_device
            observations[side] = HandObservation(
                T_device_wrist=hand.transform_device_wrist.to_matrix(),
                landmarks_device=None if landmarks is None else np.asarray(landmarks, dtype=float).reshape(-1, 3),
                confidence=float(hand.confidence),
            )
        self.fuser.on_hands(_timedelta_ns(result.tracking_timestamp), observations)

    def _on_vio_high_freq(self, pose) -> None:
        self._have_high_freq = True
        if pose.quality_score <= self.min_pose_quality:
            return
        self.fuser.on_head_pose(
            _timedelta_ns(pose.tracking_timestamp),
            pose.transform_odometry_device.to_matrix(),
            session=pose.session_uid,
            gravity=np.asarray(pose.gravity_odometry, dtype=float).reshape(3),
        )

    def _on_vio(self, vio) -> None:
        if self._have_high_freq:
            return
        if vio.status != self._vio_valid or vio.pose_quality != self._vio_good:
            return
        T = vio.transform_odometry_bodyimu.to_matrix() @ vio.transform_bodyimu_device.to_matrix()
        self.fuser.on_head_pose(
            vio.capture_timestamp_ns,
            T,
            session=vio.frontend_session_uid,
            gravity=np.asarray(vio.gravity_in_odometry, dtype=float).reshape(3),
        )

    def _on_calib(self, calib) -> None:
        self.fuser.set_device_to_head(calib.get_transform_device_cpf().to_matrix())

    def _on_eyes(self, gaze) -> None:
        v = gaze.vergence
        # Despite the names, *_blink reads 1 while that eye is open (see eyes.py).
        self.fuser.eyes.on_sample(
            _timedelta_ns(gaze.tracking_timestamp),
            bool(v.left_blink) if v.left_blink_valid else None,
            bool(v.right_blink) if v.right_blink_valid else None,
        )


# Device axes expressed in the room frame for a level head facing room +x:
# camera-style x right, y down, z forward.
_R_ROOM_DEVICE_LEVEL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


class MockSource:
    """Stand-in for the glasses: a scripted wrist path seen from a swaying head.

    The head yaws back and forth on purpose. If head-motion compensation works,
    the robot traces the scripted path and ignores the sway. Adds noise,
    dropouts and occasional glitches so the gating gets exercised too. The eyes
    stay open apart from ordinary blinks, which must never pause the robot.
    """

    def __init__(
        self,
        fuser: HandFuser,
        hand: str = "right",
        hand_rate_hz: float = 30.0,
        head_rate_hz: float = 200.0,
        noise_m: float = 0.002,
        dropout: float = 0.02,
        glitch: float = 0.005,
        head_yaw_deg: float = 20.0,
        clutch_pinch: bool = False,
        blink_every_s: float = 3.7,
        blink_s: float = 0.3,
        seed: int = 0,
    ):
        self.fuser = fuser
        self.hand = hand
        self.blink_every_s = blink_every_s
        self.blink_s = blink_s
        self.hand_rate_hz = hand_rate_hz
        self.head_rate_hz = head_rate_hz
        self.noise_m = noise_m
        self.dropout = dropout
        self.glitch = glitch
        self.head_yaw = np.radians(head_yaw_deg)
        self.clutch_pinch = clutch_pinch
        self._rng = np.random.default_rng(seed)
        self._thread: threading.Thread | None = None
        self._running = False

    # Ground truth, in the room frame (x forward, y left, z up; head near the origin).
    def head_pose(self, t: float) -> np.ndarray:
        yaw = self.head_yaw * np.sin(2 * np.pi * t / 7.0)
        p = np.array([0.02 * np.sin(2 * np.pi * t / 5.0), 0.0, 0.01 * np.sin(2 * np.pi * t / 3.0)])
        return make_transform(rot_z(yaw) @ _R_ROOM_DEVICE_LEVEL, p)

    @staticmethod
    def wrist_pose(t: float) -> np.ndarray:
        # Centred on its t=1 s position (about when you'd engage), so the robot
        # target sweeps around home instead of drifting toward the arm's base.
        s = 2 * np.pi / 8.0 * (t - 1.0)
        p = np.array([0.40 + 0.08 * np.sin(s), -0.15 + 0.10 * np.sin(0.5 * s), -0.40 + 0.05 * np.sin(2 * s)])
        R = rot_z(0.3 * np.sin(s)) @ rot_y(0.2 * np.sin(0.5 * s))
        return make_transform(R, p)

    @staticmethod
    def pinch(t: float) -> float:
        return 0.05 + 0.035 * np.sin(2 * np.pi * t / 5.0)

    def eyes_open(self, t: float) -> bool:
        return t % self.blink_every_s >= self.blink_s

    def observe(self, t: float) -> dict[str, HandObservation]:
        """What the glasses would report at time ``t`` (device frame, with noise)."""
        T_device_room = invert_transform(self.head_pose(t))
        observations = {}
        if self._rng.random() >= self.dropout:
            T_room_wrist = self.wrist_pose(t)
            observations[self.hand] = self._observe_hand(T_device_room, T_room_wrist, self.pinch(t))
            if self._rng.random() < self.glitch:
                observations[self.hand].T_device_wrist[:3, 3] += [0.3, 0.0, 0.0]
        if self.clutch_pinch:
            other = "left" if self.hand == "right" else "right"
            T_room_other = make_transform(np.eye(3), [0.35, 0.20, -0.40])
            observations[other] = self._observe_hand(T_device_room, T_room_other, 0.01 if t > 1.0 else 0.06)
        return observations

    def _observe_hand(self, T_device_room: np.ndarray, T_room_wrist: np.ndarray, pinch: float) -> HandObservation:
        R, p = T_room_wrist[:3, :3], T_room_wrist[:3, 3]
        landmarks = np.tile(p, (NUM_LANDMARKS, 1))
        landmarks[0] = p + R @ [0.10, 0.03, 0.0]  # thumb tip
        landmarks[1] = landmarks[0] + R @ [0.0, pinch, 0.0]  # index tip
        landmarks_device = landmarks @ T_device_room[:3, :3].T + T_device_room[:3, 3]
        landmarks_device += self._rng.normal(0.0, self.noise_m, landmarks_device.shape)
        T_device_wrist = T_device_room @ T_room_wrist
        T_device_wrist[:3, 3] += self._rng.normal(0.0, self.noise_m, 3)
        T_device_wrist[:3, :3] = T_device_wrist[:3, :3] @ from_rotvec(self._rng.normal(0.0, 5.0 * self.noise_m, 3))
        return HandObservation(T_device_wrist, landmarks_device, confidence=0.9)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def say(self, text: str) -> None:
        pass  # no speaker on the mock; the run loop prints state changes anyway

    def _loop(self) -> None:
        t0 = time.monotonic()
        next_head = next_hand = 0.0
        while self._running:
            t = time.monotonic() - t0
            if t >= next_head:
                self.fuser.on_head_pose(int(t * 1e9), self.head_pose(t), session=0)
                next_head += 1.0 / self.head_rate_hz
            if t >= next_hand:
                self.fuser.on_hands(int(t * 1e9), self.observe(t))
                is_open = self.eyes_open(t)
                self.fuser.eyes.on_sample(int(t * 1e9), is_open, is_open)
                next_hand += 1.0 / self.hand_rate_hz
            time.sleep(max(0.0, min(next_head, next_hand) - (time.monotonic() - t0)))
