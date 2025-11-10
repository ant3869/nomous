import asyncio
import sys
import types
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _unpatched_canny(*args, **kwargs):  # pragma: no cover - safety guard
    raise RuntimeError("unpatched")


def _noop(*args, **kwargs):  # pragma: no cover - deterministic stub
    return None


class _CascadeStub:
    def __init__(self, *args, **kwargs):
        self.loaded = True

    def detectMultiScale(self, *args, **kwargs):  # pragma: no cover - deterministic stub
        return []


class _VideoCaptureStub:
    def __init__(self, *args, **kwargs):
        self._opened = True

    def isOpened(self):
        return self._opened

    def set(self, *args, **kwargs):
        return True

    def get(self, *args, **kwargs):
        return 0

    def read(self):
        return False, None

    def release(self):
        self._opened = False


cv2_stub = types.SimpleNamespace(
    CAP_DSHOW=0,
    CAP_ANY=1,
    data=types.SimpleNamespace(haarcascades=""),
    CascadeClassifier=_CascadeStub,
    VideoCapture=_VideoCaptureStub,
    Canny=_unpatched_canny,
    countNonZero=lambda *args, **kwargs: 0,
    resize=_noop,
    cvtColor=_noop,
    absdiff=_noop,
)

sys.modules["cv2"] = cv2_stub


class _HandLandmarkStub:
    WRIST = 0
    THUMB_TIP = 1
    INDEX_FINGER_TIP = 2
    MIDDLE_FINGER_TIP = 3
    RING_FINGER_TIP = 4
    PINKY_TIP = 5


class _HandsStub:
    def __init__(self, *args, **kwargs):
        pass

    def process(self, *args, **kwargs):
        return types.SimpleNamespace(multi_hand_landmarks=None)

    def close(self):
        pass


mediapipe_stub = types.SimpleNamespace(
    solutions=types.SimpleNamespace(
        hands=types.SimpleNamespace(
            Hands=_HandsStub,
            HandLandmark=_HandLandmarkStub,
        )
    )
)

sys.modules["mediapipe"] = mediapipe_stub

from src.backend import video


class DummyBridge:
    def __init__(self):
        self.messages = []

    async def post(self, message):
        self.messages.append(message)


@pytest.fixture
def camera_config():
    return {
        "ui": {
            "vision_enabled": True,
            "snapshot_debounce": 1,
            "motion_sensitivity": 10,
            "gesture_enabled": True,
            "gesture_cooldown": 3,
            "vision_cooldown": 12,
        },
        "camera": {
            "index": 0,
            "backend": "dshow",
            "width": 640,
            "height": 480,
            "process_width": 320,
            "process_height": 240,
            "frame_skip": 1,
        },
    }


def test_camera_loop_handles_missing_numpy(monkeypatch, camera_config):
    monkeypatch.setattr(video, "NUMPY_AVAILABLE", False, raising=False)
    monkeypatch.setattr(video, "NUMPY_IMPORT_ERROR", RuntimeError("numpy source tree"), raising=False)
    monkeypatch.setattr(video, "np", None, raising=False)

    bridge = DummyBridge()
    loop = asyncio.new_event_loop()

    try:
        camera = video.CameraLoop(camera_config, bridge, loop)

        assert camera._numpy_available is False
        assert isinstance(camera._numpy_error, RuntimeError)

        class DummyEdges:
            size = 4
            shape = (2, 2)

        def fake_canny(gray, _low, _high):
            return DummyEdges()

        def fake_count(edges):
            return 3

        monkeypatch.setattr(video.cv2, "Canny", fake_canny)
        monkeypatch.setattr(video.cv2, "countNonZero", fake_count)

        description = camera._analyze_complexity(object())
        assert description == "very detailed"

        captured = []

        def fake_post_event(message):
            captured.append(message)

        monkeypatch.setattr(camera, "_post_event", fake_post_event)

        class DummyThread:
            def __init__(self, target=None, daemon=None, name=None):
                self.target = target

            def start(self):
                pass

            def join(self, timeout=None):
                pass

        monkeypatch.setattr(video.threading, "Thread", DummyThread)

        camera.start()
        camera.stop()

        assert any("NumPy" in message for message in captured)
    finally:
        loop.close()
