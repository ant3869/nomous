#!/usr/bin/env python3
"""Tests for the decoupled dashboard preview cadence (finding 9).

The preview now runs on a steady timer (~2 fps) regardless of motion,
while vision analysis stays on its own cooldown.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.backend.video import CameraLoop


def _bare_camera(preview_interval_sec=0.5):
    camera = CameraLoop.__new__(CameraLoop)
    camera.preview_interval_sec = preview_interval_sec
    return camera


def test_should_send_preview_on_interval():
    camera = _bare_camera(0.5)
    assert camera._should_send_preview(10.0, 9.0, "") is True  # 1.0s elapsed
    assert camera._should_send_preview(10.0, 9.6, "") is False  # 0.4s elapsed
    assert camera._should_send_preview(10.0, 9.5, "") is True  # exactly 0.5s


def test_should_send_preview_forces_on_gesture():
    camera = _bare_camera(0.5)
    # Gesture right after the last preview: still sends immediately.
    assert camera._should_send_preview(10.0, 10.0, "open_hand") is True
    assert camera._should_send_preview(10.0, 10.0, "") is False


def test_preview_interval_ignores_motion():
    # Motion is not an input to the preview decision at all: a still
    # room still gets previews on the timer, and heavy motion does not
    # speed the cadence up.
    camera = _bare_camera(0.5)
    assert camera._should_send_preview(10.0, 9.0, "") is True
    assert camera._should_send_preview(10.0, 9.9, "") is False


class _DummyBridge:
    def __init__(self):
        self.messages = []

    async def post(self, message):
        self.messages.append(message)


def _camera_config(**ui_overrides):
    ui = {
        "vision_enabled": True,
        "preview_interval_sec": 0.5,
        "motion_sensitivity": 10,
        "gesture_enabled": True,
        "gesture_cooldown": 3,
    }
    ui.update(ui_overrides)
    return {
        "ui": ui,
        "vision": {"vision_cooldown": 15},
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


def test_camera_loop_reads_preview_interval_from_config():
    bridge = _DummyBridge()
    loop = asyncio.new_event_loop()
    try:
        camera = CameraLoop(_camera_config(preview_interval_sec=1.25), bridge, loop)
        assert camera.preview_interval_sec == pytest.approx(1.25)
        # snapshot_debounce is gone: the attribute no longer exists.
        assert not hasattr(camera, "debounce_sec")
    finally:
        loop.close()


def test_camera_loop_defaults_preview_interval_to_half_second():
    bridge = _DummyBridge()
    loop = asyncio.new_event_loop()
    try:
        cfg = _camera_config()
        del cfg["ui"]["preview_interval_sec"]
        camera = CameraLoop(cfg, bridge, loop)
        assert camera.preview_interval_sec == pytest.approx(0.5)
    finally:
        loop.close()


def _load_run_bridge():
    path = project_root / "scripts" / "run_bridge.py"
    spec = importlib.util.spec_from_file_location("run_bridge_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_bridge_param_sets_preview_interval_and_alias():
    try:
        run_bridge = _load_run_bridge()
    except ImportError as exc:
        pytest.skip(f"run_bridge imports unavailable: {exc}")
    server = run_bridge.Server.__new__(run_bridge.Server)
    server.cam = types.SimpleNamespace(preview_interval_sec=0.5)
    server.tts = None
    server.stt = None
    server.llm = None
    await server.handle_param("preview_interval_sec", 1.5)
    assert server.cam.preview_interval_sec == pytest.approx(1.5)
    # Old dashboards still send snapshot_debounce; it maps to the same knob.
    await server.handle_param("snapshot_debounce", 2.0)
    assert server.cam.preview_interval_sec == pytest.approx(2.0)
