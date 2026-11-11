#!/usr/bin/env python3
"""Tests for gesture-to-action mappings (finding 10).

open_hand pauses in-flight speech, closed_fist toggles the mute, and
unmapped gestures only report — they do not act.
"""

import asyncio
import sys
import threading
import time
import types
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.backend.video import CameraLoop


class FakeTTS:
    def __init__(self):
        self.muted = False
        self.interrupts = 0

    def toggle_muted(self):
        self.muted = not self.muted
        return self.muted

    async def interrupt(self):
        self.interrupts += 1


class FakeBridge:
    def __init__(self):
        self.messages = []

    async def post(self, message):
        self.messages.append(message)


def _wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.fixture
def camera_with_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    camera = CameraLoop.__new__(CameraLoop)
    camera.loop = loop
    camera.bridge = FakeBridge()
    camera.llm = types.SimpleNamespace(tts=FakeTTS())
    camera._last_gesture = ""
    camera.last_gesture_time = 0
    yield camera
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def _event_texts(camera):
    return [m.get("message", "") for m in camera.bridge.messages]


def test_open_hand_pauses_speech(camera_with_loop):
    camera = camera_with_loop
    camera._dispatch_gesture("open_hand")
    assert _wait_until(lambda: camera.llm.tts.interrupts == 1)
    assert _wait_until(lambda: "✋ Paused speech" in _event_texts(camera))


def test_closed_fist_toggles_mute(camera_with_loop):
    camera = camera_with_loop
    camera._dispatch_gesture("closed_fist")
    assert camera.llm.tts.muted is True
    assert _wait_until(lambda: "🔇 Muted" in _event_texts(camera))
    camera._dispatch_gesture("closed_fist")
    assert camera.llm.tts.muted is False
    assert _wait_until(lambda: "🔊 Unmuted" in _event_texts(camera))


def test_unmapped_gesture_does_nothing(camera_with_loop):
    camera = camera_with_loop
    camera._dispatch_gesture("thumbs_up")
    camera._dispatch_gesture("peace_sign")
    time.sleep(0.2)
    assert camera.llm.tts.interrupts == 0
    assert camera.llm.tts.muted is False
    assert camera.bridge.messages == []


def test_held_fist_toggles_mute_only_once(camera_with_loop):
    camera = camera_with_loop
    camera._handle_detected_gesture("closed_fist", 100.0)
    assert camera.llm.tts.muted is True
    assert _wait_until(lambda: "🔇 Muted" in _event_texts(camera))
    events_after_first = len(camera.bridge.messages)
    # Held fist, re-detected after the cooldown: no second toggle.
    camera._handle_detected_gesture("closed_fist", 104.0)
    time.sleep(0.2)
    assert camera.llm.tts.muted is True
    assert len(camera.bridge.messages) == events_after_first
    # Released, then a fresh fist: toggles again.
    camera._handle_detected_gesture("", 108.0)
    camera._handle_detected_gesture("closed_fist", 112.0)
    assert camera.llm.tts.muted is False
    assert _wait_until(lambda: "🔊 Unmuted" in _event_texts(camera))


def test_open_hand_interrupt_fires_once_per_press(camera_with_loop):
    camera = camera_with_loop
    camera._handle_detected_gesture("open_hand", 100.0)
    assert _wait_until(lambda: camera.llm.tts.interrupts == 1)
    camera._handle_detected_gesture("open_hand", 104.0)
    time.sleep(0.2)
    assert camera.llm.tts.interrupts == 1
    camera._handle_detected_gesture("", 108.0)
    camera._handle_detected_gesture("open_hand", 112.0)
    assert _wait_until(lambda: camera.llm.tts.interrupts == 2)


def test_gesture_dispatch_without_llm_or_tts_is_silent():
    camera = CameraLoop.__new__(CameraLoop)
    camera.loop = asyncio.new_event_loop()
    camera.bridge = FakeBridge()
    camera.llm = None
    camera._dispatch_gesture("open_hand")  # must not raise
    camera.llm = types.SimpleNamespace(tts=None)
    camera._dispatch_gesture("closed_fist")  # must not raise
    camera.loop.close()
