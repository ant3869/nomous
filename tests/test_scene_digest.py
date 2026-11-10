#!/usr/bin/env python3
"""Tests for the scene digest + describe-the-scene command (finding 3)."""

import asyncio
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "llama_cpp" not in sys.modules:
    llama_stub = types.ModuleType("llama_cpp")
    llama_stub.Llama = object
    sys.modules["llama_cpp"] = llama_stub

from src.backend.llm import LocalLLM


class FakeTTS:
    def __init__(self):
        self.spoken = []

    async def speak(self, text):
        self.spoken.append(text)

    async def interrupt(self):
        pass


class FakeBridge:
    def __init__(self):
        self.posts = []

    async def post(self, message):
        self.posts.append(message)


def _bare_llm(**overrides):
    """A LocalLLM with just enough state for chat/autonomous tests."""
    instance = LocalLLM.__new__(LocalLLM)
    instance.model = None
    instance._frame_memory = []
    instance.frame_memory_items = 5
    instance.frame_memory_max_tokens = 512
    instance._cfg = {"llm": {"n_ctx": 16384}}
    instance.system_prompt = "You are Nomous."
    instance.vision_image_max_tokens = 1024
    instance._model_lock = asyncio.Lock()
    instance._lanes = {"chat": False, "vision": False, "autonomous": False}
    instance._lane_lock = asyncio.Lock()
    instance.recent_context = []
    instance.max_context_items = 8
    instance._last_vision_description = ""
    instance._last_frame_description = ""
    instance._last_frame_image_url = None
    instance._vision_available = False
    instance.scene_digest_seconds = 600
    instance._last_scene_digest = time.time()
    instance._last_digest_signature = ""
    instance.last_thought = 0
    instance.thought_cooldown = 30
    instance.autonomous_mode = True
    instance.analytics = None
    instance.memory = None
    instance.person_tracker = None
    instance.tts = FakeTTS()
    instance.bridge = FakeBridge()
    instance._last_spoken_response = ""
    for key, value in overrides.items():
        setattr(instance, key, value)
    return instance


@pytest.mark.parametrize(
    "text,expected",
    [
        ("describe the scene", True),
        ("Describe what you see", True),
        ("hey, what do you see?", True),
        ("what's around you right now", True),
        ("what is happening over there", True),
        ("what can you see", True),
        ("hello there", False),
        ("what do you think about this", False),
        ("describe the recipe", False),
    ],
)
def test_is_describe_scene_request(text, expected):
    assert LocalLLM._is_describe_scene_request(text) is expected


@pytest.mark.asyncio
async def test_describe_scene_returns_empty_without_frame():
    llm = _bare_llm()
    assert await llm.describe_scene() == ""


@pytest.mark.asyncio
async def test_describe_scene_uses_live_frame():
    llm = _bare_llm(_last_frame_image_url="data:image/jpeg;base64,xxx")
    seen = {}

    async def fake_vision(image_data_url, prompt_text, max_tokens=60):
        seen["url"] = image_data_url
        seen["prompt"] = prompt_text
        return "A person sits at a desk."

    llm._generate_vision = fake_vision
    assert await llm.describe_scene() == "A person sits at a desk."
    assert seen["url"] == "data:image/jpeg;base64,xxx"
    # The fresh description becomes the current view for later prompts.
    assert llm._last_vision_description == "A person sits at a desk."


@pytest.mark.asyncio
async def test_chat_describe_scene_bypasses_text_generation():
    llm = _bare_llm(_last_frame_image_url="data:image/jpeg;base64,xxx")

    async def fake_vision(image_data_url, prompt_text, max_tokens=60):
        return "A cat sleeps on the rug."

    llm._generate_vision = fake_vision

    async def _generate_must_not_run(*args, **kwargs):
        raise AssertionError("_generate must not run for a describe-scene request")

    llm._generate = _generate_must_not_run

    messages = []
    async for message in llm.chat("what do you see?"):
        messages.append(message)
    assert messages == [{"type": "speak", "text": "A cat sleeps on the rug."}]
    assert llm.tts.spoken == ["A cat sleeps on the rug."]


@pytest.mark.asyncio
async def test_chat_describe_scene_without_frame_says_so():
    llm = _bare_llm()

    async def _generate_must_not_run(*args, **kwargs):
        raise AssertionError("_generate must not run for a describe-scene request")

    llm._generate = _generate_must_not_run
    messages = [m async for m in llm.chat("describe the scene")]
    assert messages == [{"type": "speak", "text": "I don't have a camera frame to look at right now."}]


@pytest.mark.asyncio
async def test_chat_normal_request_still_uses_text_generation():
    llm = _bare_llm()
    llm._classify_input = lambda text: {
        "is_greeting": False,
        "is_affirmation": False,
        "is_vision_question": False,
        "skip_tools": True,
        "suggested_tokens": 50,
    }
    llm._build_prompt = lambda *args, **kwargs: "prompt"

    async def _async_return(value):
        return value

    llm._generate = lambda *a, **k: _async_return("Hello there.")
    messages = [m async for m in llm.chat("hello")]
    assert messages == [{"type": "speak", "text": "Hello there."}]


def test_scene_digest_due_timing():
    llm = _bare_llm(_last_frame_image_url="data:image/jpeg;base64,xxx")
    # Fresh boot: digest clock starts now, nothing due yet.
    assert llm._scene_digest_due() is False
    # Ten minutes on, static scene: due.
    llm._last_scene_digest = time.time() - 601
    assert llm._scene_digest_due() is True
    # Disabled when the interval is 0.
    llm.scene_digest_seconds = 0
    assert llm._scene_digest_due() is False
    # No frame at all: never due.
    llm.scene_digest_seconds = 600
    llm._last_frame_image_url = None
    assert llm._scene_digest_due() is False


def test_scene_digest_skips_when_scene_changed():
    llm = _bare_llm(
        _last_frame_image_url="data:image/jpeg;base64,xxx",
        _last_frame_description="a dog runs across the room",
        _last_digest_signature="an empty room with a chair",
    )
    llm._last_scene_digest = time.time() - 601
    assert llm._scene_digest_due() is False
    # Same description as the last digest: static, due.
    llm._last_frame_description = "an empty room with a chair"
    assert llm._scene_digest_due() is True


@pytest.mark.asyncio
async def test_autonomous_thought_posts_digest_when_due():
    llm = _bare_llm(
        _last_frame_image_url="data:image/jpeg;base64,xxx",
        _last_frame_description="an empty room with a chair",
    )
    llm._last_scene_digest = time.time() - 601

    async def fake_vision(image_data_url, prompt_text, max_tokens=60):
        return "An empty room with a chair."

    llm._generate_vision = fake_vision
    await llm.autonomous_thought()
    assert llm.bridge.posts == [{"type": "thought", "text": "An empty room with a chair."}]
    # Digest clock reset and signature recorded; not due again immediately.
    assert llm._scene_digest_due() is False
    assert llm._last_digest_signature == "an empty room with a chair"
