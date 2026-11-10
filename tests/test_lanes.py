#!/usr/bin/env python3
"""Tests for the processing lanes (finding 6) in LocalLLM.

Lanes replace the single shared _processing flag: chat/audio share one
lane, vision queues behind chat, autonomous waits for chat to be free,
and a model lock serializes inference. No model file is needed — the
LLM is constructed with load_immediately=False and generation is stubbed.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.backend.llm import LocalLLM


class FakeBridge:
    def __init__(self):
        self.posts = []

    async def post(self, msg):
        self.posts.append(msg)


class FakeTTS:
    def __init__(self):
        self.spoken = []
        self.interrupts = 0

    async def speak(self, text):
        self.spoken.append(text)

    async def interrupt(self):
        self.interrupts += 1

    async def stop(self):
        pass


def make_llm():
    cfg = {
        "paths": {"gguf_path": "models/test.gguf", "mmproj_path": ""},
        "llm": {"temperature": 0.7, "top_p": 0.95},
        "vision": {"vision_cooldown": 15},
        "ui": {"tts_enabled": True},
    }
    llm = LocalLLM(cfg, FakeBridge(), FakeTTS(), load_immediately=False)
    llm.last_vision_analysis = 0
    return llm


def _consume_chat(llm, text):
    """Drive an async-generator chat() to completion, discarding chunks."""

    async def _run():
        async for _ in llm.chat(text):
            pass

    return _run()


@pytest.mark.asyncio
async def test_chat_lane_blocks_second_chat():
    llm = make_llm()
    calls = []

    async def slow_generate(*args, **kwargs):
        calls.append(1)
        await asyncio.sleep(0.2)
        return "hello there"

    llm._generate = slow_generate
    llm._generate_vision = AsyncMock(return_value="")

    first = asyncio.create_task(_consume_chat(llm, "hi"))
    await asyncio.sleep(0.05)
    assert llm._lane_busy("chat") is True

    # Second chat while the lane is held is skipped, as before.
    await _consume_chat(llm, "hello again")
    await first

    assert len(calls) == 1
    assert llm._lane_busy("chat") is False


@pytest.mark.asyncio
async def test_chat_interrupts_tts_on_new_input():
    llm = make_llm()
    llm._generate = AsyncMock(return_value="hi")
    llm._generate_vision = AsyncMock(return_value="")

    await _consume_chat(llm, "hello")
    assert llm.tts.interrupts == 1


@pytest.mark.asyncio
async def test_process_audio_shares_chat_lane():
    llm = make_llm()
    llm._generate = AsyncMock(return_value="hi")
    llm._generate_vision = AsyncMock(return_value="")

    await llm.process_audio("hello")
    assert llm.tts.interrupts == 1
    assert llm._lane_busy("chat") is False


@pytest.mark.asyncio
async def test_vision_queues_behind_chat():
    llm = make_llm()
    llm.vision_cooldown = 0  # don't let the cooldown re-check eat the queue
    llm._generate = AsyncMock(return_value="")
    llm._generate_vision = AsyncMock(return_value="")

    impl_calls = []
    orig_impl = llm._process_vision_impl

    async def counting_impl(*args, **kwargs):
        impl_calls.append(1)
        return await orig_impl(*args, **kwargs)

    llm._process_vision_impl = counting_impl

    # Hold the chat lane: vision must queue, not drop.
    assert await llm._acquire_lane("chat")
    await llm.process_vision("Someone new has entered the scene")
    await llm.process_vision("Someone new has entered the scene")
    await llm.process_vision("A person is waving at the camera")
    assert llm._vision_queue.qsize() == 2  # duplicate dropped, distinct kept

    llm._release_lane("chat")
    for _ in range(100):
        if llm._vision_queue.empty() and len(impl_calls) == 2:
            break
        await asyncio.sleep(0.05)
    assert llm._vision_queue.empty()
    assert len(impl_calls) == 2  # both queued frames were analyzed
    assert llm._lane_busy("vision") is False
    await llm.stop()


@pytest.mark.asyncio
async def test_vision_queue_full_drops_oldest():
    llm = make_llm()
    llm._generate = AsyncMock(return_value="")
    llm._generate_vision = AsyncMock(return_value="")

    assert await llm._acquire_lane("chat")
    for i in range(10):
        await llm.process_vision(f"Frame number {i} with movement")
    assert llm._vision_queue.qsize() == 8  # maxsize bound holds
    queued = [item[0] for item in list(llm._vision_queue._queue)]
    assert "Frame number 0 with movement" not in queued
    assert "Frame number 9 with movement" in queued
    llm._release_lane("chat")
    await llm.stop()


@pytest.mark.asyncio
async def test_autonomous_skips_when_chat_busy(monkeypatch):
    llm = make_llm()
    llm._generate = AsyncMock(return_value="a thought")
    llm._generate_vision = AsyncMock(return_value="")
    llm.last_thought = 0

    assert await llm._acquire_lane("chat")
    await llm.autonomous_thought()
    llm._generate.assert_not_called()
    llm._release_lane("chat")

    # Chat free: the thought runs (force the speak branch, not silence).
    monkeypatch.setattr("random.random", lambda: 0.9)
    await llm.autonomous_thought()
    llm._generate.assert_called_once()
    assert llm._lane_busy("autonomous") is False


@pytest.mark.asyncio
async def test_reload_model_refuses_when_lane_busy():
    llm = make_llm()
    assert await llm._acquire_lane("chat")
    with pytest.raises(RuntimeError, match="while processing"):
        await llm.reload_model("models/other.gguf")
    llm._release_lane("chat")


@pytest.mark.asyncio
async def test_stop_cancels_vision_worker():
    llm = make_llm()
    llm._generate = AsyncMock(return_value="")
    llm._generate_vision = AsyncMock(return_value="")

    assert await llm._acquire_lane("chat")
    await llm.process_vision("Someone new has entered the scene")
    assert llm._vision_worker_task is not None
    llm._release_lane("chat")

    await llm.stop()
    assert llm._vision_worker_task is None
