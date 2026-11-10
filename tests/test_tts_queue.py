#!/usr/bin/env python3
"""Tests for the TTS speak queue (finding 7) in PiperTTS.

speak() enqueues and returns; one worker synthesizes + plays in order.
interrupt() stops playback and drops the queue. Old wav files are pruned.
No real Piper binary or audio device is needed — synthesis/playback are
stubbed at the instance level.
"""

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.backend.tts import PiperTTS


class FakeBridge:
    def __init__(self):
        self.posts = []

    async def post(self, msg):
        self.posts.append(msg)


def make_tts(tmp_path):
    exe = tmp_path / "piper.exe"
    exe.write_bytes(b"fake")
    voice = tmp_path / "voice.onnx"
    voice.write_bytes(b"fake")
    out_dir = tmp_path / "piper_out"
    cfg = {
        "paths": {
            "piper_exe": str(exe),
            "piper_voice": str(voice),
            "piper_out_dir": str(out_dir),
        },
        "ui": {"tts_enabled": True},
    }
    tts = PiperTTS(cfg, FakeBridge())
    assert tts.enabled is True
    return tts


async def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


@pytest.mark.asyncio
async def test_speak_enqueues_and_plays_in_order(tmp_path):
    tts = make_tts(tmp_path)
    played = []
    synthesized = []

    async def fake_synthesize(text):
        synthesized.append(text)
        return f"wav-{text}"

    async def fake_play(wav):
        played.append(wav)

    tts._synthesize = fake_synthesize
    tts._play_audio = fake_play
    try:
        await tts.speak("one")
        await tts.speak("two")
        await tts.speak("three")
        assert await _wait_until(lambda: played == ["wav-one", "wav-two", "wav-three"])
        assert synthesized == ["one", "two", "three"]
    finally:
        await tts.stop()


@pytest.mark.asyncio
async def test_interrupt_drops_queue_and_current_item(tmp_path):
    tts = make_tts(tmp_path)
    played = []
    started = asyncio.Event()

    async def slow_synthesize(text):
        started.set()
        await asyncio.sleep(0.3)
        return f"wav-{text}"

    tts._synthesize = slow_synthesize
    tts._play_audio = AsyncMock(side_effect=lambda wav: played.append(wav))
    try:
        await tts.speak("one")
        assert await _wait_until(lambda: started.is_set())
        await tts.speak("two")
        await tts.speak("three")
        await tts.interrupt()
        # The in-flight synthesis is abandoned (generation bumped) and the
        # queued items are dropped: nothing ever plays.
        await asyncio.sleep(0.5)
        assert played == []
        assert tts._speak_queue.empty()
    finally:
        await tts.stop()


@pytest.mark.asyncio
async def test_muted_speech_is_dropped_not_queued(tmp_path):
    tts = make_tts(tmp_path)
    tts._synthesize = AsyncMock(return_value="wav-x")
    tts._play_audio = AsyncMock()
    try:
        assert tts.toggle_muted() is True
        await tts.speak("hello")
        await asyncio.sleep(0.2)
        tts._synthesize.assert_not_called()
        assert tts.toggle_muted() is False
    finally:
        await tts.stop()


@pytest.mark.asyncio
async def test_prune_wav_files_keeps_newest(tmp_path):
    tts = make_tts(tmp_path)
    out_dir = Path(tts.out_dir)
    now = time.time()
    for i in range(25):
        wav = out_dir / f"speech_{i:03d}.wav"
        wav.write_bytes(b"x")
        mtime = now - (25 - i)  # speech_024 is newest
        import os

        os.utime(wav, (mtime, mtime))
    try:
        tts._prune_wav_files()
        remaining = sorted(p.name for p in out_dir.glob("speech_*.wav"))
        assert len(remaining) == 20
        assert "speech_000.wav" not in remaining
        assert "speech_024.wav" in remaining
    finally:
        await tts.stop()


@pytest.mark.asyncio
async def test_stop_cancels_worker(tmp_path):
    tts = make_tts(tmp_path)
    tts._synthesize = AsyncMock(side_effect=lambda text: asyncio.sleep(5))
    tts._play_audio = AsyncMock()
    await tts.speak("never finishes")
    await asyncio.sleep(0.1)
    assert tts._speak_worker_task is not None
    await tts.stop()
    assert tts._speak_worker_task is None
