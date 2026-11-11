# Title: Tests for AmbientHearing - hearing as a sense
# Path: tests/backend/test_hearing.py
# Purpose: Cover the hearing sense without a mic or PortAudio: VAD
# segmentation on synthetic PCM, the real webrtcvad on silence, evidence
# TTL + retention gating on the LLM, the hearing->vision prompt tick and
# its rate limit, listen_boost, and graceful failure when the mic stack
# is missing (this Linux VM has no PortAudio — start() must return False,
# never raise into startup).

import asyncio
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "llama_cpp" not in sys.modules:
    llama_stub = types.ModuleType("llama_cpp")
    llama_stub.Llama = object
    sys.modules["llama_cpp"] = llama_stub

from src.backend.config import DEFAULT, load_config
from src.backend.hearing import AmbientHearing, HearingSegmenter, _rms_normalized
from src.backend.llm import LocalLLM

FRAME = b"\x00" * 960  # 30 ms of silence @ 16 kHz int16 mono


def _loud_frame(amplitude: int = 20000) -> bytes:
    return (amplitude.to_bytes(2, "little", signed=True)) * 480


def _bare_hearing_llm(**overrides):
    """A LocalLLM without a loaded model, with the audio-seam attrs set."""
    instance = LocalLLM.__new__(LocalLLM)
    instance._last_audio_evidence = ""
    instance._last_audio_at = 0.0
    instance._audio_evidence_ttl_s = 60
    instance._audio_retain_transcripts = True
    instance.autonomous_mode = True
    instance._last_audio_prompt_tick = 0.0
    instance._hearing = None
    for key, value in overrides.items():
        setattr(instance, key, value)
    return instance


# --- Segmenter: pure logic, fake VAD ---------------------------------------


def test_segmenter_silence_only_emits_nothing():
    seg = HearingSegmenter(lambda frame: False)
    for _ in range(100):
        assert seg.feed(FRAME) == []
    assert seg.flush() == []


def test_segmenter_speech_burst_ends_on_silence():
    # Fake VAD: loud frames are speech, silence frames are not.
    seg = HearingSegmenter(lambda frame: _rms_normalized(frame) > 0.1, silence_ms=90)
    events = []
    for _ in range(5):
        events.extend(seg.feed(_loud_frame()))
    assert events == []  # segment still open
    for _ in range(3):  # 3 x 30 ms of silence ends it
        events.extend(seg.feed(FRAME))
    assert len(events) == 1
    kind, pcm = events[0]
    assert kind == "speech"
    assert pcm == _loud_frame() * 5


def test_segmenter_caps_long_segments():
    seg = HearingSegmenter(lambda frame: True, max_segment_s=1)
    events = []
    for _ in range(40):  # > 1 s of speech
        events.extend(seg.feed(_loud_frame()))
    assert len(events) >= 1
    assert events[0][0] == "speech"
    assert len(events[0][1]) == 960 * 33  # 33 frames = 990 ms cap


def test_segmenter_flush_ends_open_segment():
    seg = HearingSegmenter(lambda frame: _rms_normalized(frame) > 0.1)
    seg.feed(_loud_frame())
    seg.feed(_loud_frame())
    events = seg.flush()
    assert len(events) == 1 and events[0][0] == "speech"
    assert seg.flush() == []


def test_segmenter_loud_non_speech_burst_is_noise_event():
    seg = HearingSegmenter(lambda frame: False, noise_rms_threshold=0.35)
    events = []
    for _ in range(3):
        events.extend(seg.feed(_loud_frame()))
    assert len(events) == 1
    assert events[0][0] == "noise"
    assert events[0][1] > 0.35


def test_segmenter_noise_events_debounced():
    seg = HearingSegmenter(lambda frame: False, noise_rms_threshold=0.35, noise_debounce_s=60)
    events = []
    for _ in range(6):
        events.extend(seg.feed(_loud_frame()))
    assert len(events) == 1  # second burst inside the debounce window


def test_segmenter_quiet_frames_are_not_noise():
    seg = HearingSegmenter(lambda frame: False, noise_rms_threshold=0.35)
    events = []
    for _ in range(10):
        events.extend(seg.feed(FRAME))
    assert events == []


def test_rms_normalized():
    assert _rms_normalized(FRAME) == 0.0
    assert 0.5 < _rms_normalized(_loud_frame()) < 0.7


def test_segmenter_with_real_webrtcvad_on_silence():
    import webrtcvad

    vad = webrtcvad.Vad(2)
    seg = HearingSegmenter(lambda frame: vad.is_speech(frame, 16000))
    for _ in range(100):
        assert seg.feed(FRAME) == []


# --- AmbientHearing: construction, boost, graceful failure -----------------


def test_hearing_defaults_opt_in_off():
    hearing = AmbientHearing({}, SimpleNamespace(post=AsyncMock()))
    assert hearing.enabled is False
    assert hearing.sample_rate == 16000
    assert hearing.vad_aggressiveness == 2
    assert hearing.min_transcript_chars == 3


def test_hearing_reads_config_overrides():
    cfg = {"hearing": {"enabled": True, "vad_aggressiveness": 3, "device": "mic0"}}
    hearing = AmbientHearing(cfg, SimpleNamespace(post=AsyncMock()))
    assert hearing.enabled is True
    assert hearing.vad_aggressiveness == 3
    assert hearing.device == "mic0"


def test_listen_boost_is_noop_when_stopped():
    hearing = AmbientHearing({}, SimpleNamespace(post=AsyncMock()))
    hearing.listen_boost()
    assert hearing._boost_until == 0.0
    assert hearing._effective_aggressiveness() == 2


def test_listen_boost_lowers_aggressiveness_while_active():
    hearing = AmbientHearing({"hearing": {"vad_aggressiveness": 2}}, SimpleNamespace(post=AsyncMock()))
    hearing._running = True
    hearing.listen_boost(30)
    assert hearing._effective_aggressiveness() == 1
    hearing._boost_until = time.monotonic() - 1  # expired
    assert hearing._effective_aggressiveness() == 2


def test_listen_boost_never_below_zero():
    hearing = AmbientHearing({"hearing": {"vad_aggressiveness": 0}}, SimpleNamespace(post=AsyncMock()))
    hearing._running = True
    hearing.listen_boost(30)
    assert hearing._effective_aggressiveness() == 0


def test_transcribe_without_recognizer_returns_empty():
    hearing = AmbientHearing({}, SimpleNamespace(post=AsyncMock()))
    assert hearing._transcribe(b"\x00" * 9600) == ""


@pytest.mark.asyncio
async def test_start_returns_false_without_mic_stack():
    # This VM has no PortAudio: start() must fail soft, never raise.
    bridge = SimpleNamespace(post=AsyncMock())
    hearing = AmbientHearing({}, bridge)
    assert await hearing.start() is False
    assert hearing._running is False
    bridge.post.assert_awaited()
    posted = bridge.post.await_args.args[0]
    assert posted["type"] == "hearing"
    assert posted["state"] == "error"


@pytest.mark.asyncio
async def test_set_enabled_false_after_failed_start():
    hearing = AmbientHearing({}, SimpleNamespace(post=AsyncMock()))
    await hearing.set_enabled(True)
    assert hearing.enabled is False  # start failed -> stays disabled


@pytest.mark.asyncio
async def test_emit_evidence_posts_hearing_message_and_fills_slot():
    bridge = SimpleNamespace(post=AsyncMock())
    llm = _bare_hearing_llm()
    llm.autonomous_thought = AsyncMock()
    llm.autonomous_mode = False  # no prompt tick in this test
    hearing = AmbientHearing({}, bridge)
    hearing.set_llm(llm)
    await hearing._emit_evidence("hello from the room", "speech")
    bridge.post.assert_awaited()
    posted = bridge.post.await_args.args[0]
    assert posted["type"] == "hearing"
    assert posted["state"] == "listening"
    assert posted["text"] == "hello from the room"
    assert posted["kind"] == "speech"
    assert llm._last_audio_evidence == "hello from the room"


# --- LLM: evidence slot, prompt tick, retention gate -----------------------


@pytest.mark.asyncio
async def test_note_audio_evidence_fills_slot_and_prompts_tick():
    llm = _bare_hearing_llm()
    llm.autonomous_thought = AsyncMock()
    await llm.note_audio_evidence("someone said hello", "speech")
    assert llm._last_audio_evidence == "someone said hello"
    assert llm._last_audio_at > 0
    await asyncio.sleep(0)  # let the fire-and-forget prompt tick run
    llm.autonomous_thought.assert_awaited_once_with(force=True)


@pytest.mark.asyncio
async def test_note_audio_evidence_noise_kind():
    llm = _bare_hearing_llm()
    llm.autonomous_thought = AsyncMock()
    await llm.note_audio_evidence("a loud noise", "noise")
    assert llm._last_audio_evidence == "a loud noise"
    await asyncio.sleep(0)  # let the fire-and-forget prompt tick run
    llm.autonomous_thought.assert_awaited_once_with(force=True)


@pytest.mark.asyncio
async def test_note_audio_evidence_rate_limits_prompt_ticks():
    llm = _bare_hearing_llm()
    llm.autonomous_thought = AsyncMock()
    await llm.note_audio_evidence("first sound", "speech")
    await llm.note_audio_evidence("second sound", "speech")
    await asyncio.sleep(0)  # let the fire-and-forget prompt tick run
    assert llm.autonomous_thought.await_count == 1


@pytest.mark.asyncio
async def test_note_audio_evidence_no_tick_when_autonomous_off():
    llm = _bare_hearing_llm(autonomous_mode=False)
    llm.autonomous_thought = AsyncMock()
    await llm.note_audio_evidence("someone said hello", "speech")
    assert llm._last_audio_evidence == "someone said hello"
    llm.autonomous_thought.assert_not_awaited()


@pytest.mark.asyncio
async def test_note_audio_evidence_ignores_empty_text():
    llm = _bare_hearing_llm()
    llm.autonomous_thought = AsyncMock()
    await llm.note_audio_evidence("   ", "speech")
    assert llm._last_audio_evidence == ""
    llm.autonomous_thought.assert_not_awaited()


def test_idle_audio_evidence_ttl():
    llm = _bare_hearing_llm(_last_audio_evidence="hello", _last_audio_at=time.time())
    assert llm._idle_audio_evidence(time.time()) == "hello"
    assert llm._idle_audio_evidence(time.time() + 120) == ""


def test_should_record_autonomous_retention_gate():
    now = time.time()
    fresh = _bare_hearing_llm(_last_audio_evidence="hello", _last_audio_at=now)
    assert fresh._should_record_autonomous(now) is True

    no_retain = _bare_hearing_llm(
        _last_audio_evidence="hello", _last_audio_at=now, _audio_retain_transcripts=False
    )
    assert no_retain._should_record_autonomous(now) is False

    stale = _bare_hearing_llm(
        _last_audio_evidence="hello", _last_audio_at=now - 120, _audio_retain_transcripts=False
    )
    assert stale._should_record_autonomous(now) is True  # stale audio: normal recording

    none = _bare_hearing_llm(_audio_retain_transcripts=False)
    assert none._should_record_autonomous(now) is True


@pytest.mark.asyncio
async def test_autonomous_thought_force_skips_cooldown():
    llm = _bare_hearing_llm()
    llm.last_thought = time.time()  # inside the cooldown window
    llm.thought_cooldown = 30
    llm._lane_busy = lambda lane: False
    llm._acquire_lane = AsyncMock(return_value=True)
    llm._release_lane = MagicMock()
    llm._scene_digest_due = lambda: False
    llm._idle_decide = AsyncMock()

    await llm.autonomous_thought()  # not forced: cooldown holds
    llm._idle_decide.assert_not_awaited()

    await llm.autonomous_thought(force=True)  # forced: tick runs
    llm._idle_decide.assert_awaited_once()


def test_set_hearing_connects_sense():
    llm = _bare_hearing_llm()
    hearing = AmbientHearing({}, SimpleNamespace(post=AsyncMock()))
    llm.set_hearing(hearing)
    assert llm._hearing is hearing


# --- Config ----------------------------------------------------------------


def test_hearing_config_defaults_opt_in():
    hearing = DEFAULT["hearing"]
    assert hearing["enabled"] is False
    assert hearing["sample_rate"] == 16000
    assert hearing["vad_aggressiveness"] == 2
    assert hearing["segment_silence_ms"] == 500
    assert hearing["max_segment_s"] == 10
    assert hearing["min_transcript_chars"] == 3
    assert hearing["noise_rms_threshold"] == 0.35
    assert hearing["retain_transcripts"] is True
    assert hearing["evidence_ttl_s"] == 60
    assert hearing["listen_boost_seconds"] == 30


def test_load_config_merges_hearing_overrides(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("hearing:\n  enabled: true\n  vad_aggressiveness: 1\n")
    cfg = load_config(str(yaml_path))
    assert cfg["hearing"]["enabled"] is True
    assert cfg["hearing"]["vad_aggressiveness"] == 1
    assert cfg["hearing"]["evidence_ttl_s"] == 60  # untouched keys keep defaults
