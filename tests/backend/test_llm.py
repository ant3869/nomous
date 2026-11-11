import asyncio
import sys
import time
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "llama_cpp" not in sys.modules:
    llama_stub = types.ModuleType("llama_cpp")
    llama_stub.Llama = object
    sys.modules["llama_cpp"] = llama_stub

from src.backend.llm import LocalLLM


def sanitize(text: str) -> str:
    instance = LocalLLM.__new__(LocalLLM)
    return LocalLLM._sanitize_response(instance, text)


def test_sanitize_removes_explicit_scheduling_tokens_with_equals():
    text = "I'll handle this respond_in=5s before continuing."
    sanitized = sanitize(text)
    assert "respond_in=5s" not in sanitized
    assert "I'll handle this before continuing." == sanitized


def test_sanitize_removes_explicit_scheduling_tokens_with_words():
    text = "Set a reminder to respond-in 10 seconds, then proceed."
    sanitized = sanitize(text)
    assert "respond-in 10 seconds" not in sanitized
    assert "Set a reminder to  then proceed." == sanitized


def test_sanitize_preserves_non_scheduling_phrase():
    text = "Please respond in 3 steps so I can follow along."
    sanitized = sanitize(text)
    assert "respond in 3 steps" in sanitized
    assert sanitized.endswith("follow along.")


def _bare_llm(**overrides):
    """A LocalLLM without a loaded model, for pure-helper tests."""
    instance = LocalLLM.__new__(LocalLLM)
    instance.model = None
    instance._frame_memory = []
    instance.frame_memory_items = 5
    instance.frame_memory_max_tokens = 512
    instance._cfg = {"llm": {"n_ctx": 16384}}
    instance.system_prompt = "You are Nomous."
    instance.vision_image_max_tokens = 1024
    # _generate/_generate_vision serialize inference on the model lock.
    instance._model_lock = asyncio.Lock()
    for key, value in overrides.items():
        setattr(instance, key, value)
    return instance


def test_extract_think_blocks_returns_reasoning_and_clean_text():
    reasoning, cleaned = LocalLLM._extract_think_blocks(
        "<think>I should check who is in the room first.</think>I see one person sitting."
    )
    assert reasoning == "I should check who is in the room first."
    assert "<think>" not in cleaned
    assert "I see one person sitting." in cleaned


def test_extract_think_blocks_unclosed_tail():
    reasoning, cleaned = LocalLLM._extract_think_blocks("<think>reasoning cut off mid-sentence")
    assert reasoning == "reasoning cut off mid-sentence"
    assert "<think>" not in cleaned


def test_extract_think_blocks_no_thinking():
    reasoning, cleaned = LocalLLM._extract_think_blocks("Just a plain answer.")
    assert reasoning == ""
    assert cleaned == "Just a plain answer."


def test_sanitize_strips_think_blocks_for_speech():
    sanitized = sanitize("<think>secret plan: watch the door</think>Hello there.")
    assert "secret plan" not in sanitized
    assert "Hello there." in sanitized


def test_frame_memory_records_and_renders_oldest_first():
    llm = _bare_llm()
    llm._record_frame_observation("The room is bright with one person.", "I see one person reading.")
    llm._record_frame_observation("A cat walks across the floor.", "")
    block = llm._frame_memory_block()
    assert block.startswith("What you have seen recently (oldest first):")
    assert block.index("bright with one person") < block.index("cat walks")
    assert "you noted: I see one person reading." in block


def test_frame_memory_dedups_similar_observations():
    llm = _bare_llm()
    llm._record_frame_observation("I see 1 person in a moderately lit environment.", "One person sitting.")
    llm._record_frame_observation("I see 1 person in a moderately lit environment, very still.", "Still one person.")
    assert len(llm._frame_memory) == 1
    assert llm._frame_memory[0]["response"] == "Still one person."


def test_frame_memory_keeps_entries_when_response_changes():
    llm = _bare_llm()
    llm._record_frame_observation("I see 1 person in a moderately lit environment.", "One person is reading a book.")
    llm._record_frame_observation("I see 1 person in a moderately lit environment, waving.", "The person is waving at the camera.")
    assert len(llm._frame_memory) == 2
    assert llm._frame_memory[0]["response"] == "One person is reading a book."
    assert llm._frame_memory[1]["response"] == "The person is waving at the camera."


def test_frame_memory_trims_after_refresh_grows_entry():
    llm = _bare_llm(frame_memory_max_tokens=64)
    llm._record_frame_observation("A red bicycle leans against a brick wall.", "Short.")
    llm._record_frame_observation(
        "A red bicycle leans against a brick wall, very still.",
        "A long detailed observation about the bicycle and the wall and everything around it in the room.",
    )
    assert llm._estimate_tokens(llm._frame_memory_block()) <= 64


@pytest.mark.asyncio
async def test_generate_vision_trims_history_to_fit():
    llm = _bare_llm()
    llm.model = object()
    llm._vision_available = True
    llm.vision_show_thinking = False
    llm.vision_thinking_max_tokens = 128
    llm._cfg = {"llm": {"n_ctx": 1200}}
    llm._run_vision_completion = lambda messages, max_tokens: "I see one person sitting."
    posts = []

    class FakeBridge:
        async def post(self, payload):
            posts.append(payload)

    llm.bridge = FakeBridge()
    llm._record_frame_observation(
        "I see 1 person in a moderately lit environment with a red bicycle leaning against the brick wall nearby.",
        "I see one person sitting near a red bicycle leaning against a brick wall in a moderately lit room.",
    )
    assert llm._frame_memory
    result = await llm._generate_vision("data:image/jpeg;base64,xxx", "Describe what you see.", max_tokens=60)
    assert result == "I see one person sitting."
    assert llm._frame_memory == []
    assert not [p for p in posts if p.get("type") == "event"]


@pytest.mark.asyncio
async def test_generate_vision_still_skips_when_bare_frame_too_big():
    llm = _bare_llm()
    llm.model = object()
    llm._vision_available = True
    llm.vision_show_thinking = False
    llm.vision_thinking_max_tokens = 128
    llm._cfg = {"llm": {"n_ctx": 1000}}
    llm._run_vision_completion = lambda messages, max_tokens: "I see one person sitting."
    posts = []

    class FakeBridge:
        async def post(self, payload):
            posts.append(payload)

    llm.bridge = FakeBridge()
    llm._record_frame_observation("I see 1 person in a moderately lit environment.", "I see one person sitting.")
    result = await llm._generate_vision("data:image/jpeg;base64,xxx", "Describe what you see.", max_tokens=60)
    assert result == ""
    assert any("vision skipped" in p.get("message", "") for p in posts if p.get("type") == "event")


def test_frame_memory_caps_items():
    llm = _bare_llm(frame_memory_items=2)
    llm._record_frame_observation("A red bicycle leans against a brick wall.", "")
    llm._record_frame_observation("A dog sleeps on a blue rug.", "")
    llm._record_frame_observation("Rain streaks the tall window.", "")
    assert len(llm._frame_memory) == 2
    assert "bicycle" not in llm._frame_memory_block()
    assert "Rain streaks" in llm._frame_memory_block()


def test_frame_memory_caps_tokens():
    llm = _bare_llm(frame_memory_max_tokens=200)
    for i in range(5):
        llm._record_frame_observation(f"Observation {i}: a long detailed scene with many objects and people moving around the room.", "")
    assert llm._estimate_tokens(llm._frame_memory_block()) <= 200


def test_vision_budget_counts_long_prompt():
    llm = _bare_llm()
    assert llm._vision_fits_budget("Describe what you see.", 60) is True
    assert llm._vision_fits_budget("x" * 70000, 60) is False


@pytest.mark.asyncio
async def test_generate_vision_includes_frame_memory_and_posts_reasoning():
    llm = _bare_llm()
    llm.model = object()
    llm._vision_available = True
    llm.vision_show_thinking = True
    llm.vision_thinking_max_tokens = 128
    captured = {}

    def fake_completion(messages, max_tokens):
        captured["messages"] = messages
        captured["max_tokens"] = max_tokens
        return "<think>I see a person sitting down.</think>I see one person sitting."

    llm._run_vision_completion = fake_completion
    posts = []

    class FakeBridge:
        async def post(self, payload):
            posts.append(payload)

    llm.bridge = FakeBridge()
    llm._record_frame_observation(
        "I see 1 person in a moderately lit environment.", "I see one person sitting."
    )
    result = await llm._generate_vision("data:image/jpeg;base64,xxx", "Describe what you see.", max_tokens=60)
    assert result == "I see one person sitting."
    user_text = captured["messages"][1]["content"][1]["text"]
    assert "What you have seen recently" in user_text
    assert "I see 1 person" in user_text
    assert captured["max_tokens"] == 60 + 128
    thought_texts = [p.get("text", "") for p in posts if p.get("type") == "thought"]
    assert any(t.startswith("Thinking:") and "sitting down" in t for t in thought_texts)
    assert "sitting down" not in result


@pytest.mark.asyncio
async def test_generate_vision_no_thinking_when_disabled():
    llm = _bare_llm()
    llm.model = object()
    llm._vision_available = True
    llm.vision_show_thinking = False
    llm.vision_thinking_max_tokens = 128
    captured = {}

    def fake_completion(messages, max_tokens):
        captured["max_tokens"] = max_tokens
        return "I see one person sitting."

    llm._run_vision_completion = fake_completion
    posts = []

    class FakeBridge:
        async def post(self, payload):
            posts.append(payload)

    llm.bridge = FakeBridge()
    result = await llm._generate_vision("data:image/jpeg;base64,xxx", "Describe what you see.", max_tokens=60)
    assert result == "I see one person sitting."
    assert captured["max_tokens"] == 60
    assert not [p for p in posts if p.get("type") == "thought"]


def _minimal_cfg(vision_overrides=None):
    cfg = {
        "paths": {"gguf_path": "H:/Models/Qwen3VL-8B-Instruct-Q4_K_M.gguf", "mmproj_path": ""},
        "llm": {"temperature": 0.7, "top_p": 0.95, "n_ctx": 16384, "n_threads": 8},
        "vision": {},
    }
    if vision_overrides:
        cfg["vision"].update(vision_overrides)
    return cfg


def test_init_wires_frame_memory_and_thinking_defaults():
    import asyncio

    llm = LocalLLM(_minimal_cfg(), None, None, None, loop=asyncio.new_event_loop(), load_immediately=False)
    assert llm.vision_show_thinking is True
    assert llm.vision_thinking_max_tokens == 128
    assert llm.show_thinking is True
    assert llm.frame_memory_items == 5
    assert llm.frame_memory_max_tokens == 512
    assert llm._frame_memory == []
    assert llm._last_frame_description == ""
    assert llm._last_frame_image_url is None


def test_init_respects_show_thinking_false():
    import asyncio

    llm = LocalLLM(
        _minimal_cfg({"show_thinking": False}),
        None, None, None, loop=asyncio.new_event_loop(), load_immediately=False,
    )
    assert llm.vision_show_thinking is False


# ---------------------------------------------------------------------------
# Autonomous idle loop (think / speak / act)
# ---------------------------------------------------------------------------


class _FakeTTS:
    def __init__(self):
        self.spoken = []

    async def speak(self, text):
        self.spoken.append(text)


def _idle_llm(**overrides):
    """_bare_llm rigged for idle-loop tests: lanes, context, bridge, tts."""
    posts = []
    tts = _FakeTTS()

    class FakeBridge:
        async def post(self, payload):
            posts.append(payload)

    base = dict(
        recent_context=[],
        max_context_items=20,
        thinking_prompt="",
        prompt_style="plain",
        tools_enabled=False,
        tools=None,
        person_tracker=None,
        memory=None,
        tts=tts,
        bridge=FakeBridge(),
        _lanes={"chat": False, "vision": False, "autonomous": False},
        _lane_lock=asyncio.Lock(),
        last_thought=0.0,
        thought_cooldown=30,
        autonomous_mode=True,
        idle_speak_enabled=True,
        idle_min_speak_interval=300,
        _last_autonomous_speech=0.0,
        _last_audio_evidence="",
        _last_audio_at=0.0,
        _audio_evidence_ttl_s=60,
        scene_digest_seconds=0,
        _last_frame_description="",
        _last_frame_image_url="data:image/jpeg;base64,xxx",
        _vision_available=True,
    )
    base.update(overrides)
    llm = _bare_llm(**base)
    llm._idle_posts = posts
    llm._idle_tts = tts
    return llm


def test_parse_idle_decision_say_think_unmarked():
    assert LocalLLM._parse_idle_decision("SAY: Hello there.") == ("speak", "Hello there.")
    assert LocalLLM._parse_idle_decision("THINK: Quiet room.") == ("think", "Quiet room.")
    assert LocalLLM._parse_idle_decision("say: lower case works") == ("speak", "lower case works")
    assert LocalLLM._parse_idle_decision("Just looking around.") == ("think", "Just looking around.")
    assert LocalLLM._parse_idle_decision("") == ("think", "")
    assert LocalLLM._parse_idle_decision("SAY: First line\nTHINK: second line") == ("speak", "First line")


def test_idle_speech_allowed_respects_toggle_and_interval():
    llm = _idle_llm(idle_speak_enabled=False)
    assert llm._idle_speech_allowed(1000.0) is False
    llm = _idle_llm(idle_speak_enabled=True, idle_min_speak_interval=300, _last_autonomous_speech=1000.0)
    assert llm._idle_speech_allowed(1100.0) is False
    assert llm._idle_speech_allowed(1300.0) is True
    llm = _idle_llm(idle_speak_enabled=True, idle_min_speak_interval=0, _last_autonomous_speech=1000.0)
    assert llm._idle_speech_allowed(1000.0) is True


def test_idle_speech_allowed_suppressed_when_chat_lane_busy():
    llm = _idle_llm()
    llm._lanes["chat"] = True
    assert llm._idle_speech_allowed(10000.0) is False


def test_init_wires_idle_loop_config():
    cfg = _minimal_cfg()
    cfg["autonomous"] = {
        "enabled": False,
        "thought_cooldown": 45,
        "speak_enabled": False,
        "min_speak_interval_seconds": 60,
    }
    llm = LocalLLM(cfg, None, None, None, loop=asyncio.new_event_loop(), load_immediately=False)
    assert llm.autonomous_mode is False
    assert llm.thought_cooldown == 45
    assert llm.idle_speak_enabled is False
    assert llm.idle_min_speak_interval == 60
    assert llm._last_autonomous_speech == 0.0


def test_init_idle_loop_defaults():
    llm = LocalLLM(_minimal_cfg(), None, None, None, loop=asyncio.new_event_loop(), load_immediately=False)
    assert llm.autonomous_mode is True
    assert llm.thought_cooldown == 30
    assert llm.idle_speak_enabled is True
    assert llm.idle_min_speak_interval == 300
    assert llm._last_autonomous_speech == 0.0
    assert llm._last_audio_evidence == ""
    assert llm._last_audio_at == 0.0


def test_idle_audio_evidence_fresh_stale_empty():
    llm = _idle_llm(_last_audio_evidence="someone knocked", _last_audio_at=1000.0)
    assert llm._idle_audio_evidence(1010.0) == "someone knocked"
    assert llm._idle_audio_evidence(1070.0) == ""
    llm = _idle_llm(_last_audio_evidence="", _last_audio_at=1000.0)
    assert llm._idle_audio_evidence(1010.0) == ""


def test_build_prompt_autonomous_act_includes_tools_and_silent_framing():
    class FakeTools:
        def get_tools_prompt(self):
            return 'Use TOOL_CALL: {"tool": "recall_person", "args": {}} to recall.'

    llm = _bare_llm(
        recent_context=[],
        max_context_items=20,
        thinking_prompt="",
        prompt_style="plain",
        tools_enabled=True,
        tools=FakeTools(),
    )
    prompt = llm._build_prompt("A new face at the door.", "autonomous_act")
    assert "A new face at the door." in prompt
    assert "TOOL_CALL" in prompt
    assert "nothing you write here will be spoken aloud" in prompt


@pytest.mark.asyncio
async def test_idle_decide_vision_say_speaks():
    llm = _idle_llm()
    llm._generate_vision = AsyncMock(return_value="SAY: Hey, good to see you.")
    llm._generate = AsyncMock(return_value="should not be called")
    await llm.autonomous_thought()
    speak_posts = [p for p in llm._idle_posts if p.get("type") == "speak"]
    assert len(speak_posts) == 1
    assert speak_posts[0]["text"] == "Hey, good to see you."
    assert llm._idle_tts.spoken == ["Hey, good to see you."]
    llm._generate.assert_not_called()


@pytest.mark.asyncio
async def test_idle_decide_vision_say_held_by_speech_gate_posts_thought():
    llm = _idle_llm(_last_autonomous_speech=time.time())
    llm._generate_vision = AsyncMock(return_value="SAY: Hello again.")
    llm._generate = AsyncMock(return_value="unused")
    await llm.autonomous_thought()
    assert not [p for p in llm._idle_posts if p.get("type") == "speak"]
    assert llm._idle_tts.spoken == []
    thoughts = [p.get("text", "") for p in llm._idle_posts if p.get("type") == "thought"]
    assert any("Hello again." in t for t in thoughts)


@pytest.mark.asyncio
async def test_idle_decide_vision_think_posts_thought_and_runs_act_phase():
    llm = _idle_llm()
    llm._generate_vision = AsyncMock(return_value="THINK: A new face at the door.")
    llm._generate = AsyncMock(return_value="No match in memory; nothing to store.")
    await llm.autonomous_thought()
    thoughts = [p.get("text", "") for p in llm._idle_posts if p.get("type") == "thought"]
    assert any("A new face at the door." in t for t in thoughts)
    assert any("No match in memory" in t for t in thoughts)
    assert not [p for p in llm._idle_posts if p.get("type") == "speak"]
    assert llm._idle_tts.spoken == []
    # The act phase saw the vision observation.
    act_prompt = llm._generate.call_args[0][0]
    assert "A new face at the door." in act_prompt


@pytest.mark.asyncio
async def test_idle_act_forces_say_output_silent():
    llm = _idle_llm()
    llm._generate_vision = AsyncMock(return_value="THINK: Someone new.")
    llm._generate = AsyncMock(return_value="SAY: Hello, stranger!")
    await llm.autonomous_thought()
    assert not [p for p in llm._idle_posts if p.get("type") == "speak"]
    assert llm._idle_tts.spoken == []
    thoughts = [p.get("text", "") for p in llm._idle_posts if p.get("type") == "thought"]
    assert any("Hello, stranger!" in t for t in thoughts)


@pytest.mark.asyncio
async def test_idle_decide_unmarked_vision_stays_quiet():
    llm = _idle_llm()
    llm._generate_vision = AsyncMock(return_value="Just looking around.")
    llm._generate = AsyncMock(return_value="Quiet room.")
    await llm.autonomous_thought()
    assert not [p for p in llm._idle_posts if p.get("type") == "speak"]
    thoughts = [p.get("text", "") for p in llm._idle_posts if p.get("type") == "thought"]
    assert any("Just looking around." in t for t in thoughts)


@pytest.mark.asyncio
async def test_idle_decide_text_fallback_when_no_frame():
    llm = _idle_llm(_last_frame_image_url=None)
    llm._generate = AsyncMock(return_value="THINK: Quiet room, nothing happening.")
    await llm.autonomous_thought()
    thoughts = [p.get("text", "") for p in llm._idle_posts if p.get("type") == "thought"]
    assert any("Quiet room" in t for t in thoughts)
    assert not [p for p in llm._idle_posts if p.get("type") == "speak"]
    # The text fallback prompt carries the SAY:/THINK: decision instruction.
    assert "SAY:" in llm._generate.call_args[0][0]


@pytest.mark.asyncio
async def test_autonomous_thought_skips_when_chat_lane_busy():
    llm = _idle_llm()
    llm._lanes["chat"] = True
    llm._generate_vision = AsyncMock(return_value="SAY: Hi.")
    llm._generate = AsyncMock(return_value="Hi.")
    await llm.autonomous_thought()
    assert llm._idle_posts == []
    llm._generate_vision.assert_not_called()
    llm._generate.assert_not_called()


@pytest.mark.asyncio
async def test_idle_decide_vision_prompt_carries_fresh_audio_evidence():
    llm = _idle_llm(_last_audio_evidence="someone knocked twice", _last_audio_at=time.time())
    llm._generate_vision = AsyncMock(return_value="THINK: A knock at the door.")
    llm._generate = AsyncMock(return_value="Probably a visitor.")
    await llm.autonomous_thought()
    vision_prompt = llm._generate_vision.call_args[0][1]
    assert "someone knocked twice" in vision_prompt
    assert "cross-wired" in vision_prompt


@pytest.mark.asyncio
async def test_idle_decide_vision_prompt_drops_stale_audio_evidence():
    llm = _idle_llm(_last_audio_evidence="someone knocked twice", _last_audio_at=0.0)
    llm._generate_vision = AsyncMock(return_value="THINK: Quiet room.")
    llm._generate = AsyncMock(return_value="Quiet room.")
    await llm.autonomous_thought()
    vision_prompt = llm._generate_vision.call_args[0][1]
    assert "someone knocked twice" not in vision_prompt


@pytest.mark.asyncio
async def test_idle_decide_text_fallback_carries_audio_when_no_frame():
    llm = _idle_llm(
        _last_frame_image_url=None,
        _last_audio_evidence="a dog barking outside",
        _last_audio_at=time.time(),
    )
    llm._generate = AsyncMock(return_value="THINK: A dog barking outside.")
    await llm.autonomous_thought()
    text_prompt = llm._generate.call_args[0][0]
    assert "a dog barking outside" in text_prompt


@pytest.mark.asyncio
async def test_idle_decide_say_suppressed_when_chat_starts_mid_call():
    # Codex P2 on PR #87: chat input arriving during the idle vision call
    # must not end up as speech queued over (or ahead of) the live reply.
    llm = _idle_llm()

    async def vision_with_chat_arriving(*args, **kwargs):
        llm._lanes["chat"] = True  # user started talking mid-call
        return "SAY: Hey there."

    llm._generate_vision = vision_with_chat_arriving
    llm._generate = AsyncMock(return_value="unused")
    await llm.autonomous_thought()
    assert not [p for p in llm._idle_posts if p.get("type") == "speak"]
    assert llm._idle_tts.spoken == []
    thoughts = [p.get("text", "") for p in llm._idle_posts if p.get("type") == "thought"]
    assert any("Hey there." in t for t in thoughts)
