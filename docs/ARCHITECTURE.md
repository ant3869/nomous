# Architecture

Nomous is a local, offline-first multimodal agent: a Qwen vision model
running under llama.cpp, a camera loop, Vosk speech-to-text, Piper
text-to-speech, a SQLite memory store, and a React dashboard, all wired
together over a local WebSocket bridge (`scripts/run_bridge.py`).

This doc describes the layering as of v0.5.0 — what calls what, where the
boundaries are, and what is deliberately *not* built yet.

## Layers

```
dashboard (frontend/) ──WebSocket──► bridge (scripts/run_bridge.py)
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
              reasoning core      tool layer            I/O
              src/backend/llm.py  src/backend/tools.py  camera / mic / TTS
              LocalLLM            ToolExecutor          video.py, audio.py,
                                  ToolContext           tts.py, person_tracker.py
```

- **Reasoning core — `LocalLLM` (`src/backend/llm.py`).** Owns the model
  instance, prompt building, the think/speech split, frame memory, and
  the processing lanes. It does *not* own tool implementations or I/O
  devices; it reaches them through the tool registry and the bridge.
  It is still a large class (model loading, vision, chat, audio,
  autonomous thought, person tracking, context memory) — splitting it
  into a small core + services is the known next structural step, but
  it was deliberately *not* done in v0.5.0: the lanes + registry give
  the same safety with far less churn. See "Known gaps".
- **Tool layer — `ToolExecutor` + `ToolContext` (`src/backend/tools.py`).**
  The registry. See "Tools" below.
- **I/O — `CameraLoop`, `audio.py`, `PiperTTS`, `PersonTracker`.**
  Sensors and actuators. They never call the model directly; the camera
  loop hands frames/descriptions to `LocalLLM.process_vision`, audio
  hands transcripts to `LocalLLM.process_audio`, and speech goes out
  through `PiperTTS.speak`. The bridge (`scripts/run_bridge.py`) owns
  the WebSocket and routes dashboard messages to these pieces.

## Tools

Tool definitions are the single source of truth. Each `Tool` carries a
name, description, typed parameters, a handler, a category, and a return
description. From that one definition the registry derives:

- **MCP-style schemas** — `Tool.to_mcp_schema()` /
  `ToolExecutor.get_mcp_tools()` return
  `{"name", "description", "inputSchema"}` (Model Context Protocol
  shape). Local only; there is no MCP network transport in this repo.
- **OpenAI-style schemas** — `get_tools_schema()` derives the
  `{"name", "description", "parameters"}` shape from the same MCP
  definitions, so the two can never disagree about a parameter.
- **The model prompt** — `get_tools_prompt()` renders the same registry
  into the `TOOL_CALL: {"tool": ..., "args": {...}}` instructions the
  local model follows. The model has no native function-calling path
  yet (see "Known gaps"); the text protocol is parsed by
  `parse_tool_calls`, validated/coerced by `_validate_arguments`, and
  dispatched by `execute_tool`, which records every execution in a
  bounded history (`execution_history`, also exposed as tools).

Handlers never touch the LLM object. They receive a `ToolContext`
(memory, bridge, person tracker, the live short-term-context list, a
reinforcement hook) built by `ToolContext.from_llm(llm)` — or passed
directly when there is no LLM in scope. That is what lets the bridge
serve tools without a model round-trip:

- `tools/list` → `{"type": "tools_list", "tools": [...]}` (MCP schemas)
- `tools/call` → `{"type": "tool_result", ...}` (same validated
  dispatch + execution payload the model path produces)

## Processing lanes

The old code had one shared `_processing` flag: any lane running meant
every other lane skipped its work — a vision comment was dropped outright
whenever chat was mid-generation. v0.5.0 replaces it with three lanes
plus a model lock:

- **chat** — `chat()` (dashboard text) and `process_audio()` (voice).
  Taking the lane also interrupts in-flight TTS, so new input always
  wins over old speech. A second chat while the lane is held is still
  skipped, as before.
- **vision** — `process_vision()` only enqueues (bounded queue, 8).
  Near-duplicate descriptions already queued are dropped; a full queue
  drops its oldest frame so the freshest frame wins. A worker drains
  the queue one frame at a time and waits for the chat lane to clear
  before running `_process_vision_impl`, so camera observations queue
  behind conversation instead of being dropped.
- **autonomous** — `autonomous_thought()` (the 30s loop in the bridge)
  skips when the chat lane is busy or its own lane is held.

- **model lock** — `_model_lock` wraps the actual inference calls in
  `_generate` (streaming text) and `_generate_vision` (threaded chat
  completion). The llama instance is single; the lock is what makes
  concurrent lanes safe even when the lane flags say otherwise (e.g.
  vision worker + autonomous thought while chat is idle).

`LocalLLM.stop()` cancels the vision worker and stops TTS.
`reload_model()` refuses while any lane is busy.

## Speech

`PiperTTS.speak()` enqueues and returns immediately. One worker task
synthesizes with Piper and plays through sounddevice in order, so
overlapping chat/vision/autonomous replies queue instead of racing.
`interrupt()` stops current playback, kills the in-flight Piper process,
and drops the queue (generation counter, so a stale in-flight item is
abandoned even mid-synthesis). `muted` drops speech without queuing it
for later; `set_muted` / `toggle_muted` are what the gesture actions
use. `piper_out` is pruned to the newest 20 `speech_*.wav` files after
each synthesis.

## Vision & presence

- **Preview vs analysis.** The dashboard preview is a steady ~2 fps
  timer (`ui.preview_interval_sec`, default 0.5) regardless of motion;
  vision analysis stays on its own cooldown (`vision.vision_cooldown`)
  or a gesture. The two were one gate; they are independent now.
- **Scene digest.** In a static scene `autonomous_thought` describes
  the room every `vision.scene_digest_minutes` (default 10), but only
  when the latest description is similar to the last digested one — a
  changed scene gets its own vision comment instead.
- **Describe the scene.** A chat/voice request matching
  "describe the scene" / "what do you see" / "what's around" bypasses
  the text model and asks the vision model about the freshest frame
  (`describe_scene()`), with no suppression or cooldown.
- **Person recognition.** `CameraLoop` embeds each detected face crop
  (MobileNetV3, 1024-dim) and `PersonTracker` matches on cosine
  similarity (≥ `vision.embedding_match_threshold`, default 0.7), with
  a veto below 0.4 and the size/position heuristic as fallback. Stored
  embeddings are an exponential moving average, so one wobbly frame
  cannot re-anchor an identity. When a known face is present, the
  vision prompt carries a one-line recall of their last conversation.
- **Gesture actions.** open_hand pauses speech (`tts.interrupt()`),
  closed_fist toggles the mute (`tts.toggle_muted()`); dispatched from
  the camera thread, on by default (`ui.gesture_enabled: true`).

## Config

`ConfigLoader` deep-merges `config.yaml` over `DEFAULT` in
`src/backend/config.py`. One rule: **each knob lives in exactly one
place.** `vision.vision_cooldown` (default 15) is the single source for
the vision cadence — `LocalLLM` and `CameraLoop.analysis_cooldown` both
read it, and the bridge's live `vision_cooldown` param updates both.
The dead copies are gone: no `autonomous.vision_cooldown`, no
`ui.vision_cooldown`, no `vision.interesting_keywords` (the VLM-first
design made keyword triggers obsolete), no duplicated
`vision.motion_sensitivity` / `vision.snapshot_debounce` (those are
`ui` keys; the camera loop reads `ui`). `snapshot_debounce` has since
been replaced by `ui.preview_interval_sec` (finding 9): the dashboard
preview now runs on a steady ~2 fps cadence instead of only on motion.

## First run

MediaPipe task models are vendored in `assets/mediapipe/` (pinned URLs
+ sha256 in `assets/mediapipe/manifest.json`, verified by
`src/backend/mediapipe_assets.py`); the legacy `modules/models/mediapipe/`
download dir is only a fallback. The Vosk model and Piper
voice/executable are resolved from `paths` in config, and
`scripts/start.py` runs a preflight that verifies every local asset
before the bridge starts — a missing or corrupt MediaPipe model is
fatal, the Nexus-side assets only warn.

## Known gaps (deliberately not done in v0.5.0)

- **Native function calling.** The model still emits `TOOL_CALL:` text
  that is parsed and dispatched. llama-cpp-python's `create_chat_completion`
  supports native `tools=`; switching the text path to it (and the
  vision path, which already uses chat completion) would remove a whole
  parsing layer, but it changes prompt construction and stop behavior
  for the local Qwen model, so it is a follow-up, not part of this
  refactor. The registry is ready for it: schemas are already
  MCP-shaped and dispatch is already validated.
- **Splitting `LocalLLM`.** It is still a god object (~2000 lines).
  The lanes + registry contain the blast radius; a core/services split
  is the next structural step if it ever earns its churn.
- **`handlers.py` was deleted** (v0.5.0): nothing imported it and it
  referenced APIs that no longer exist (`STTEngine`,
  `LocalLLM(broadcaster)`, `llm.generate`, `tts.say`). `run_bridge.py`
  owns the WebSocket handling.
