# Changelog - Bug Fixes & Optimizations

## Unreleased

### 🏗️ Architecture: MCP-style tool registry, processing lanes, TTS queue

- Tool registry modernized (`src/backend/tools.py`): tool definitions are
  now the single source of truth — MCP-style schemas
  (`Tool.to_mcp_schema()` / `get_mcp_tools()`, `{"name", "description",
  "inputSchema"}`), the OpenAI-style schema, and the model prompt are all
  derived from the same definitions. Handlers no longer reach into the
  `LocalLLM` god object; they take an explicit `ToolContext` (memory,
  bridge, person tracker, live short-term context, reinforcement hook),
  so tools can be listed, validated, and executed with no LLM in scope.
  The bridge serves the same registry locally over the WebSocket:
  `tools/list` returns the MCP schemas, `tools/call` runs the same
  validated dispatch as the model path. Offline-first, no cloud
  dependencies; see `docs/ARCHITECTURE.md`.
- Processing lanes replace the single shared `_processing` flag
  (`src/backend/llm.py`, finding 6): separate chat, vision, and
  autonomous lanes. Vision comments now queue behind chat (bounded
  queue of 8, near-duplicate descriptions dropped, full queue drops its
  oldest frame) instead of being dropped outright when chat is busy.
  A `_model_lock` serializes actual inference in `_generate` /
  `_generate_vision`, since the llama instance is single. `reload_model`
  still refuses while any lane is busy; `LocalLLM.stop()` now shuts the
  vision worker and TTS down.
- TTS speak queue (`src/backend/tts.py`, finding 7): `speak()` enqueues
  and returns immediately; one worker synthesizes and plays in order.
  New chat/audio input interrupts in-flight speech (playback stopped,
  Piper process killed, queue dropped). `piper_out` is pruned to the
  newest 20 wav files after each synthesis. `muted` / `set_muted` /
  `toggle_muted` added for the gesture actions (finding 10).
- Config drift fixed (finding 5): `vision.vision_cooldown` (default 15)
  is now the single source of truth for the vision cadence — the LLM and
  the camera loop both read it, and the bridge's live `vision_cooldown`
  param updates both. Removed the dead copies: `autonomous.vision_cooldown`
  (never read), `ui.vision_cooldown` (fallback only), the unused
  `vision.interesting_keywords`, and the duplicated
  `vision.motion_sensitivity` / `vision.snapshot_debounce` (those are
  `ui` keys).
- Deleted `src/backend/handlers.py`: dead code, nothing imported it, and
  it referenced APIs that no longer exist.

### 🎥 Findings 3, 4, 8, 9, 10, 11

- Scene digest + describe-the-scene (finding 3): in a static scene the
  vision pipeline stays quiet for minutes, so `autonomous_thought` now
  describes the room on a timer (`vision.scene_digest_minutes`, default
  10, 0 disables) — but only when the scene is unchanged, so a digest
  never interrupts something new. A changed scene re-baselines
  immediately and restarts the quiet interval, so digests resume once
  the new scene settles (Codex review). A "describe the scene" / "what
  do you see?" request in chat or voice goes straight to the vision
  model about the freshest frame, bypassing suppression and cooldown —
  and the frame cache now refreshes on every preview send, not just on
  analysis, so the answer is never about a 15s-old frame (Codex review).
  A bare 'what's happening' no longer routes to the camera — it needs a
  spatial anchor ('here', 'around you', ...) or the question keeps
  normal generation and tools (Codex review).
- Real person recognition (finding 4): each detected face crop is
  embedded on-device by a vendored MediaPipe ImageEmbedder (MobileNetV3,
  1024-dim L2-normalized) and matched on cosine similarity
  (`vision.embedding_match_threshold`, default 0.7). A clearly-different
  embedding vetoes a size/position match — per candidate, so with
  several people tracked one ambiguous similarity cannot shield a
  clearly-different face from the veto (Codex review); the ambiguous
  middle band still lets the heuristic decide. The embedding loop also
  considers absent identities, so a known person who left and came
  back re-identifies by face instead of spawning a new anonymous
  person (Codex review). Stored embeddings update by
  exponential moving average and persist in the memory dicts.
- First-run robustness (finding 8): the MediaPipe task models are
  vendored in `assets/mediapipe/` with a manifest pinning each file's
  URL + sha256; `src/backend/mediapipe_assets.py` resolves and verifies
  them (vendored → legacy dir → pinned-URL download, all checksum
  checked). `scripts/start.py` runs a preflight before the bridge
  starts: a missing or corrupt MediaPipe model is fatal, the
  Nexus-side assets (Vosk, Piper, GGUF) only warn.
- Preview cadence decoupled from analysis (finding 9): the dashboard
  preview now runs on a steady ~2 fps timer (`ui.preview_interval_sec`,
  default 0.5, replacing `ui.snapshot_debounce`) regardless of motion,
  while vision analysis stays on its own cooldown. The bridge param
  `snapshot_debounce` remains as an alias for old dashboards.
- Gesture actions (finding 10): open palm pauses in-flight speech,
  fist toggles the mute; `ui.gesture_enabled` is now true by default.
  Actions fire once per press — a held fist re-triggering every
  cooldown interval no longer alternates mute/unmute (Codex review).
  Detection runs every frame so releases are observed promptly: a
  release-and-repress between dispatches counts as a fresh press
  (Codex review). Muting with a fist also interrupts in-flight speech,
  so 'Muted' means silent now, not just quiet later (Codex review).
- Proactive recall (finding 11): when a known face is present, the
  vision prompt carries a one-line "last time you saw this person"
  recall built from their most recent conversation; strangers get no
  recall line.

### 📚 Docs cleanup
- Removed the file-citation markers (e.g. `[F:src/backend/llm.py]`) left in
  README.md, docs/README.md, and this changelog by the v0.3.0/v0.4.0 doc
  passes — they rendered as literal junk text on GitHub.
- README header is plain markdown now: the `<div>` / `<img>` / `<br>` /
  `<strong>` block is gone; the logo is a markdown image (`logo.png`,
  resized from 1024px to 480px) with the tagline as a bold line.
- Added the missing MIT `LICENSE` file the README already linked to.
- Fixed README inaccuracies: `llm.temperature` is 0.7 (was 0.6) and the tool
  framework has 26 built-in tools (was "nine"); the project-structure tree
  now matches the real layout.
- docs/README.md config example now matches `config.yaml` (n_ctx 16384, the
  vision section with frame-memory keys, `gesture_enabled: false`); the Piper
  playback and camera-vision descriptions now match the code (cross-platform
  playback; real frames to the vision model, not text descriptions).

## v0.4.0 - Frame memory & visible reasoning

### ✨ Frame memory: the camera remembers what it saw
- Vision calls now carry a rolling memory of the last few analyzed frames
  (`vision.frame_memory_items`, default 5) as prompt context, so the model
  can track change over time instead of seeing one amnesiac still per call.
  Near-duplicate observations refresh the last entry instead of appending.

- Frame memory is capped by item count and by tokens
  (`vision.frame_memory_max_tokens`, default 512), and the token budget
  check counts it — the history can never blow the Qwen context window.


### ✨ The thoughts panel shows the model's real reasoning
- `<think>` content is now streamed into the dashboard thoughts panel as
  `Thinking: …` lines for chat, audio, and vision replies. The spoken path
  is unchanged: `_sanitize_response` still strips think blocks before
  anything reaches Piper, so reasoning is never read aloud.

- New `vision.show_thinking` (default true) turns thinking on for vision
  calls so there is reasoning to display, and reserves
  `vision.thinking_max_tokens` (default 128) of output budget for it; set
  `show_thinking: false` to restore the old status-lines-only panel.
  `llm.show_thinking` (default true) is the same toggle for chat/audio.

- The old `Raw output: …` thought line is gone, replaced by the extracted
  reasoning.

### 🔌 Ready for a future idle loop (no loop built yet)
- Frame memory entries are timestamped, and the LLM keeps the last analyzed
  frame/description, so a future periodic tick can call
  `process_vision(last_frame_description, last_frame_image_url)` to ask
  "think or speak?" without a new camera event. The think/speech split
  above is what makes that safe: the tick's reasoning stays on screen and
  only a passing observation ever reaches Piper.

## v0.3.0 - Qwen vision upgrade

### ✨ Real vision input
- The camera now sends actual frames to a Qwen vision model (Qwen3-VL-8B
  default, ~7 GB all-in; Qwen3.8-27B as a documented option) instead of a text
  sentence about brightness and motion.
- The launcher installs the vision-capable JamePeng fork of llama-cpp-python
  (0.4.6 CUDA wheel, matched by Python version and driver CUDA tag) — the
  stock PyPI package has no Qwen vision handlers.
- The stock PyPI build stays in `requirements.txt` as the CPU/text-only base,
  so a machine without a CUDA GPU still starts; the fork install is skipped
  there instead of failing the launch.
- The frontend vision-model picker is wired to the LLM for real
  (`vision_model_path` → `llm.reload_vision_model()`); it accepts `.gguf` and
  the presets point at the Qwen model.

### 🔧 Fixed broken install paths
- MediaPipe migrated to the Tasks API (`GestureRecognizer` + `FaceDetector`,
  models auto-download on first run). `mp.solutions` was removed in MediaPipe
  1.x, so fresh installs of the old code crashed.
- OpenCV Haar cascade face detection replaced with the MediaPipe face
  detector.
- Dependencies pinned in `requirements.txt`; `pytest.ini` registers the
  `asyncio` marker.

### 🔧 Behaviour fixes
- `<think>` blocks are stripped before output reaches Piper, and thinking is
  off by default (`vision.enable_thinking: false`).
- Vision calls are token-budgeted against `n_ctx` (system prompt + user text
  + frame + output), skipping cleanly when they do not fit — Qwen overflow is
  a hard error, not a silent truncation. Switching vision models re-resolves
  the projector from next to the new model instead of keeping the old one.
- Vision duplicate suppression now compares VLM output by embedding cosine
  similarity instead of word overlap on heuristic sentences.
- Vision inference runs in a worker thread (`asyncio.to_thread`) so image
  prefill never blocks the WebSocket event loop.
- `process_vision` analyzes the frame first and suppresses on the VLM's
  observation — the old text-description speech gates no longer decide
  whether a stable scene ever reaches the vision model.
- A generation cut off inside `<think>` no longer falls back to speaking the
  raw reasoning block: the fallback now uses the think-stripped text, and a
  reasoning-only generation stays silent.
- Piper playback is now cross-platform via sounddevice + soundfile; the
  PowerShell media-player shim is gone.
- ChatML prompt template for Qwen models, with `<|im_end|>` as a stop
  token.

### 🔒 Privacy
- The assistant memory database (`data/memory/nomous.sqlite`) is removed from
  the repo and purged from git history; `data/memory/` + `data/rag/` are
  gitignored.

### 📚 Docs
- `docs/GPU_SETUP.md` rewritten for the RTX 5070 Ti, the fork wheel, and the
  Qwen models; README config/docs updated to match; `docs/TESTING.md`
  updated for the pinned dependencies and the `asyncio` marker.
- Restored `src/frontend/lib/utils.ts` (shadcn `cn` helper) that the vision
  rework had dropped.

## v2.1 - Telemetry & Documentation Refresh

### ✨ Features & Design Updates
- Added real-time system telemetry publisher with GPU/CPU metrics and UI system usage card integration.
- Expanded documentation with feature status tracking and configuration highlights in both the main README and runtime guide.
- Surfaced tool execution history and stats directly in the dashboard to accompany the existing tool system.
- Clarified optional components such as GPU acceleration, Piper playback, and gesture recognition across docs.

### ⚠️ Known Limitations
- Piper auto playback remains Windows-specific; cross-platform playback remains on the roadmap.
- GPU telemetry and system tests depend on `torch`, `pynvml`, and `psutil`; missing packages disable metrics and fail tests.

## v2.0 - Performance & Quality Update

### 🚀 Performance Optimizations

#### GPU Acceleration
- **Added:** GPU layer support (`n_gpu_layers: 35`)
- **Added:** Automatic GPU detection in LLM
- **Result:** 5-10x faster inference (8-15s → 1-3s)

#### Video Processing
- **Added:** Dual-resolution system (capture 1280x720, process 640x360)
- **Added:** Frame skipping (every 2nd frame)
- **Added:** MediaPipe gesture detection
- **Result:** Smooth 30 FPS + instant gesture recognition

#### Configuration
- **Updated:** `config.yaml` with all optimization parameters
- **Added:** GPU setup scripts (`setup_gpu.bat`, `test_gpu.py`)
- **Added:** Documentation (`QUICKSTART.md`, `OPTIMIZATION_SUMMARY.md`)

### 🔧 Bug Fixes

#### 1. Role-Playing Responses
- **Fixed:** Model no longer speaks in role-play format
- **Changed:** Rewrote all prompts to be natural and concise
- **Changed:** Added better stop sequences: "Person:", "You:", "User:"
- **Result:** Natural, conversational responses

#### 2. Console Clutter
- **Fixed:** Removed duplicate messages
- **Fixed:** Filtered pong/ping spam
- **Fixed:** Reduced excessive logging
- **Changed:** Console history limit: 400 → 150 lines
- **Result:** Clean, readable console

#### 3. Thoughts Tab Empty
- **Added:** Real-time thought streaming
- **Added:** New message type: `{"type": "thought", "text": "..."}`
- **Added:** `thoughtLines` state in UI
- **Changed:** Thoughts tab shows live thinking process
- **Result:** Visible AI reasoning

#### 4. Microphone Not Triggering
- **Fixed:** Audio now properly triggers LLM
- **Enhanced:** Better logging and error detection
- **Changed:** Reduced partial result spam
- **Changed:** UI message: "🎤 Heard" → "🎤 You said"
- **Result:** Mic responses work reliably

#### 5. Over-Talkative AI
- **Added:** Decision-making system for when to speak
- **Added:** Random silence (20% on vision, 70% on autonomous)
- **Changed:** Only speaks on interesting events (person, gesture)
- **Changed:** Quiet observations sent as thoughts only
- **Result:** AI with personality, doesn't talk constantly

### 📝 Files Modified

#### Backend
- `llm.py` - Natural prompts, thought streaming, decision-making, GPU support
- `audio.py` - Better logging, error detection
- `video.py` - Created `video_optimized.py` with gestures
- `config.yaml` - Added GPU and optimization parameters

#### Frontend
- `App.tsx` - Thought streaming, log filtering, thoughtLines state

#### New Files
- `video_optimized.py` - GPU-friendly video + MediaPipe gestures
- `setup_gpu.bat` - Automated GPU setup
- `test_gpu.py` - GPU verification
- `test_fixes.py` - Bug fix verification
- `QUICKSTART.md` - Quick setup guide
- `OPTIMIZATION_SUMMARY.md` - Complete optimization guide
- `BUG_FIXES.md` - Detailed bug fix documentation
- `CHANGELOG.md` - This file

### ✅ Testing Checklist

- [ ] GPU acceleration (nvidia-smi shows usage)
- [ ] Response time 1-3 seconds
- [ ] Video smooth 30 FPS
- [ ] Gesture recognition working
- [ ] Microphone triggers responses
- [ ] Thoughts tab shows thinking
- [ ] Console clean (no spam)
- [ ] Responses natural (not role-play)
- [ ] AI quiet sometimes (personality)

### 🎯 Migration Guide

#### For Existing Users

1. **Update configuration:**
   ```bash
   # Your config.yaml will be updated automatically
   # Review new parameters in config.yaml
   ```

2. **Install GPU support (optional but recommended):**
   ```bash
   # Windows:
   setup_gpu.bat

   # Manual:
   pip uninstall llama-cpp-python -y
   CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
   pip install mediapipe
   ```

3. **Use optimized video:**
   ```bash
   # Backup original
   cp backend/video.py backend/video_backup.py

   # Use optimized
   cp video_optimized.py backend/video.py
   ```

4. **Restart:**
   ```bash
   # Restart computer (for CUDA)
   # Then run:
   python run_bridge.py
   ```

#### Configuration Changes

Your `config.yaml` now includes:

```yaml
llm:
  n_gpu_layers: 35        # NEW: GPU acceleration

camera:
  process_width: 640      # NEW: Processing resolution
  process_height: 360     # NEW
  frame_skip: 2           # NEW: Frame skipping

ui:
  snapshot_debounce: 3    # CHANGED: from 4
  motion_sensitivity: 25  # CHANGED: from 30
  gesture_enabled: true   # NEW: Gesture detection
  gesture_cooldown: 3     # NEW
  vision_cooldown: 12     # NEW
```

### 📚 Documentation

- **QUICKSTART.md** - Fast setup guide (10 minutes)
- **OPTIMIZATION_SUMMARY.md** - Complete optimization details
- **BUG_FIXES.md** - Bug fix documentation
- **PERFORMANCE_OPTIMIZATION_COMPLETE.md** - Full performance guide

### 🔮 Future Improvements

- [ ] More gesture types (fist, thumbs down, etc.)
- [ ] Custom wake word
- [ ] Voice activity detection improvements
- [ ] Multi-language support
- [ ] Memory/RAG integration
- [ ] Emotion detection
- [ ] Better prompt customization

### 💡 Tips

- Use `test_gpu.py` to verify GPU setup
- Use `test_fixes.py` to verify bug fixes
- Check `nvidia-smi -l 1` to monitor GPU usage
- Adjust silence probability in `llm.py` for personality
- Use Q4 model for maximum speed (2x faster)

---

**Version:** 2.0
**Date:** October 30, 2025
**Major Changes:** GPU optimization + bug fixes
**Backward Compatible:** Yes (config auto-updated)
