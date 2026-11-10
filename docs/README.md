# Nomous Local Autonomy Runtime

This document provides a runtime-centric view of how Nomous stitches together the Python workers and the React dashboard.

## System Components
- **WebSocket bridge** orchestrates messages between the UI and backend subsystems.【F:scripts/run_bridge.py†L1-L213】
- **Local LLM (`llama.cpp`)** streams tokens, surfaces thought events, and can reload models at runtime.【F:src/backend/llm.py†L21-L212】
- **Speech input** converts PCM16 microphone data into transcripts using Vosk and automatically triggers LLM turns.【F:src/backend/audio.py†L33-L118】
- **Piper TTS** synthesises audio and (on Windows) triggers local playback, with toggles exposed through WebSocket parameters.【F:src/backend/tts.py†L29-L138】【F:scripts/run_bridge.py†L72-L157】
- **Camera & vision loop** streams preview frames, detects motion/gestures, and prompts the LLM for descriptions.【F:src/backend/video.py†L28-L123】【F:src/backend/video.py†L360-L437】
- **Memory store** persists interactions to SQLite and rebroadcasts the graph to the dashboard for analytics.【F:src/backend/memory.py†L13-L167】
- **System monitor** pushes CPU/GPU telemetry so the UI can display device health in real time.【F:src/backend/system.py†L1-L152】

## Feature Status Overview
### ✅ Complete
- Tool execution framework with nine built-in tools and UI activity feed.【F:src/backend/tools.py†L32-L187】【F:src/frontend/components/ToolActivity.tsx†L6-L213】
- Gesture-aware camera loop with MediaPipe integration and vision cooldown controls.【F:src/backend/video.py†L28-L123】【F:src/backend/video.py†L360-L437】
- Load-progress overlay for LLM initialisation and model reloads.【F:src/backend/llm.py†L65-L115】【F:src/frontend/App.tsx†L330-L383】
- Behaviour metrics, token throughput charts, and reward accounting surfaced in the dashboard.【F:src/backend/handlers.py†L24-L43】【F:src/frontend/App.tsx†L1614-L2158】

### ⚠️ WIP / Limited
- GPU telemetry requires `torch`, `pynvml`, and `psutil`; missing dependencies disable metrics and break `tests/test_system.py` during import.【F:src/backend/system.py†L1-L152】【F:tests/test_system.py†L1-L120】

### 🚧 Not Implemented Yet
- Additional gesture classes and wake-word activation remain roadmap items from the changelog.【F:docs/CHANGELOG.md†L136-L170】

## Repository Layout
```
nomous/
├─ scripts/
│  ├─ start.py          # Launch backend + React dev server
│  └─ run_bridge.py     # Standalone backend runtime
├─ src/
│  ├─ backend/          # Python workers (LLM, STT, TTS, vision, memory)
│  └─ frontend/         # React dashboard (tabs, charts, controls)
└─ docs/                # Guides, changelog, implementation notes
```

## Key Configuration Fields
```yaml
paths:
  gguf_path: /path/to/model.gguf
  vosk_model_dir: /path/to/vosk-model
  piper_exe: /path/to/piper.exe
  piper_voice: /path/to/voice.onnx
  piper_out_dir: /path/to/output-dir

llm:
  n_ctx: 2048
  n_threads: 4
  temperature: 0.6
  top_p: 0.95
  tools_enabled: true

audio:
  sample_rate: 16000
  sensitivity: 60

camera:
  backend: dshow
  index: 0
  width: 1280
  height: 720
  process_width: 640
  process_height: 360
  frame_skip: 2

ui:
  snapshot_debounce: 4
  motion_sensitivity: 30
  vision_enabled: true
  gesture_enabled: true
  vision_cooldown: 12
  gesture_cooldown: 3
  tts_enabled: true

memory:
  enable: true
  db_path: ./data/memory/nomous.sqlite
```

### Optional Integrations
- Set `llm.n_gpu_layers` and install CUDA-enabled `llama-cpp-python` for GPU offload.【F:src/backend/llm.py†L89-L115】
- Install `mediapipe` to enable gesture recognition inside the camera loop.【F:src/backend/video.py†L37-L64】
- Provide a valid Piper voice/executable pair to enable text-to-speech in addition to subtitles.【F:src/backend/tts.py†L29-L138】
