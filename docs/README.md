# Nomous Local Autonomy Runtime

This document provides a runtime-centric view of how Nomous stitches together the Python workers and the React dashboard.

## System Components
- **WebSocket bridge** orchestrates messages between the UI and backend subsystems.
- **Local LLM (`llama.cpp`)** streams tokens, surfaces the model's real reasoning as thought events (speech is a separate, think-stripped channel), and can reload models at runtime.
- **Speech input** converts PCM16 microphone data into transcripts using Vosk and automatically triggers LLM turns.
- **Piper TTS** synthesises audio and triggers local playback (cross-platform via sounddevice), with toggles exposed through WebSocket parameters.
- **Camera & vision loop** streams preview frames, detects motion/gestures, and sends real frames to the Qwen vision model for analysis.
- **Memory store** persists interactions to SQLite and rebroadcasts the graph to the dashboard for analytics.
- **System monitor** pushes CPU/GPU telemetry so the UI can display device health in real time.

## Feature Status Overview
### ✅ Complete
- Tool execution framework with 26 built-in tools and UI activity feed.
- Gesture-aware camera loop with MediaPipe integration and vision cooldown controls.
- Load-progress overlay for LLM initialisation and model reloads.
- Behaviour metrics, token throughput charts, and reward accounting surfaced in the dashboard.

### ⚠️ WIP / Limited
- GPU telemetry requires `torch`, `pynvml`, and `psutil`; missing dependencies disable metrics and break `tests/test_system.py` during import.

### 🚧 Not Implemented Yet
- Additional gesture classes and wake-word activation remain roadmap items from the changelog.

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
  gguf_path: H:/Models/Qwen3VL-8B-Instruct-Q4_K_M.gguf
  mmproj_path: H:/Models/mmproj-Qwen3VL-8B-Instruct-F16.gguf
  vosk_model_dir: modules/models/vosk-model-small-en-us-0.15
  piper_exe: modules/piper/piper.exe
  piper_voice: modules/piper/voices/en_US-amy-medium.onnx
  piper_out_dir: modules/piper/piper_out

llm:
  n_ctx: 16384
  n_threads: 8
  temperature: 0.7
  top_p: 0.95
  show_thinking: true

vision:
  enabled: true
  model: qwen3-vl-8b
  image_max_tokens: 1024
  frame_width: 640
  enable_thinking: false
  show_thinking: true
  thinking_max_tokens: 128
  frame_memory_items: 5
  frame_memory_max_tokens: 512
  vision_cooldown: 15

camera:
  backend: dshow
  index: 0
  width: 1280
  height: 720
  process_width: 640
  process_height: 360
  frame_skip: 2

autonomous:
  enabled: true
  thought_cooldown: 30

ui:
  preview_interval_sec: 0.5
  motion_sensitivity: 30
  vision_enabled: true
  gesture_enabled: false
  gesture_cooldown: 3
  tts_enabled: true

memory:
  enable: true
  db_path: ./data/memory/nomous.sqlite
```

### Optional Integrations
- Set `llm.n_gpu_layers` and install CUDA-enabled `llama-cpp-python` for GPU offload.
- Install `mediapipe` to enable gesture recognition inside the camera loop.
- Provide a valid Piper voice/executable pair to enable text-to-speech in addition to subtitles.
