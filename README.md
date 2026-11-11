# Nomous

![Nomous Logo](logo.png)

**A Local Autonomy Runtime Bridge**

## Overview
Nomous is an offline-first WebSocket bridge that connects a TypeScript React dashboard to local AI services. The runtime wires together:
- 🧠 **Local LLM** loading via `llama.cpp`, including live load progress reporting and runtime sampling controls
- 🎤 **Speech-to-Text** streaming from the browser microphone to Vosk with configurable sensitivity and automatic LLM triggering
- 🎧 **Ambient hearing** — the host mic as a second sense alongside vision: continuous capture with VAD segmentation and Vosk transcription, feeding the idle loop so a noise can trigger a look and a new sighting can trigger listening harder. Opt-in (off by default), with a dashboard mic-live indicator; raw audio is never written to disk
- 🗣️ **Text-to-Speech** synthesis through Piper with runtime toggles, voice switching, and optional auto playback
- 👁️ **Computer Vision** capture with MediaPipe Tasks gesture/face detection, motion-aware frame streaming, and real image input to a Qwen vision model — the camera frame itself goes to the model, not a text summary of it
- 📈 **System telemetry** that surfaces CPU/GPU utilisation and memory pressure inside the dashboard UI
- 📚 **Persistent memory graph & tool system** that lets the model store observations, run self-evaluations, and surface tool activity in the UI
- 👤 **Person identity tracking** that recognizes individuals via on-device face embeddings (MediaPipe ImageEmbedder, cosine-matched in the person tracker), learns names, associates conversations, and builds relationships across sessions

## Feature Highlights
- ✨ Real-time token streaming, load overlays, and chat playback for every assistant response
- 🛠️ **25+ built-in LLM tools** covering memory search, observation logging, learning, social safeguards, and person tracking with UI visualisation of tool usage and stats
- 👤 **Person identity tracking** with face-embedding matching, automatic name detection, conversation binding, and relationship building
- 📹 Live camera preview with motion thresholds, gesture detection, face detection, and configurable preview-interval/cooldown windows
- 🎙️ Browser microphone streaming (PCM16) with automatic speech detection and configurable sensitivity
- 🔊 Optional Piper TTS synthesis with per-session toggles for enablement, auto playback, volume, and voice selection
- 🧠 Persistent memory graph with SQLite backing plus front-end visualisation and analytics helpers
- 📊 Dashboard telemetry (system metrics, behaviour scoring, reward accounting, token throughput) surfaced through dedicated cards and charts

## Feature Status
### ✅ Complete & Working
- Local llama.cpp LLM loading with progress events, sampling controls, and autonomous thought/vision loops
- 🤖 Autonomous idle loop: free-running tick (default 30s) that looks at the freshest frame and decides think / speak / act — silent thoughts and tool calls stay visible in the Think stream, speech is rate-limited
- 🌙 Dreaming: nightly memory consolidation (light → REM → deep) that promotes durable, recurring notes into long-term memory, with a Dream Diary of every decision
- Vosk-based microphone pipeline that feeds transcripted speech into the LLM and echoes microphone activity back to the UI
- MediaPipe-enabled camera loop with motion gating, gesture detection, and asynchronous vision prompts for the model
- Tool execution framework with 26 tools, execution history, and dedicated dashboard widgets
- System telemetry publishing CPU/GPU metrics and surfacing device context in the UI system usage card
- Persistent memory store that records interactions and streams the graph to the UI via WebSocket events

### ⚠️ Work in Progress / Limited Support
- GPU metrics require `torch`, `pynvml`, and `psutil`; missing dependencies will disable telemetry and cause the Python test suite to fail when system monitoring tests import them

### 🚧 Not Yet Implemented / Planned
- Gesture actions are live: open palm pauses speech, fist toggles mute (enabled by default). Additional gesture classifiers (thumbs down, etc.) and configurable gesture-to-action mappings remain roadmap items
- Custom wake-word activation for hands-free microphone capture

## Quick Start

### One-command bootstrap
Run the launcher to automatically create a virtual environment, install Python
and Node dependencies, validate required tooling, and start both the backend and
frontend services:

```bash
python run_nomous.py
# Legacy invocation supported for convenience
python start.py
```

If any prerequisite is missing or a step fails, the launcher prints a detailed
message explaining what happened and how to resolve it.

### Manual setup
1. Clone the repository
```bash
git clone https://github.com/ant3869/nomous.git
cd nomous
```

2. Install dependencies
```bash
# Backend
pip install -r requirements.txt

# Frontend
npm install
```

3. Configure `config.yaml` with your model paths and settings

4. Start the full bridge (backend workers + React dev server)
```bash
python scripts/start.py
```

To run only the backend bridge without launching the UI dev server:
```bash
python scripts/run_bridge.py
```

## Project Structure
```
nomous/
├─ src/
│  ├─ backend/         # Python WebSocket server
│  │  ├─ audio.py     # Audio processing
│  │  ├─ video.py     # Video capture
│  │  ├─ llm.py       # LLM integration
│  │  ├─ tts.py       # Text-to-speech
│  │  ├─ memory.py    # Persistent memory graph
│  │  ├─ tools.py     # LLM tool system
│  │  └─ ...          # config, handlers, system, person tracking
│  └─ frontend/       # React TypeScript UI
│     ├─ components/  # UI components
│     └─ App.tsx      # Main application
│
├─ scripts/           # Launcher + bridge entry points
├─ tests/             # Test suite
├─ docs/              # Guides + changelog
└─ config.yaml        # Configuration
```

## Configuration
Key configuration sections in `config.yaml`:

```yaml
paths:
  gguf_path: H:/Models/Qwen3VL-8B-Instruct-Q4_K_M.gguf
  mmproj_path: H:/Models/mmproj-Qwen3VL-8B-Instruct-F16.gguf
  embed_gguf_path: models/embed/bge-small-en-v1.5-f16.gguf
  vosk_model_dir: path/to/vosk-model
  piper_exe: path/to/piper.exe
  piper_voice: path/to/voice.onnx
  piper_out_dir: path/to/output

llm:
  enable: true
  n_ctx: 16384
  n_threads: 8
  temperature: 0.7
  top_p: 0.95

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
  temperature: 0.5
```

The vision model is Qwen3-VL-8B by default (~7 GB all-in, fits a 16 GB card).
The 27B model needs a 2–3-bit quant to fit; see `vision.model: qwen3.8-27b` in
`config.yaml` and [docs/GPU_SETUP.md](docs/GPU_SETUP.md). The launcher installs
the vision-capable llama-cpp-python fork wheel automatically — the stock PyPI
package has no Qwen vision handlers.

Vision calls carry a rolling memory of the last few analyzed frames
(`frame_memory_items` / `frame_memory_max_tokens`), so the model can track
change over time instead of seeing one still per call. The dashboard thoughts
panel shows the model's real reasoning (`show_thinking`, on by default);
reasoning is stripped before anything reaches Piper, so it is never spoken
aloud.

See [INSTALL_INSTRUCTIONS.md](docs/INSTALL_INSTRUCTIONS.md) for detailed setup steps.

### Optional Components
- **GPU acceleration** – enable by setting `llm.n_gpu_layers`; the launcher installs the matching torch CUDA build and the vision-capable llama-cpp-python fork wheel automatically (see [docs/GPU_SETUP.md](docs/GPU_SETUP.md)). Telemetry additionally needs `torch` + `pynvml` (see `requirements.txt`).
- **Piper text-to-speech** – requires `paths.piper_exe`, `paths.piper_voice`, and `paths.piper_out_dir` to exist; playback is via sounddevice, cross-platform.
- **Vision analysis & gestures** – ensure OpenCV can access your camera and install `mediapipe` for gesture detection.

## Testing
Run the test suite:
```bash
python -m pytest tests/
```

## Documentation
- [Installation Guide](docs/INSTALL_INSTRUCTIONS.md)
- [Frontend Documentation](docs/README_FE.md)
- [LLM Tool System](docs/TOOLS.md)
- [Implementation Summary](docs/IMPLEMENTATION_SUMMARY.md)
- [Changelog](docs/CHANGELOG.md)
- [Bug Fixes](docs/BUG_FIXES.md)
- [Testing Guide](docs/TESTING.md)

## GitHub Copilot Collections
The following Copilot collections are installed to enhance development:

### Python Development & ML Collection
- 📝 **Instructions**: GPU optimization, ML model integration, async patterns
- 🎯 **Prompts**: GPU optimization guide
- 💭 **Chat Mode**: Python ML expert for guidance

### Frontend Web Development Collection
- 📝 **Instructions**: React, TypeScript, WebSocket best practices
- 🎯 **Prompts**: React component generator
- 💭 **Chat Mode**: Frontend expert for assistance

### Testing & Test Automation Collection
- 📝 **Instructions**: WebSocket and ML model testing patterns
- 🎯 **Prompts**: Test suite generator
- 💭 **Chat Mode**: Testing expert for guidance

### Security & Code Quality Collection
- 📝 **Instructions**: Security best practices and code quality
- 🎯 **Prompts**: Code quality enhancer
- 💭 **Chat Mode**: Security & quality expert

The collections are located in `.github/copilot/` and include:
```
.github/copilot/
├─ instructions/    # Best practices and guidelines
├─ prompts/        # Task-specific generators
└─ chatmodes/      # Expert chat modes
```

## Technical Stack
- **Backend**: Python, asyncio, websockets, OpenCV, Vosk, Piper
- **Frontend**: React 18+, TypeScript, Tailwind CSS, shadcn/ui
- **ML**: llama.cpp, GGUF models, BGE embeddings
- **Build**: Vite, PostCSS

## License
MIT License - See [LICENSE](LICENSE) for details