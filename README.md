# Nomous

<div align="center">
  <img src="logo.png" alt="Nomous Logo" width="800"/>
  <br>
  <strong>A Local Autonomy Runtime Bridge</strong>
</div>

## Overview
Nomous is an offline-first WebSocket bridge that seamlessly connects your React dashboard to powerful local AI capabilities:
- 🧠 **Local LLM** via `llama.cpp` (GGUF models)
- 🎤 **Speech-to-Text** via **Vosk** + **WebRTC VAD**
- 🗣️ **Text-to-Speech** via **Piper** (Windows-optimized)
- 👁️ **Computer Vision** via OpenCV (DirectShow)

## Features
- ✨ Real-time token streaming with partial speech text to UI
- 🎯 Reinforcement learning controls (reward/penalty) with running total
- 📹 Live camera preview with configurable snapshot debounce
- 🎙️ Browser microphone streaming (PCM16) with Vosk recognition
- 🔊 Optional Piper TTS synthesis for model outputs
- 🔄 Thread-safe WebSocket broadcast system

## Quick Start
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

4. Start the application (both frontend and backend)
```bash
python scripts/start.py
```

This will:
- Start the backend server
- Launch the frontend development server
- Open your browser to the application automatically
- Press Ctrl+C in the terminal to stop all services
```

3. Configure `config.yaml` with your model paths and settings

4. Start the bridge
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
│  │  └─ tts.py       # Text-to-speech
│  │
│  └─ frontend/       # React TypeScript UI
│     ├─ components/  # UI components
│     └─ App.tsx      # Main application
│
├─ scripts/           # Utility scripts
├─ tests/            # Test suite
└─ config.yaml       # Configuration
```

## Configuration
Key configuration sections in `config.yaml`:

```yaml
paths:
  gguf_path: path/to/model.gguf
  embed_gguf_path: models/embed/bge-small-en-v1.5-f16.gguf
  vosk_model_dir: path/to/vosk-model
  piper_exe: path/to/piper.exe
  piper_voice: path/to/voice.onnx
  piper_out_dir: path/to/output

llm:
  enable: true
  n_ctx: 2048
  n_threads: 4
  temperature: 0.6
  top_p: 0.95
```

See [INSTALL_INSTRUCTIONS.md](docs/INSTALL_INSTRUCTIONS.md) for detailed setup steps.

## Testing
Run the test suite:
```bash
python -m pytest tests/
```

## Documentation
- [Installation Guide](docs/INSTALL_INSTRUCTIONS.md)
- [Frontend Documentation](docs/README_FE.md)
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