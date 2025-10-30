# Nomous WS Bridge (Local Autonomy Runtime)

Offline-first local bridge that connects your React dashboard to:
- **Local LLM** via `llama.cpp` (GGUF path in `config.yaml`)
- **STT** via **Vosk** + **WebRTC VAD**
- **TTS** via **Piper** (Windows-friendly CLI)
- **Vision** via OpenCV (DirectShow)

## Features
- Token streaming + partial speech text to the UI
- Reinforcement knobs (reward/penalty) with running total
- Live camera preview (data URLs) with configurable snapshot debounce
- Mic streaming from browser → PCM16 → Vosk recognizer
- Optional Piper TTS synthesis on final model output (saved to `paths.piper_out_dir`)
- Thread-safe WebSocket broadcast; no coroutine warnings

## Folder Structure
```
nomous-ws-bridge-full/
├─ run_bridge.py
├─ config.yaml
└─ backend/
   ├─ config.py
   ├─ utils.py
   ├─ handlers.py
   ├─ video.py
   ├─ audio.py
   ├─ llm.py
   └─ tts.py
```

## Config (keys are stable – do not rename)
```yaml
paths:
  gguf_path: E:/Models/.../model.gguf
  embed_gguf_path: models/embed/bge-small-en-v1.5-f16.gguf
  vosk_model_dir: C:/Tools/vosk-model-small-en-us-0.15
  piper_exe: C:/Tools/piper/piper.exe
  piper_voice: C:/Tools/voices/en_US-amy-low.onnx
  piper_out_dir: C:/Tools/piper_out

llm:
  enable: true
  n_ctx: 2048
  n_threads: 4
  temperature: 0.6
  top_p: 0.95

audio:
  sample_rate: 16000
  vad_aggressiveness: 2
  chunk_ms: 250
  device: default

camera:
  backend: dshow
  index: 0
  width: 1280
  height: 720

ws:
  host: 127.0.0.1
  port: 8765

memory:
  enable: true
  db_path: ./data/memory/nomous.sqlite

rag:
  enable: true
  db_path: ./data/rag/chroma.sqlite

ui:
  snapshot_debounce: 4
  motion_sensitivity: 30
  tts_enabled: true
  vision_enabled: true

piper:
  rate: 22050
```
