# GPU Setup and Troubleshooting Guide

## Current Setup

**GPU:** RTX 5070 Ti (16 GB VRAM)
**CUDA driver:** 13.x (the launcher reads the driver version and picks matching wheels)
**PyTorch:** CUDA build from the `cu128` wheel index (torch ships no wheels past cu128)
**LLM runtime:** `llama-cpp-python` **JamePeng fork 0.4.6** — the stock PyPI package has no
Qwen vision chat handlers, so image input only works with the fork wheel.

The launcher (`python scripts/start.py`) does all of this automatically: it detects
the driver CUDA version, installs the matching torch CUDA build, and installs the
matching fork wheel from the JamePeng GitHub releases. The steps below are for
manual setup or troubleshooting when the launcher path fails.

## Vision model

Default (fits the 16 GB card, ~7 GB all-in):

- `Qwen3VL-8B-Instruct-Q4_K_M.gguf` + `mmproj-Qwen3VL-8B-Instruct-F16.gguf`
  from `Qwen/Qwen3-VL-8B-Instruct-GGUF` on Hugging Face
- Config: `paths.gguf_path` and `paths.mmproj_path` in `config.yaml`
  (mmproj is also auto-detected next to the model file)

Option (tight fit, needs a 2–3-bit quant): Qwen3.8-27B. Set `vision.model: qwen3.8-27b`
in `config.yaml`; the launcher picks `GenericMTMDChatHandler` for it instead of
`Qwen3VLChatHandler`. Without an mmproj file the model runs text-only — the camera
frames never reach it.

## Manual setup

### Step 1: Python dependencies

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins every dependency, including `mediapipe==1.2.2` (Tasks API)
and `opencv-contrib-python==4.12.0.88`. Do not let pip float them: MediaPipe 1.x
removed `mp.solutions` (the old Hands API the code used to use), and OpenCV 5
dropped the Haar `CascadeClassifier` the old face detector relied on.

### Step 2: PyTorch with CUDA

```powershell
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```

The launcher clamps the torch wheel to cu128 even on CUDA 13.x drivers — there is
no newer torch CUDA wheel. Verify with:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### Step 3: Vision-capable llama-cpp-python (JamePeng fork)

The fork is never on PyPI. On Windows the launcher downloads the matching wheel
from <https://github.com/JamePeng/llama-cpp-python/releases> (matched by Python
version, e.g. `cp312`, and CUDA tag, e.g. `cu132`), for example:

```text
llama_cpp_python-0.4.6+cu132-cp312-cp312-win_amd64.whl
```

Manual install:

```powershell
pip install --force-reinstall --no-cache-dir "<wheel URL from the releases page>"
```

On Linux there are no prebuilt wheels — build the fork from source:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall --no-cache-dir \
  "llama-cpp-python @ git+https://github.com/JamePeng/llama-cpp-python.git@v0.4.6"
```

Verify the handlers are present:

```powershell
python -c "from llama_cpp.llama_multimodal import Qwen3VLChatHandler, GenericMTMDChatHandler; print('vision handlers OK')"
```

If that import fails, vision calls fall back to text-only and the log will say so —
the app still runs, it just cannot see images.

### Step 4: MediaPipe Task models

The gesture recognizer and face detector models auto-download on first run into
`modules/models/mediapipe/` (gitignored):

- `gesture_recognizer.task` (~8.4 MB)
- `blaze_face_short_range.tflite` (~230 KB)

If the download fails (no network), gestures and face detection stay off and the
camera loop keeps running on motion + brightness only.

### Step 5: Verify

```powershell
python tests\test_gpu.py
python -m pytest tests/
```

## Troubleshooting

### llama-cpp-python has no vision handlers

You have the stock PyPI build. Uninstall it and install the fork wheel (Step 3).
Do not `pip install "llama-cpp-python[cuda]"` — that is the stock package.

### PyTorch shows CPU-only

Check `nvidia-smi` for the driver CUDA version, then reinstall torch from the
cu128 index (Step 2). The launcher prints the detected CUDA tag at startup.

### MediaPipe import error / `mp.solutions` missing

`mp.solutions` was removed in MediaPipe 1.x. Install the pinned version
(`pip install mediapipe==1.2.2`) — the code uses the Tasks API
(`GestureRecognizer`, `FaceDetector`) only.

### NumPy DLL errors on Windows

A NumPy/architecture mismatch (e.g. mixing a 32-bit interpreter or a venv created
inside the NumPy source tree). Recreate the venv from a clean directory and
reinstall from `requirements.txt`.

### If llama-cpp-python won't build with CUDA (Linux source build)

- Install Visual Studio Build Tools ("Desktop development with C++") on Windows,
  or a C++ toolchain + CMake on Linux.
- Restart the shell so the new tools are on PATH, then retry Step 3.

## Configuration

`config.yaml` (current defaults):

```yaml
paths:
  gguf_path: H:/Models/Qwen3VL-8B-Instruct-Q4_K_M.gguf
  mmproj_path: H:/Models/mmproj-Qwen3VL-8B-Instruct-F16.gguf

llm:
  enable: true
  n_ctx: 16384
  n_gpu_layers: -1  # all layers on GPU
  n_threads: 8
  temperature: 0.6
  top_p: 0.95

vision:
  enabled: true
  model: qwen3-vl-8b
  image_max_tokens: 1024
  frame_width: 640
  enable_thinking: false
  temperature: 0.5
```

Notes:

- Qwen disables context shift, so a context overflow is a hard error. The LLM
  budgets every vision call (frame + prompt + output vs `n_ctx`), trims rolling
  context first, and retries once after clearing context instead of crashing.
- Thinking is off by default (`vision.enable_thinking: false`) and `<think>`
  blocks are stripped from output before it reaches Piper, so reasoning is never
  read aloud.

## Performance Expectations

| Metric | CPU-only | GPU (RTX 5070 Ti) |
|--------|----------|-------------------|
| Model loading (8B) | 30–60 s | 5–15 s |
| Inference | 8–15 s | 1–3 s |
| Tokens/second | 5–10 | 40–80 |
| Vision frame (640px) | n/a | ~1 s per analysis |
