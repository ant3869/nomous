# GPU Setup and Troubleshooting Guide

## Current Status

**GPU Detected:** ✅ RTX 2080 Ti (11GB VRAM)  
**CUDA Version:** 13.0 (Driver 581.57)  
**PyTorch CUDA:** ❌ Not working (CPU-only or broken numpy)

## Issues Found

### 1. NumPy Compatibility Error
```
ImportError: DLL load failed while importing _multiarray_umath
```
This indicates a Python/NumPy architecture mismatch or corrupted installation.

### 2. PyTorch May Not Have CUDA Support
The `requirements.txt` just has `torch` without specifying CUDA version.

## Solution

### Step 1: Fix NumPy and PyTorch with CUDA

Run this command to install proper GPU-enabled packages:

```powershell
# Uninstall potentially broken packages
pip uninstall torch numpy -y

# Install PyTorch with CUDA 11.8 (compatible with CUDA 13.0 driver)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install compatible numpy
pip install "numpy<2.0" --force-reinstall
```

### Step 2: Install GPU-Accelerated llama-cpp-python

```powershell
# Set environment variable for CUDA compilation
$env:CMAKE_ARGS="-DGGML_CUDA=on"

# Uninstall CPU version
pip uninstall llama-cpp-python -y

# Install GPU version (this compiles from source, takes 5-10 minutes)
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose
```

### Step 3: Verify Installation

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
PyTorch: 2.x.x+cu118
CUDA Available: True
GPU: NVIDIA GeForce RTX 2080 Ti
```

### Step 4: Test GPU Profiler

```powershell
python tests\test_performance.py
```

Expected output:
```
GPU Enabled: True
Device Count: 1
Device Name: NVIDIA GeForce RTX 2080 Ti
```

## Updated Requirements

Create `requirements-gpu.txt`:

```txt
# Core dependencies
websockets
numpy<2.0
opencv-contrib-python
Pillow
PyYAML
requests
diskcache
Jinja2
psutil
rich
pynvml

# Audio/VAD/ASR
sounddevice
soundfile
webrtcvad
vosk

# LLM (install separately with CUDA)
# llama-cpp-python (see setup instructions)
sentencepiece

# Visualization
matplotlib
mediapipe

# Testing
pytest-asyncio

# PyTorch with CUDA (install separately)
# torch (see setup instructions)
```

## Quick Setup Script

Create `setup_gpu_full.ps1`:

```powershell
# Nomous GPU Setup Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Nomous GPU Optimization Setup" -ForegroundColor Cyan
Write-Host "  RTX 2080 Ti (CUDA 13.0)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Step 1: Fix NumPy
Write-Host "`nStep 1: Fixing NumPy..." -ForegroundColor Yellow
pip uninstall numpy -y
pip install "numpy<2.0" --force-reinstall

# Step 2: Install PyTorch with CUDA
Write-Host "`nStep 2: Installing PyTorch with CUDA..." -ForegroundColor Yellow
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 3: Install GPU llama-cpp-python
Write-Host "`nStep 3: Installing GPU-accelerated llama-cpp-python..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes..." -ForegroundColor Gray
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip uninstall llama-cpp-python -y
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir

# Step 4: Verify
Write-Host "`nStep 4: Verifying installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nRun: python tests\test_performance.py" -ForegroundColor Cyan
```

## Troubleshooting

### If PyTorch still shows CPU-only:

1. **Check CUDA version compatibility:**
   ```powershell
   nvidia-smi
   ```
   Look for "CUDA Version: X.X"

2. **Try different CUDA version:**
   - CUDA 11.8: `--index-url https://download.pytorch.org/whl/cu118`
   - CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`

3. **Verify PATH includes CUDA:**
   ```powershell
   $env:PATH
   ```
   Should include paths like `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin`

### If llama-cpp-python won't compile with CUDA:

1. **Install Visual Studio Build Tools:**
   - Download from: https://visualstudio.microsoft.com/downloads/
   - Install "Desktop development with C++" workload

2. **Install CMake:**
   ```powershell
   choco install cmake
   ```

3. **Restart PowerShell** to refresh environment variables

### If NumPy still breaks:

```powershell
# Nuclear option: clean reinstall
pip uninstall numpy -y
pip cache purge
pip install "numpy==1.24.3" --force-reinstall --no-cache-dir
```

## Performance Expectations

Once GPU is working properly:

| Metric | Before (CPU) | After (GPU RTX 2080 Ti) |
|--------|--------------|-------------------------|
| Model Loading | 30-60s | 5-15s |
| Inference (8B model) | 8-15s | 1-3s |
| Tokens/Second | 5-10 | 40-80 |
| Memory Usage | 8GB RAM | 6GB VRAM |
| Concurrent Processing | ❌ | ✅ |

## Configuration

Update `config.yaml`:

```yaml
llm:
  enable: true
  n_ctx: 2048
  n_gpu_layers: -1  # Use ALL GPU layers (recommended for RTX 2080 Ti)
  n_threads: 4      # Reduced since GPU handles most work
  temperature: 0.7
  top_p: 0.95
```

## Next Steps

After GPU is working:

1. ✅ Run performance tests
2. ✅ Monitor GPU usage with `nvidia-smi -l 1`
3. ✅ Test think/speak separation
4. ✅ Verify TTS speaks final responses only
5. ✅ Check GPU memory cleanup between generations

## Support

If issues persist, check:
- `logs/nomous.log` for detailed errors
- GPU temperature (should stay under 80°C)
- VRAM usage (should leave ~2GB free)
- Windows Event Viewer for driver issues
