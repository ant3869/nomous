@echo off
REM Quick verification that GPU and think/speak fixes are working
echo ========================================
echo   Nomous GPU + Think/Speak Verification
echo ========================================
echo.

echo [1/3] Checking Python environment...
H:\nomous\.venv\Scripts\python.exe --version
echo.

echo [2/3] Verifying GPU (CUDA)...
H:\nomous\.venv\Scripts\python.exe -c "import torch; print('  PyTorch:', torch.__version__); print('  CUDA Available:', torch.cuda.is_available()); print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('  VRAM:', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2), 'GB' if torch.cuda.is_available() else '')"
echo.

echo [3/3] Running performance tests...
H:\nomous\.venv\Scripts\python.exe tests\test_performance.py
echo.

echo ========================================
echo   Verification Complete!
echo ========================================
echo.
echo Next step: Run the application and test:
echo   1. Ask model a question
echo   2. Check thought window (should show processing, NOT "Generating: ...")
echo   3. Check TTS speaks final clean response only
echo   4. Monitor GPU: nvidia-smi -l 1
echo.
pause
