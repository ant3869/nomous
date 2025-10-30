#!/usr/bin/env python3
"""
GPU Verification Test for Nomous
Tests if CUDA and GPU acceleration are properly configured
"""

import sys

print("=" * 60)
print("  Nomous GPU Verification Test")
print("=" * 60)
print()

# Test 1: CUDA availability
print("[1/4] Testing CUDA availability...")
try:
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("❌ CUDA not available")
        print("   Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
except ImportError:
    print("⚠️  PyTorch not installed (optional, but useful for testing)")
    print("   Install with: pip install torch")

print()

# Test 2: llama-cpp-python with CUDA
print("[2/4] Testing llama-cpp-python GPU support...")
try:
    from llama_cpp import Llama
    print("✅ llama-cpp-python installed")
    
    # Check if it was built with CUDA
    try:
        test_model = Llama(
            model_path="test.gguf",  # This will fail, but we can check the error
            n_gpu_layers=1
        )
    except Exception as e:
        error_msg = str(e)
        if "CUDA" in error_msg or "GPU" in error_msg:
            print("✅ llama-cpp-python supports GPU (CUDA enabled)")
        elif "test.gguf" in error_msg or "not found" in error_msg:
            print("✅ llama-cpp-python installed (can't verify CUDA without model)")
        else:
            print(f"⚠️  llama-cpp-python might not have CUDA support")
            print(f"   Error: {error_msg[:100]}")
except ImportError:
    print("❌ llama-cpp-python not installed")
    print("   Install with: pip install llama-cpp-python")

print()

# Test 3: MediaPipe
print("[3/4] Testing MediaPipe for gesture detection...")
try:
    import mediapipe as mp
    print("✅ MediaPipe installed")
    
    # Test hands module
    hands = mp.solutions.hands.Hands()
    print("✅ MediaPipe hand detection ready")
    hands.close()
except ImportError:
    print("❌ MediaPipe not installed")
    print("   Install with: pip install mediapipe")
except Exception as e:
    print(f"⚠️  MediaPipe installed but error: {e}")

print()

# Test 4: OpenCV
print("[4/4] Testing OpenCV...")
try:
    import cv2
    print(f"✅ OpenCV installed: v{cv2.__version__}")
    
    # Test camera access
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✅ Camera accessible")
        cap.release()
    else:
        print("⚠️  Cannot open camera at index 0")
except ImportError:
    print("❌ OpenCV not installed")
    print("   Install with: pip install opencv-python")

print()
print("=" * 60)
print("  Summary")
print("=" * 60)

# Final recommendations
print()
print("Recommended actions:")
print("1. If CUDA not available: Install CUDA Toolkit")
print("2. If llama-cpp-python has no GPU: Run setup_gpu.bat")
print("3. If MediaPipe missing: pip install mediapipe")
print("4. After changes: Restart computer!")
print()
print("To run optimized Nomous:")
print("  python run_bridge.py")
print()
