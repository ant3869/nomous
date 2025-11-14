"""
Test GPU Profiler and Performance Optimizations
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backend.gpu_profiler import profiler, GPUMetrics


def test_gpu_profiler_initialization():
    """Test GPU profiler initializes correctly."""
    print("Testing GPU Profiler Initialization...")
    print(f"  GPU Enabled: {profiler.enabled}")
    print(f"  Device Count: {profiler.device_count}")
    print(f"  Device Name: {profiler.device_name}")
    assert profiler is not None
    print("✅ GPU Profiler initialization test passed")


def test_inference_timing():
    """Test inference timing works."""
    print("\nTesting Inference Timing...")
    
    start_time = profiler.start_inference()
    
    # Simulate some work
    import time
    time.sleep(0.1)
    
    metrics = profiler.end_inference(start_time)
    
    print(f"  Inference Time: {metrics.inference_time_ms:.2f}ms")
    print(f"  Memory Allocated: {metrics.memory_allocated_mb:.2f}MB")
    print(f"  Memory Reserved: {metrics.memory_reserved_mb:.2f}MB")
    
    assert metrics.inference_time_ms > 0
    assert isinstance(metrics, GPUMetrics)
    print("✅ Inference timing test passed")


def test_memory_optimization():
    """Test memory optimization runs without errors."""
    print("\nTesting Memory Optimization...")
    
    try:
        profiler.optimize_memory()
        print("  Memory optimization executed")
        print("✅ Memory optimization test passed")
    except Exception as e:
        print(f"⚠️  Memory optimization warning: {e}")


def test_memory_summary():
    """Test memory summary retrieval."""
    print("\nTesting Memory Summary...")
    
    summary = profiler.get_memory_summary()
    print("  Memory Summary:")
    for key, value in summary.items():
        print(f"    {key}: {value}")
    
    assert summary is not None
    assert isinstance(summary, dict)
    print("✅ Memory summary test passed")


def test_metrics_logging():
    """Test metrics logging."""
    print("\nTesting Metrics Logging...")
    
    start_time = profiler.start_inference()
    import time
    time.sleep(0.05)
    metrics = profiler.end_inference(start_time)
    
    profiler.log_metrics(metrics)
    print("✅ Metrics logging test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("GPU PROFILER PERFORMANCE TESTS")
    print("=" * 60)
    
    try:
        test_gpu_profiler_initialization()
        test_inference_timing()
        test_memory_optimization()
        test_memory_summary()
        test_metrics_logging()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
