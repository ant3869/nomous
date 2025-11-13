#!/usr/bin/env python3
"""
Test suite for system monitoring and device detection
Tests SystemMonitor class and detect_compute_device function
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pytest
except ImportError:
    # Define a dummy decorator if pytest is not available
    class _MockMark:
        @staticmethod
        def asyncio(func):
            return func
    
    class _MockPytest:
        mark = _MockMark()
    
    pytest = _MockPytest()

from src.backend.system import (
    ComputeDeviceInfo,
    detect_compute_device,
    SystemMonitor,
)


class MockBridge:
    """Mock bridge for testing."""
    def __init__(self):
        self.messages = []
    
    async def post(self, message):
        """Record posted messages."""
        self.messages.append(message)


class MockVirtualMemory:
    """Mock psutil virtual memory."""
    def __init__(self):
        self.total = 16 * 1024 * 1024 * 1024  # 16GB
        self.used = 8 * 1024 * 1024 * 1024    # 8GB
        self.available = 8 * 1024 * 1024 * 1024
        self.percent = 50.0


class MockSwapMemory:
    """Mock psutil swap memory."""
    def __init__(self):
        self.total = 4 * 1024 * 1024 * 1024   # 4GB
        self.used = 1 * 1024 * 1024 * 1024    # 1GB
        self.percent = 25.0


class MockCPUFreq:
    """Mock psutil CPU frequency."""
    def __init__(self):
        self.current = 2400.0


class MockNVMLMemoryInfo:
    """Mock NVML memory info."""
    def __init__(self):
        self.total = 8 * 1024 * 1024 * 1024   # 8GB
        self.used = 2 * 1024 * 1024 * 1024    # 2GB


class MockNVMLUtilization:
    """Mock NVML utilization rates."""
    def __init__(self):
        self.gpu = 50.0


# Test ComputeDeviceInfo dataclass
def test_compute_device_info_cpu():
    """Test ComputeDeviceInfo for CPU."""
    info = ComputeDeviceInfo(
        backend="CPU",
        name="Intel Core i7",
        reason="No CUDA detected"
    )
    
    assert info.backend == "CPU"
    assert info.name == "Intel Core i7"
    assert info.is_gpu is False
    print("✅ ComputeDeviceInfo CPU configuration works")


def test_compute_device_info_gpu():
    """Test ComputeDeviceInfo for GPU."""
    info = ComputeDeviceInfo(
        backend="GPU",
        name="NVIDIA RTX 3080",
        reason="CUDA available",
        cuda_version="11.8",
        gpu_count=1,
        cuda_ready=True
    )

    assert info.backend == "GPU"
    assert info.name == "NVIDIA RTX 3080"
    assert info.is_gpu is True
    assert info.cuda_version == "11.8"
    assert info.gpu_count == 1
    print("✅ ComputeDeviceInfo GPU configuration works")


# Test device detection without CUDA
@patch('src.backend.system.torch')
@patch('src.backend.system.shutil.which')
def test_detect_compute_device_no_cuda(mock_which, mock_torch):
    """Test device detection when CUDA is not available."""
    mock_torch.cuda.is_available.return_value = False
    mock_torch.version.cuda = None
    mock_which.return_value = None
    
    with patch('src.backend.system.platform.processor', return_value="Intel i7"):
        device_info = detect_compute_device()

    assert device_info.backend == "CPU"
    assert device_info.is_gpu is False
    assert device_info.cuda_ready is False
    assert device_info.gpu_count == 0
    assert device_info.cuda_version is None
    assert "No CUDA-capable GPU detected" in device_info.reason
    print("✅ Device detection without CUDA works")


# Test device detection with CUDA
@patch('src.backend.system.torch')
@patch('src.backend.system.shutil.which')
def test_detect_compute_device_with_cuda(mock_which, mock_torch):
    """Test device detection when CUDA is available."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 3080"
    mock_torch.version.cuda = "11.8"
    
    device_info = detect_compute_device()

    assert device_info.backend == "GPU"
    assert device_info.is_gpu is True
    assert device_info.cuda_ready is True
    assert device_info.gpu_count == 1
    assert device_info.cuda_version == "11.8"
    assert device_info.name == "NVIDIA RTX 3080"
    assert "CUDA runtime detected" in device_info.reason
    print("✅ Device detection with CUDA works")


# Test device detection with nvidia-smi but no CUDA runtime
@patch('src.backend.system.torch')
@patch('src.backend.system.shutil.which')
def test_detect_compute_device_nvidia_smi_no_runtime(mock_which, mock_torch):
    """Test device detection when nvidia-smi exists but CUDA runtime unavailable."""
    mock_torch.cuda.is_available.return_value = False
    mock_torch.version.cuda = None
    mock_which.return_value = "/usr/bin/nvidia-smi"
    
    with patch('src.backend.system.platform.processor', return_value="Intel i7"):
        device_info = detect_compute_device()

    assert device_info.backend == "CPU"
    assert device_info.is_gpu is False
    assert device_info.cuda_ready is False
    assert "NVIDIA GPU detected but CUDA drivers" in device_info.reason
    print("✅ Device detection with nvidia-smi but no CUDA runtime works")


# Test device detection with CUDA exception
@patch('src.backend.system.torch')
def test_detect_compute_device_cuda_exception(mock_torch):
    """Test device detection when CUDA query raises exception."""
    mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA error")
    
    with patch('src.backend.system.platform.processor', return_value="Intel i7"):
        device_info = detect_compute_device()

    assert device_info.backend == "CPU"
    assert device_info.is_gpu is False
    assert device_info.cuda_ready is False
    assert "Unable to query CUDA runtime" in device_info.reason
    print("✅ Device detection with CUDA exception works")


# Test SystemMonitor initialization
@patch('src.backend.system.detect_compute_device')
@patch('src.backend.system.psutil.cpu_percent')
def test_system_monitor_init(mock_cpu_percent, mock_detect):
    """Test SystemMonitor initialization."""
    mock_detect.return_value = ComputeDeviceInfo(
        backend="CPU",
        name="Intel i7",
        reason="Test"
    )
    
    bridge = MockBridge()
    monitor = SystemMonitor(bridge, interval=1.0)
    
    assert monitor.bridge == bridge
    assert monitor.interval == 1.0
    assert monitor._running is False
    assert monitor._task is None
    assert monitor._nvml_initialised is False
    mock_cpu_percent.assert_called_once_with(interval=None)
    print("✅ SystemMonitor initialization works")


# Test NVML initialization success
@patch('src.backend.system.nvmlInit')
@patch('src.backend.system.nvmlDeviceGetCount')
@patch('src.backend.system.nvmlDeviceGetHandleByIndex')
@patch('src.backend.system.nvmlDeviceGetName')
@patch('src.backend.system.detect_compute_device')
@patch('src.backend.system.psutil.cpu_percent')
@patch('src.backend.system.shutil.which')
def test_system_monitor_nvml_init_success(
    mock_which, mock_cpu_percent, mock_detect, mock_get_name, 
    mock_get_handle, mock_get_count, mock_nvml_init
):
    """Test successful NVML initialization."""
    mock_which.return_value = "/usr/bin/nvidia-smi"
    mock_detect.return_value = ComputeDeviceInfo(
        backend="GPU",
        name="NVIDIA RTX 3080",
        reason="CUDA available",
        cuda_version="11.8",
        gpu_count=1,
        cuda_ready=True
    )
    mock_get_count.return_value = 1
    mock_get_name.return_value = b"NVIDIA RTX 3080"
    
    bridge = MockBridge()
    monitor = SystemMonitor(bridge)
    
    assert monitor._nvml_initialised is True
    assert monitor._gpu_name_override == "NVIDIA RTX 3080"
    mock_nvml_init.assert_called_once()
    print("✅ NVML initialization success works")


# Test NVML initialization failure
@patch('src.backend.system.nvmlInit')
@patch('src.backend.system.detect_compute_device')
@patch('src.backend.system.psutil.cpu_percent')
@patch('src.backend.system.shutil.which')
def test_system_monitor_nvml_init_failure(
    mock_which, mock_cpu_percent, mock_detect, mock_nvml_init
):
    """Test NVML initialization failure handling."""
    from pynvml import NVMLError
    
    mock_which.return_value = "/usr/bin/nvidia-smi"
    mock_detect.return_value = ComputeDeviceInfo(
        backend="GPU",
        name="NVIDIA RTX 3080",
        reason="CUDA available",
        cuda_ready=True
    )
    mock_nvml_init.side_effect = NVMLError(1)
    
    bridge = MockBridge()
    monitor = SystemMonitor(bridge)
    
    assert monitor._nvml_initialised is False
    assert "NVML unavailable" in monitor.device_info.reason
    print("✅ NVML initialization failure handling works")


# Test metric collection with mock psutil data
@patch('src.backend.system.psutil')
@patch('src.backend.system.detect_compute_device')
@patch('src.backend.system.os.getloadavg')
def test_collect_metrics_cpu_only(mock_getloadavg, mock_detect, mock_psutil):
    """Test metric collection for CPU-only system."""
    mock_detect.return_value = ComputeDeviceInfo(
        backend="CPU",
        name="Intel i7",
        reason="Test"
    )
    
    mock_psutil.cpu_percent.return_value = 45.5
    mock_psutil.cpu_count.return_value = 8
    mock_psutil.cpu_freq.return_value = MockCPUFreq()
    mock_psutil.virtual_memory.return_value = MockVirtualMemory()
    mock_psutil.swap_memory.return_value = MockSwapMemory()
    mock_getloadavg.return_value = (1.5, 2.0, 2.5)
    
    bridge = MockBridge()
    monitor = SystemMonitor(bridge)
    
    metrics = monitor._collect_metrics()
    
    assert "timestamp" in metrics
    assert "device" in metrics
    assert "cpu" in metrics
    assert "memory" in metrics
    assert "swap" in metrics
    
    # Check CPU metrics
    assert metrics["cpu"]["percent"] == 45.5
    assert metrics["cpu"]["cores"] == 8
    assert metrics["cpu"]["frequency"] == 2400.0
    assert metrics["cpu"]["load"] == [1.5, 2.0, 2.5]
    
    # Check memory metrics
    assert metrics["memory"]["total"] == 16 * 1024 * 1024 * 1024
    assert metrics["memory"]["used"] == 8 * 1024 * 1024 * 1024
    assert metrics["memory"]["percent"] == 50.0
    
    # Check swap metrics
    assert metrics["swap"]["total"] == 4 * 1024 * 1024 * 1024
    assert metrics["swap"]["used"] == 1 * 1024 * 1024 * 1024
    assert metrics["swap"]["percent"] == 25.0
    
    # No GPU metrics for CPU-only system
    assert "gpu" not in metrics
    
    print("✅ Metric collection for CPU-only system works")


# Test metric collection with NVML GPU metrics
@patch('src.backend.system.nvmlDeviceGetHandleByIndex')
@patch('src.backend.system.nvmlDeviceGetUtilizationRates')
@patch('src.backend.system.nvmlDeviceGetMemoryInfo')
@patch('src.backend.system.nvmlDeviceGetTemperature')
@patch('src.backend.system.nvmlInit')
@patch('src.backend.system.nvmlDeviceGetCount')
@patch('src.backend.system.nvmlDeviceGetName')
@patch('src.backend.system.psutil')
@patch('src.backend.system.detect_compute_device')
@patch('src.backend.system.shutil.which')
def test_collect_metrics_with_nvml(
    mock_which, mock_detect, mock_psutil, mock_get_name, mock_get_count,
    mock_nvml_init, mock_get_temp, mock_get_mem, mock_get_util, mock_get_handle
):
    """Test metric collection with NVML GPU metrics."""
    mock_which.return_value = "/usr/bin/nvidia-smi"
    mock_detect.return_value = ComputeDeviceInfo(
        backend="GPU",
        name="NVIDIA RTX 3080",
        reason="CUDA available",
        cuda_version="11.8",
        gpu_count=1,
        cuda_ready=True
    )
    
    mock_get_count.return_value = 1
    mock_get_name.return_value = b"NVIDIA RTX 3080"
    mock_get_util.return_value = MockNVMLUtilization()
    mock_get_mem.return_value = MockNVMLMemoryInfo()
    mock_get_temp.return_value = 65.0
    
    mock_psutil.cpu_percent.return_value = 45.5
    mock_psutil.cpu_count.return_value = 8
    mock_psutil.cpu_freq.return_value = MockCPUFreq()
    mock_psutil.virtual_memory.return_value = MockVirtualMemory()
    mock_psutil.swap_memory.return_value = MockSwapMemory()
    
    bridge = MockBridge()
    monitor = SystemMonitor(bridge)
    
    metrics = monitor._collect_metrics()
    
    # Check GPU metrics
    assert "gpu" in metrics
    assert metrics["gpu"]["name"] == "NVIDIA RTX 3080"
    assert metrics["gpu"]["percent"] == 50.0
    assert metrics["gpu"]["memory_total"] == 8 * 1024 * 1024 * 1024
    assert metrics["gpu"]["memory_used"] == 2 * 1024 * 1024 * 1024
    assert metrics["gpu"]["temperature"] == 65.0
    
    print("✅ Metric collection with NVML GPU metrics works")


# Test metric collection with PyTorch GPU fallback
@patch('src.backend.system.torch')
@patch('src.backend.system.psutil')
@patch('src.backend.system.detect_compute_device')
def test_collect_metrics_torch_fallback(mock_detect, mock_psutil, mock_torch):
    """Test metric collection using PyTorch CUDA fallback when NVML unavailable."""
    mock_detect.return_value = ComputeDeviceInfo(
        backend="GPU",
        name="NVIDIA RTX 3080",
        reason="CUDA available",
        cuda_version="11.8",
        gpu_count=1,
        cuda_ready=True
    )
    
    mock_torch.cuda.mem_get_info.return_value = (
        6 * 1024 * 1024 * 1024,  # free
        8 * 1024 * 1024 * 1024   # total
    )
    
    mock_psutil.cpu_percent.return_value = 45.5
    mock_psutil.cpu_count.return_value = 8
    mock_psutil.cpu_freq.return_value = MockCPUFreq()
    mock_psutil.virtual_memory.return_value = MockVirtualMemory()
    mock_psutil.swap_memory.return_value = MockSwapMemory()
    
    bridge = MockBridge()
    monitor = SystemMonitor(bridge)
    
    # NVML should not be initialized without nvidia-smi
    assert monitor._nvml_initialised is False
    
    metrics = monitor._collect_metrics()
    
    # Check GPU metrics from PyTorch
    assert "gpu" in metrics
    assert metrics["gpu"]["name"] == "NVIDIA RTX 3080"
    assert metrics["gpu"]["memory_total"] == 8 * 1024 * 1024 * 1024
    assert metrics["gpu"]["memory_used"] == 2 * 1024 * 1024 * 1024
    assert metrics["gpu"]["temperature"] is None
    
    print("✅ Metric collection with PyTorch CUDA fallback works")


# Test async monitoring loop lifecycle
@pytest.mark.asyncio
@patch('src.backend.system.detect_compute_device')
@patch('src.backend.system.psutil')
async def test_monitoring_loop_lifecycle(mock_psutil, mock_detect):
    """Test async monitoring loop start, run, and stop."""
    mock_detect.return_value = ComputeDeviceInfo(
        backend="CPU",
        name="Intel i7",
        reason="Test"
    )
    
    mock_psutil.cpu_percent.return_value = 45.5
    mock_psutil.cpu_count.return_value = 8
    mock_psutil.cpu_freq.return_value = MockCPUFreq()
    mock_psutil.virtual_memory.return_value = MockVirtualMemory()
    mock_psutil.swap_memory.return_value = MockSwapMemory()
    
    bridge = MockBridge()
    monitor = SystemMonitor(bridge, interval=0.1)
    
    # Start monitoring
    await monitor.start()
    
    assert monitor._running is True
    assert monitor._task is not None
    assert len(bridge.messages) == 1  # Initial metrics published
    assert bridge.messages[0]["type"] == "system_metrics"
    
    # Let it run for a bit
    await asyncio.sleep(0.25)
    
    # Should have more messages now
    assert len(bridge.messages) >= 2
    
    # Stop monitoring
    await monitor.stop()
    
    assert monitor._running is False
    assert monitor._task is None
    
    print("✅ Async monitoring loop lifecycle works")


# Test monitoring loop doesn't start twice
@pytest.mark.asyncio
@patch('src.backend.system.detect_compute_device')
@patch('src.backend.system.psutil')
async def test_monitoring_loop_no_double_start(mock_psutil, mock_detect):
    """Test that monitoring loop doesn't start twice."""
    mock_detect.return_value = ComputeDeviceInfo(
        backend="CPU",
        name="Intel i7",
        reason="Test"
    )
    
    mock_psutil.cpu_percent.return_value = 45.5
    mock_psutil.cpu_count.return_value = 8
    mock_psutil.cpu_freq.return_value = MockCPUFreq()
    mock_psutil.virtual_memory.return_value = MockVirtualMemory()
    mock_psutil.swap_memory.return_value = MockSwapMemory()
    
    bridge = MockBridge()
    monitor = SystemMonitor(bridge, interval=0.1)
    
    # Start monitoring
    await monitor.start()
    task1 = monitor._task
    
    # Try to start again
    await monitor.start()
    task2 = monitor._task
    
    # Should be the same task
    assert task1 == task2
    
    await monitor.stop()
    
    print("✅ Monitoring loop prevents double start")


# Test monitoring gracefully handles cancellation
@pytest.mark.asyncio
@patch('src.backend.system.detect_compute_device')
@patch('src.backend.system.psutil')
async def test_monitoring_cancellation(mock_psutil, mock_detect):
    """Test monitoring loop handles cancellation gracefully."""
    mock_detect.return_value = ComputeDeviceInfo(
        backend="CPU",
        name="Intel i7",
        reason="Test"
    )
    
    mock_psutil.cpu_percent.return_value = 45.5
    mock_psutil.cpu_count.return_value = 8
    mock_psutil.cpu_freq.return_value = MockCPUFreq()
    mock_psutil.virtual_memory.return_value = MockVirtualMemory()
    mock_psutil.swap_memory.return_value = MockSwapMemory()
    
    bridge = MockBridge()
    monitor = SystemMonitor(bridge, interval=0.1)
    
    await monitor.start()
    
    # Stop should handle cancellation gracefully
    await monitor.stop()
    
    # Should complete without errors
    assert monitor._running is False
    
    print("✅ Monitoring loop handles cancellation gracefully")


# Test NVML shutdown on stop
@pytest.mark.asyncio
@patch('src.backend.system.nvmlShutdown')
@patch('src.backend.system.nvmlInit')
@patch('src.backend.system.nvmlDeviceGetCount')
@patch('src.backend.system.nvmlDeviceGetHandleByIndex')
@patch('src.backend.system.nvmlDeviceGetName')
@patch('src.backend.system.detect_compute_device')
@patch('src.backend.system.psutil')
@patch('src.backend.system.shutil.which')
async def test_nvml_shutdown_on_stop(
    mock_which, mock_psutil, mock_detect, mock_get_name, mock_get_handle,
    mock_get_count, mock_nvml_init, mock_nvml_shutdown
):
    """Test NVML is properly shut down when monitoring stops."""
    mock_which.return_value = "/usr/bin/nvidia-smi"
    mock_detect.return_value = ComputeDeviceInfo(
        backend="GPU",
        name="NVIDIA RTX 3080",
        reason="CUDA available",
        cuda_ready=True
    )
    
    mock_get_count.return_value = 1
    mock_get_name.return_value = b"NVIDIA RTX 3080"
    
    mock_psutil.cpu_percent.return_value = 45.5
    mock_psutil.cpu_count.return_value = 8
    mock_psutil.cpu_freq.return_value = MockCPUFreq()
    mock_psutil.virtual_memory.return_value = MockVirtualMemory()
    mock_psutil.swap_memory.return_value = MockSwapMemory()
    
    bridge = MockBridge()
    monitor = SystemMonitor(bridge, interval=0.1)
    
    assert monitor._nvml_initialised is True
    
    await monitor.start()
    await monitor.stop()
    
    # NVML should be shut down
    mock_nvml_shutdown.assert_called_once()
    assert monitor._nvml_initialised is False
    
    print("✅ NVML properly shuts down on stop")


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("  Testing System Monitoring Module")
    print("=" * 60)
    print()
    
    # Synchronous tests
    sync_tests = [
        test_compute_device_info_cpu,
        test_compute_device_info_gpu,
        test_detect_compute_device_no_cuda,
        test_detect_compute_device_with_cuda,
        test_detect_compute_device_nvidia_smi_no_runtime,
        test_detect_compute_device_cuda_exception,
        test_system_monitor_init,
        test_system_monitor_nvml_init_success,
        test_system_monitor_nvml_init_failure,
        test_collect_metrics_cpu_only,
        test_collect_metrics_with_nvml,
        test_collect_metrics_torch_fallback,
    ]
    
    # Async tests
    async_tests = [
        test_monitoring_loop_lifecycle,
        test_monitoring_loop_no_double_start,
        test_monitoring_cancellation,
        test_nvml_shutdown_on_stop,
    ]
    
    # Run synchronous tests
    for test_func in sync_tests:
        try:
            test_func()
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            raise
    
    # Run async tests
    async def run_async_tests():
        for test_func in async_tests:
            try:
                await test_func()
            except Exception as e:
                print(f"❌ {test_func.__name__} failed: {e}")
                raise
    
    asyncio.run(run_async_tests())
    
    print()
    print("=" * 60)
    print(f"  All {len(sync_tests) + len(async_tests)} Tests Passed!")
    print("=" * 60)
    print()
    print("Test coverage includes:")
    print("  ✅ Device detection with/without CUDA")
    print("  ✅ Metric collection with mock psutil data")
    print("  ✅ NVML initialization failure handling")
    print("  ✅ Async monitoring loop lifecycle")
    print("  ✅ GPU fallback to PyTorch when NVML unavailable")
    print("  ✅ Proper cleanup and shutdown")


if __name__ == "__main__":
    run_tests()
