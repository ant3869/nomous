# Title: GPU Performance Profiler
# Path: backend/gpu_profiler.py
# Purpose: Profile GPU usage, memory, and performance metrics

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


@dataclass
class GPUMetrics:
    """GPU performance metrics."""
    memory_allocated_mb: float
    memory_reserved_mb: float
    memory_free_mb: float
    utilization_percent: Optional[float]
    temperature_c: Optional[float]
    inference_time_ms: float


class GPUProfiler:
    """Profile GPU performance for ML operations."""
    
    def __init__(self):
        self.enabled = TORCH_AVAILABLE and torch.cuda.is_available()
        self._last_metrics: Optional[GPUMetrics] = None
        
        if self.enabled:
            self.device_count = torch.cuda.device_count()
            self.device_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU Profiler initialized: {self.device_name}")
        else:
            self.device_count = 0
            self.device_name = "CPU"
            logger.info("GPU Profiler: Running on CPU")
    
    def start_inference(self) -> float:
        """Mark the start of an inference operation."""
        return time.time()
    
    def end_inference(self, start_time: float) -> GPUMetrics:
        """Mark the end of inference and collect metrics."""
        inference_time_ms = (time.time() - start_time) * 1000
        
        if not self.enabled:
            return GPUMetrics(
                memory_allocated_mb=0,
                memory_reserved_mb=0,
                memory_free_mb=0,
                utilization_percent=None,
                temperature_c=None,
                inference_time_ms=inference_time_ms
            )
        
        try:
            # Get memory stats
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            free = total - allocated
            
            # Try to get utilization (may not be available on all systems)
            utilization = None
            temperature = None
            
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util_rates.gpu
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                pynvml.nvmlShutdown()
            except Exception:
                pass  # NVML not available
            
            metrics = GPUMetrics(
                memory_allocated_mb=allocated,
                memory_reserved_mb=reserved,
                memory_free_mb=free,
                utilization_percent=utilization,
                temperature_c=temperature,
                inference_time_ms=inference_time_ms
            )
            
            self._last_metrics = metrics
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to collect GPU metrics: {e}")
            return GPUMetrics(
                memory_allocated_mb=0,
                memory_reserved_mb=0,
                memory_free_mb=0,
                utilization_percent=None,
                temperature_c=None,
                inference_time_ms=inference_time_ms
            )
    
    def optimize_memory(self):
        """Optimize GPU memory usage."""
        if not self.enabled:
            return
        
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Try to defragment memory
            if hasattr(torch.cuda, 'memory_summary'):
                logger.debug("GPU memory cleared and optimized")
        except Exception as e:
            logger.warning(f"GPU memory optimization failed: {e}")
    
    def get_memory_summary(self) -> dict:
        """Get detailed memory usage summary."""
        if not self.enabled:
            return {"status": "CPU mode"}
        
        try:
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            
            return {
                "device": self.device_name,
                "allocated_mb": round(allocated, 2),
                "reserved_mb": round(reserved, 2),
                "total_mb": round(total, 2),
                "free_mb": round(total - allocated, 2),
                "utilization_pct": round((allocated / total) * 100, 1)
            }
        except Exception as e:
            logger.error(f"Failed to get memory summary: {e}")
            return {"error": str(e)}
    
    def log_metrics(self, metrics: GPUMetrics):
        """Log performance metrics."""
        if metrics.utilization_percent is not None:
            logger.info(
                f"GPU: {metrics.utilization_percent}% util, "
                f"{metrics.memory_allocated_mb:.1f}MB allocated, "
                f"{metrics.inference_time_ms:.1f}ms inference"
            )
        else:
            logger.info(
                f"Inference: {metrics.inference_time_ms:.1f}ms, "
                f"Memory: {metrics.memory_allocated_mb:.1f}MB allocated"
            )


# Global profiler instance
profiler = GPUProfiler()
