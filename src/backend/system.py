"""Utilities for detecting compute devices and streaming system metrics."""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import psutil
import torch
from pynvml import (
    NVMLError,
    NVML_TEMPERATURE_GPU,
    nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlDeviceGetTemperature,
    nvmlDeviceGetUtilizationRates,
    nvmlInit,
    nvmlShutdown,
)


@dataclass
class ComputeDeviceInfo:
    """Represents the compute backend currently in use."""

    backend: str
    name: str
    reason: str
    cuda_version: Optional[str] = None
    gpu_count: int = 0

    @property
    def is_gpu(self) -> bool:
        return self.backend.upper() == "GPU"


def detect_compute_device() -> ComputeDeviceInfo:
    """Inspect the host system and return the active compute backend."""

    backend = "CPU"
    name = platform.processor() or "CPU"
    reason = "No CUDA-capable GPU detected; defaulting to CPU."
    cuda_version: Optional[str] = None
    gpu_count = 0

    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            backend = "GPU"
            cuda_version = torch.version.cuda
            reason = "CUDA runtime detected and PyTorch GPU build is active."
        else:
            if shutil.which("nvidia-smi"):
                reason = (
                    "NVIDIA GPU detected but CUDA drivers or PyTorch GPU build are not available."
                )
            elif torch.version.cuda:
                reason = "PyTorch was compiled with CUDA support but no GPU is currently accessible."
    except Exception as exc:  # pragma: no cover - hardware specific
        reason = f"Unable to query CUDA runtime: {exc}"

    return ComputeDeviceInfo(
        backend=backend,
        name=name,
        reason=reason,
        cuda_version=cuda_version,
        gpu_count=gpu_count,
    )


class SystemMonitor:
    """Stream CPU, memory, and GPU utilization metrics to the UI."""

    def __init__(self, bridge, interval: float = 2.0):
        self.bridge = bridge
        self.interval = interval
        self.device_info = detect_compute_device()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._nvml_initialized = False
        self._nvml_device_index = 0
        self._gpu_name_override: Optional[str] = None

        # Prime CPU utilization so subsequent reads are meaningful
        psutil.cpu_percent(interval=None)
        self._initialize_nvml()

    def _initialize_nvml(self) -> None:
        should_attempt = self.device_info.is_gpu or shutil.which("nvidia-smi")
        if not should_attempt:
            return

        try:
            nvmlInit()
            self._nvml_initialized = True
            count = nvmlDeviceGetCount()
            if count:
                self._nvml_device_index = 0
                handle = nvmlDeviceGetHandleByIndex(self._nvml_device_index)
                self._gpu_name_override = nvmlDeviceGetName(handle).decode("utf-8")
        except NVMLError as exc:
            self._nvml_initialized = False
            self.device_info.reason = f"NVML unavailable: {exc}"

    async def start(self) -> None:
        if self._task:
            return
        self._running = True
        await self._publish_metrics()
        self._task = asyncio.create_task(self._run(), name="system-metrics")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                # Task cancellation is expected during shutdown; safe to ignore.
                pass
            self._task = None
        if self._nvml_initialized:
            try:
                nvmlShutdown()
            except NVMLError:  # pragma: no cover - shutdown errors are non-fatal
                pass
            self._nvml_initialized = False

    async def _run(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self.interval)
                await self._publish_metrics()
        except asyncio.CancelledError:
            return

    async def _publish_metrics(self) -> None:
        payload = self._collect_metrics()
        await self.bridge.post({"type": "system_metrics", "payload": payload})

    def _collect_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "timestamp": time.time(),
            "device": asdict(self.device_info),
            "cpu": self._cpu_metrics(),
            "memory": self._memory_metrics(),
        }

        gpu_metrics = self._gpu_metrics()
        if gpu_metrics:
            metrics["gpu"] = gpu_metrics

        swap = psutil.swap_memory()
        metrics["swap"] = {
            "total": int(swap.total),
            "used": int(swap.used),
            "percent": float(swap.percent),
        }

        return metrics

    @staticmethod
    def _cpu_metrics() -> Dict[str, Any]:
        freq = psutil.cpu_freq()
        metrics: Dict[str, Any] = {
            "percent": float(psutil.cpu_percent(interval=None)),
            "cores": psutil.cpu_count(logical=True),
            "frequency": float(freq.current) if freq else None,
        }
        if hasattr(os, "getloadavg"):
            try:
                metrics["load"] = list(os.getloadavg())
            except OSError:
                metrics["load"] = None
        return metrics

    @staticmethod
    def _memory_metrics() -> Dict[str, Any]:
        mem = psutil.virtual_memory()
        return {
            "total": int(mem.total),
            "used": int(mem.used),
            "available": int(mem.available),
            "percent": float(mem.percent),
        }

    def _gpu_metrics(self) -> Optional[Dict[str, Any]]:
        if self._nvml_initialized:
            try:
                handle = nvmlDeviceGetHandleByIndex(self._nvml_device_index)
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)
                name = self._gpu_name_override or self.device_info.name
                temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
                return {
                    "name": name,
                    "percent": float(util.gpu),
                    "memory_total": int(mem.total),
                    "memory_used": int(mem.used),
                    "memory_percent": (float(mem.used) / float(mem.total) * 100.0)
                    if mem.total
                    else 0.0,
                    "temperature": float(temperature),
                }
            except NVMLError as exc:
                self.device_info.reason = f"NVML error: {exc}"
                return None
        if self.device_info.is_gpu:
            try:
                free, total = torch.cuda.mem_get_info(0)
                used = total - free
                return {
                    "name": self.device_info.name,
                    "percent": 0.0,
                    "memory_total": int(total),
                    "memory_used": int(used),
                    "memory_percent": (float(used) / float(total) * 100.0) if total else 0.0,
                    "temperature": None,
                }
            except Exception:
                return None
        return None
