"""Utilities for detecting compute devices and streaming system metrics."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import psutil


logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency varies per host
    import pynvml  # type: ignore
except Exception as exc:  # pragma: no cover - handled gracefully below
    pynvml = None  # type: ignore[assignment]
    NVML_AVAILABLE = False
    NVML_IMPORT_ERROR: Optional[Exception] = exc
    logger.debug("NVML import failed: %s", exc)
else:
    NVML_AVAILABLE = True
    NVML_IMPORT_ERROR = None

if NVML_AVAILABLE and pynvml is not None:
    nvmlInit = pynvml.nvmlInit
    nvmlShutdown = pynvml.nvmlShutdown
    nvmlDeviceGetCount = pynvml.nvmlDeviceGetCount
    nvmlDeviceGetHandleByIndex = pynvml.nvmlDeviceGetHandleByIndex
    nvmlDeviceGetName = pynvml.nvmlDeviceGetName
    nvmlDeviceGetMemoryInfo = pynvml.nvmlDeviceGetMemoryInfo
    nvmlDeviceGetUtilizationRates = pynvml.nvmlDeviceGetUtilizationRates
    nvmlDeviceGetTemperature = pynvml.nvmlDeviceGetTemperature
    NVML_TEMPERATURE_GPU = pynvml.NVML_TEMPERATURE_GPU
    NVMLError = pynvml.NVMLError
    nvmlSystemGetCudaDriverVersion_v2 = getattr(
        pynvml, "nvmlSystemGetCudaDriverVersion_v2", None
    )
else:  # pragma: no cover - NVML missing in CI
    NVML_TEMPERATURE_GPU = None

    class NVMLError(Exception):
        """Fallback NVML error type when the library is unavailable."""

    def _nvml_missing(*_args, **_kwargs):  # pragma: no cover - runtime guard
        raise RuntimeError("NVML library is not available on this host")

    nvmlInit = nvmlShutdown = _nvml_missing
    nvmlDeviceGetCount = nvmlDeviceGetHandleByIndex = _nvml_missing
    nvmlDeviceGetName = nvmlDeviceGetMemoryInfo = _nvml_missing
    nvmlDeviceGetUtilizationRates = nvmlDeviceGetTemperature = _nvml_missing
    nvmlSystemGetCudaDriverVersion_v2 = None


try:  # pragma: no cover - success path depends on local environment
    import torch
except Exception as exc:  # pragma: no cover - exercised in unit tests via monkeypatch
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR: Optional[Exception] = exc
    logger.warning(
        "PyTorch import failed (%s). GPU metrics will be unavailable.",
        exc,
    )
else:
    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = None


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


def _probe_nvml_device() -> Optional[Tuple[str, int, Optional[str]]]:
    """Return ``(name, count, cuda_version)`` when NVML can see an NVIDIA GPU."""

    if not NVML_AVAILABLE or pynvml is None:
        return None

    try:
        nvmlInit()
        count = nvmlDeviceGetCount()
        if count <= 0:
            return None

        handle = nvmlDeviceGetHandleByIndex(0)
        raw_name = nvmlDeviceGetName(handle)
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)

        cuda_version: Optional[str] = None
        if nvmlSystemGetCudaDriverVersion_v2 is not None:
            with contextlib.suppress(Exception):
                version = nvmlSystemGetCudaDriverVersion_v2()
                if version:
                    major = version // 1000
                    minor = (version - major * 1000) // 10
                    cuda_version = f"{major}.{minor}"

        return name or "NVIDIA GPU", int(count), cuda_version
    except Exception as exc:  # pragma: no cover - depends on host driver state
        logger.debug("NVML probe failed: %s", exc)
        return None
    finally:
        with contextlib.suppress(Exception):
            if NVML_AVAILABLE and pynvml is not None:
                nvmlShutdown()


def detect_compute_device() -> ComputeDeviceInfo:
    """Inspect the host system and return the active compute backend."""

    backend = "CPU"
    name = platform.processor() or "CPU"
    nvml_probe = _probe_nvml_device()

    if not TORCH_AVAILABLE:
        import_error = TORCH_IMPORT_ERROR
        if nvml_probe:
            gpu_name, gpu_count, cuda_version = nvml_probe
            detail = f" ({import_error})" if import_error else ""
            reason = (
                "PyTorch unavailable"
                f"{detail} but NVML detected an NVIDIA GPU. Install CUDA-enabled bindings to"
                " enable acceleration."
            )
            return ComputeDeviceInfo(
                backend="GPU",
                name=gpu_name,
                reason=reason,
                cuda_version=cuda_version,
                gpu_count=gpu_count,
            )

        error_detail = f": {import_error}" if import_error else ""
        reason = f"PyTorch unavailable{error_detail}. Falling back to CPU monitoring only."
        return ComputeDeviceInfo(
            backend=backend,
            name=name,
            reason=reason,
            cuda_version=None,
            gpu_count=0,
        )

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
            if nvml_probe:
                gpu_name, gpu_count, cuda_version = nvml_probe
                name = gpu_name
                backend = "GPU"
                reason = (
                    "NVML detected an NVIDIA GPU but PyTorch cannot use CUDA. Install the "
                    "CUDA toolkit or GPU-enabled PyTorch build to activate acceleration."
                )
            elif shutil.which("nvidia-smi"):
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
        self._nvml_initialised = False
        self._nvml_device_index = 0
        self._gpu_name_override: Optional[str] = None
        self._nvml_active_once = False

        # Prime CPU utilization so subsequent reads are meaningful
        psutil.cpu_percent(interval=None)
        self._initialize_nvml()

    def _set_nvml_initialized(self, value: bool) -> None:
        self._nvml_initialized = value
        self._nvml_initialised = value

    def _initialize_nvml(self) -> None:
        if not NVML_AVAILABLE or pynvml is None:
            if NVML_IMPORT_ERROR and self.device_info.is_gpu:
                self.device_info.reason = (
                    f"NVML unavailable: {NVML_IMPORT_ERROR}. GPU metrics will be limited."
                )
            return

        should_attempt = self.device_info.is_gpu or shutil.which("nvidia-smi")
        if not should_attempt:
            return

        try:
            nvmlInit()
            self._set_nvml_initialized(True)
            self._nvml_active_once = True
            count = nvmlDeviceGetCount()
            if count:
                self._nvml_device_index = 0
                handle = nvmlDeviceGetHandleByIndex(self._nvml_device_index)
                gpu_name = nvmlDeviceGetName(handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode("utf-8")
                self._gpu_name_override = gpu_name
        except NVMLError as exc:
            self._set_nvml_initialized(False)
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
        if self._nvml_initialized or self._nvml_active_once:
            with contextlib.suppress(Exception):  # pragma: no cover - shutdown errors are non-fatal
                nvmlShutdown()
            self._set_nvml_initialized(False)
            self._nvml_active_once = False

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
                memory_percent = (float(mem.used) / float(mem.total) * 100.0) if mem.total else 0.0
                return {
                    "name": name,
                    "percent": float(util.gpu),
                    "memory_total": int(mem.total),
                    "memory_used": int(mem.used),
                    "memory_percent": memory_percent,
                    "temperature": float(temperature),
                }
            except NVMLError as exc:
                self._set_nvml_initialized(False)
                self.device_info.reason = f"NVML metrics unavailable: {exc}"
                return None

        if self.device_info.is_gpu and TORCH_AVAILABLE and torch is not None:
            try:
                free, total = torch.cuda.mem_get_info(0)
            except Exception:
                return None
            used = total - free
            memory_percent = (float(used) / float(total) * 100.0) if total else 0.0
            return {
                "name": self.device_info.name,
                "percent": 0.0,
                "memory_total": int(total),
                "memory_used": int(used),
                "memory_percent": memory_percent,
                "temperature": None,
            }

        return None
