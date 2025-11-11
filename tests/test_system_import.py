import builtins
import importlib
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings(
    "ignore",
    message=r".*pynvml package is deprecated.*",
    category=FutureWarning,
)


@pytest.mark.filterwarnings("ignore:.*pynvml package is deprecated.*:FutureWarning")
def test_detect_compute_device_handles_torch_import_error(monkeypatch):
    module_name = "src.backend.system"
    sys.modules.pop(module_name, None)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("numpy source tree")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    system = importlib.import_module(module_name)

    assert system.TORCH_AVAILABLE is False
    assert system.TORCH_IMPORT_ERROR is not None

    device_info = system.detect_compute_device()

    assert device_info.backend == "CPU"
    assert "PyTorch unavailable" in device_info.reason
    assert "numpy source tree" in device_info.reason
