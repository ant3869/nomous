from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backend import config as config_module


def create_sample_config(tmp_path: Path, payload: Dict[str, Any]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_module.yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_loader_returns_deepcopy_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loader = config_module.ConfigLoader()
    # Force the loader to read from a temporary project root to avoid touching
    # the repository level configuration.
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)

    config = loader.load("non-existent.yaml")

    assert config["audio"]["sample_rate"] == config_module.DEFAULT["audio"]["sample_rate"]
    assert Path(config["paths"]["piper_out_dir"]).is_absolute()
    # Modifying the returned mapping must not mutate the module level default.
    config["audio"]["sample_rate"] = 1234
    assert config_module.DEFAULT["audio"]["sample_rate"] != 1234


def test_loader_merges_nested_dictionaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    payload = {
        "audio": {"sample_rate": 8000},
        "paths": {"piper_out_dir": "artifacts"},
    }
    path = create_sample_config(tmp_path, payload)

    loader = config_module.ConfigLoader()
    config = loader.load(path)

    assert config["audio"]["sample_rate"] == 8000
    assert config["audio"]["device"] == config_module.DEFAULT["audio"]["device"]
    assert Path(config["paths"]["piper_out_dir"]).is_absolute()


def test_loader_rejects_invalid_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    path = create_sample_config(tmp_path, ["not", "a", "mapping"])

    loader = config_module.ConfigLoader()
    with pytest.raises(config_module.ConfigError):
        loader.load(path)


def test_loader_rejects_non_mapping_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    payload = {"paths": "invalid"}
    path = create_sample_config(tmp_path, payload)

    loader = config_module.ConfigLoader()
    with pytest.raises(config_module.ConfigError):
        loader.load(path)
