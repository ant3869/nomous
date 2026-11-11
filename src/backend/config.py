"""Configuration loader used by the runtime services.

The original implementation relied on ad-hoc dictionary copies, inline imports
and shallow merging which made it surprisingly easy to end up with partially
initialised configuration (for example nested dictionaries were mutated across
calls).  This module now exposes a small ``ConfigLoader`` helper that performs
deep merges, path normalisation and basic validation while keeping the public
``load_config`` function backwards compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT: Dict[str, Any] = {
    "paths": {
        "gguf_path": "modules/models/local.gguf",
        "mmproj_path": "",
        "embed_gguf_path": "modules/models/embed/bge-small-en-v1.5-f16.gguf",
        "vosk_model_dir": "modules/models/vosk-model-small-en-us-0.15",
        "piper_exe": "modules/piper/piper.exe",
        "piper_voice": "modules/piper/voices/en_US-libritts-high.onnx",
        "piper_out_dir": "modules/piper/piper_out",
    },
    "llm": {
        "enable": True,
        "n_ctx": 16384,
        "n_threads": 8,
        "temperature": 0.6,
        "top_p": 0.95,
        "tools_enabled": True,
        "prompt_style": "auto",
        "show_thinking": True,
        "system_prompt": (
            "You are Nomous, an autonomous multimodal AI orchestrator. "
            "Support the operator with concise, confident guidance, coordinate "
            "sensors and tools, and narrate decisions with a collaborative tone."
        ),
        "thinking_prompt": (
            "Think in small, verifiable steps. Reference available tools and "
            "memories, note uncertainties, and decide on an action plan before "
            "committing to a response. Keep thoughts structured and actionable."
        ),
    },
    "vision": {
        "enabled": True,
        "model": "qwen3-vl-8b",
        "image_max_tokens": 1024,
        "frame_width": 640,
        "enable_thinking": False,
        "show_thinking": True,
        "thinking_max_tokens": 128,
        "frame_memory_items": 5,
        "frame_memory_max_tokens": 512,
        "temperature": 0.5,
        # Single source of truth for the vision cadence (finding 5): the
        # LLM's process_vision cooldown and the camera loop's analysis
        # cooldown both read vision.vision_cooldown. There is no
        # autonomous.vision_cooldown or ui.vision_cooldown anymore.
        "vision_cooldown": 15,
        # How often to describe the room when the scene is static
        # (finding 3). 0 disables the digest.
        "scene_digest_minutes": 10,
        # Minimum cosine similarity between face embeddings to call two
        # faces the same person (finding 4). MobileNetV3 embeddings are
        # generic image vectors, not face-specific — tune on real faces.
        "embedding_match_threshold": 0.7,
    },
    "audio": {
        "sample_rate": 16000,
        "vad_aggressiveness": 2,
        "chunk_ms": 250,
        "device": "default",
    },
    "hearing": {
        # Ambient hearing — the host mic as a sense on its own, alongside
        # vision (NOT chat STT; that's audio.* + the dashboard mic toggle).
        # Opt-in: off until the user enables it (dashboard toggle or here).
        "enabled": False,
        "device": "default",
        "sample_rate": 16000,
        # webrtcvad aggressiveness 0-3 (higher = stricter, fewer false hits).
        # Vision boosts sensitivity by one step for a while after it sees
        # something new (listen_boost_seconds).
        "vad_aggressiveness": 2,
        # Silence that ends a speech segment; segments cap at max_segment_s.
        "segment_silence_ms": 500,
        "max_segment_s": 10,
        # Ignore transcripts shorter than this (VAD false hits, single words).
        "min_transcript_chars": 3,
        # RMS (0-1) of a loud non-speech burst (door slam, dropped object).
        "noise_rms_threshold": 0.35,
        # Persist audio-derived idle-tick lines to memory. Raw audio is
        # never written to disk either way; transcripts are transient.
        "retain_transcripts": True,
        # Seconds a heard thing stays fresh for the idle tick.
        "evidence_ttl_s": 60,
        # How long vision boosts hearing sensitivity after a new sighting.
        "listen_boost_seconds": 30,
    },
    "camera": {"backend": "dshow", "index": 0, "width": 1280, "height": 720},
    "ws": {"host": "127.0.0.1", "port": 8765},
    "memory": {"enable": True, "db_path": "modules/memory/rag.sqlite"},
    "rag": {"enable": True, "db_path": "modules/memory/rag/corpus"},
    "ui": {
        "preview_interval_sec": 0.5,
        "motion_sensitivity": 30,
        "tts_enabled": True,
        "vision_enabled": True,
        "gesture_enabled": True,
        "gesture_cooldown": 3,
    },
}


class ConfigError(RuntimeError):
    """Raised when the configuration file cannot be processed."""


def _deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> None:
    """Recursively merge ``updates`` into ``base``.

    The helper keeps mutable values isolated by copying nested dictionaries
    before modifying them.  This prevents callers from accidentally mutating the
    module level ``DEFAULT`` mapping.
    """

    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            nested = copy.deepcopy(base[key])
            _deep_update(nested, value)
            base[key] = nested
        else:
            base[key] = copy.deepcopy(value)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle) or {}
            if not isinstance(content, Mapping):
                raise ConfigError(
                    f"Configuration file {path} must contain a mapping at the root level."
                )
            return content
    except OSError as exc:
        raise ConfigError(f"Unable to read configuration file {path}: {exc}") from exc
    except yaml.YAMLError as exc:  # pragma: no cover - yaml reports are descriptive
        raise ConfigError(f"Unable to parse YAML configuration: {exc}") from exc
def _normalise_paths(paths: Mapping[str, Any]) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for key, value in paths.items():
        if isinstance(value, str) and value:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = PROJECT_ROOT / candidate
            resolved[key] = str(candidate.resolve())
    return resolved


def _ensure_directories(paths: Iterable[str]) -> None:
    for candidate in paths:
        if candidate:
            os.makedirs(candidate, exist_ok=True)


@dataclass(slots=True)
class ConfigLoader:
    """Load and normalise runtime configuration files."""

    path: Path = field(default_factory=lambda: PROJECT_ROOT / "config.yaml")
    defaults: Mapping[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT))

    def load(self, override_path: Optional[str | os.PathLike[str]] = None) -> Dict[str, Any]:
        """Return a deeply merged configuration dictionary."""

        config_path = self._resolve_path(override_path)
        data = copy.deepcopy(self.defaults)
        overrides = _load_yaml(config_path)
        if overrides:
            _deep_update(data, overrides)

        paths = data.get("paths", {})
        if not isinstance(paths, Mapping):
            raise ConfigError("Configuration entry 'paths' must be a mapping")

        resolved_paths = _normalise_paths(paths)
        data["paths"] = {**paths, **resolved_paths}
        _ensure_directories([data["paths"].get("piper_out_dir", "")])

        return data

    def _resolve_path(self, override_path: Optional[str | os.PathLike[str]]) -> Path:
        if override_path:
            candidate = Path(override_path)
        else:
            candidate = self.path

        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        return candidate


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Backward compatible convenience wrapper."""

    loader = ConfigLoader()
    return loader.load(path)


__all__ = ["ConfigError", "ConfigLoader", "DEFAULT", "load_config"]
