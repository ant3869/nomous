# Title: Config Loader
# Path: backend/config.py
# Purpose: Load YAML config and lock keys expected by runtime.

import os, yaml
from pathlib import Path

# Get project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

DEFAULT = {
    "paths": {
        "gguf_path": "modules/models/local.gguf",
        "embed_gguf_path": "modules/models/embed/bge-small-en-v1.5-f16.gguf",
        "vosk_model_dir": "modules/models/vosk-model-small-en-us-0.15",
        "piper_exe": "modules/piper/piper.exe",
        "piper_voice": "modules/piper/voices/en_US-libritts-high.onnx",  # Updated to match actual file
        "piper_out_dir": "modules/piper/piper_out"
    },
    "llm": {"enable": True, "n_ctx": 2048, "n_threads": 4, "temperature": 0.6, "top_p": 0.95, "tools_enabled": True},
    "audio": {"sample_rate": 16000, "vad_aggressiveness": 2, "chunk_ms": 250, "device": "default"},
    "camera": {"backend": "dshow", "index": 0, "width": 1280, "height": 720},
    "ws": {"host": "127.0.0.1", "port": 8765},
    "memory": {"enable": True, "db_path": "modules/memory/rag.sqlite"},
    "rag": {"enable": True, "db_path": "modules/memory/rag/corpus"},
    "ui": {"snapshot_debounce": 4, "motion_sensitivity": 30, "tts_enabled": True, "vision_enabled": True}
}

def load_config(path: str="config.yaml"):
    data = DEFAULT.copy()
    config_path = Path(PROJECT_ROOT) / path
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
            for k, v in DEFAULT.items():
                if isinstance(v, dict):
                    data[k] = {**v, **(y.get(k, {}) or {})}
                else:
                    data[k] = y.get(k, v)
    
    # Make paths absolute relative to project root
    for key, value in data["paths"].items():
        if not os.path.isabs(value):  # If it's not already an absolute path
            data["paths"][key] = str(Path(PROJECT_ROOT) / value)
    
    # Create output directory
    os.makedirs(data["paths"]["piper_out_dir"], exist_ok=True)
    return data
