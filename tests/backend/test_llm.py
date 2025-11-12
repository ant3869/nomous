import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "llama_cpp" not in sys.modules:
    llama_stub = types.ModuleType("llama_cpp")
    llama_stub.Llama = object
    sys.modules["llama_cpp"] = llama_stub

from src.backend.llm import LocalLLM


def sanitize(text: str) -> str:
    instance = LocalLLM.__new__(LocalLLM)
    return LocalLLM._sanitize_response(instance, text)


def test_sanitize_removes_explicit_scheduling_tokens_with_equals():
    text = "I'll handle this respond_in=5s before continuing."
    sanitized = sanitize(text)
    assert "respond_in=5s" not in sanitized
    assert "I'll handle this before continuing." == sanitized


def test_sanitize_removes_explicit_scheduling_tokens_with_words():
    text = "Set a reminder to respond-in 10 seconds, then proceed."
    sanitized = sanitize(text)
    assert "respond-in 10 seconds" not in sanitized
    assert "Set a reminder to  then proceed." == sanitized


def test_sanitize_preserves_non_scheduling_phrase():
    text = "Please respond in 3 steps so I can follow along."
    sanitized = sanitize(text)
    assert "respond in 3 steps" in sanitized
    assert sanitized.endswith("follow along.")
