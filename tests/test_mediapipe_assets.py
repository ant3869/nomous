#!/usr/bin/env python3
"""Tests for vendored MediaPipe assets (finding 8).

Covers: the manifest pins a URL + sha256 per model, every vendored file
matches its manifest checksum, resolve_model prefers the vendored copy
without downloading, and checksum mismatches are caught.
"""

import hashlib
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.backend import mediapipe_assets


def test_manifest_pins_url_and_sha256_per_model():
    manifest = mediapipe_assets.load_manifest()
    assert set(manifest) == {
        "gesture_recognizer.task",
        "blaze_face_short_range.tflite",
        "mobilenet_v3_small.tflite",
    }
    for filename, entry in manifest.items():
        assert entry["url"].startswith("https://storage.googleapis.com/mediapipe-models/")
        assert len(entry["sha256"]) == 64
        assert entry["sha256"] == entry["sha256"].lower()


def test_vendored_files_match_manifest():
    """The files actually vendored in git match the pinned checksums."""
    manifest = mediapipe_assets.load_manifest()
    for filename, entry in manifest.items():
        path = mediapipe_assets.vendored_dir() / filename
        assert path.exists(), f"vendored model missing: {filename}"
        assert mediapipe_assets.sha256_file(path) == entry["sha256"]


def test_verify_all_vendored_reports_all_ok():
    results = mediapipe_assets.verify_all_vendored()
    assert results, "no models verified"
    assert all(results.values()), results


def test_verify_file_detects_corruption(tmp_path):
    good = tmp_path / "model.tflite"
    good.write_bytes(b"model-bytes")
    digest = hashlib.sha256(b"model-bytes").hexdigest()
    assert mediapipe_assets.verify_file(good, digest) is True
    assert mediapipe_assets.verify_file(good, "0" * 64) is False
    assert mediapipe_assets.verify_file(tmp_path / "missing.tflite", digest) is False
    assert mediapipe_assets.verify_file(good, None) is True


def test_resolve_model_prefers_vendored_without_downloading(monkeypatch):
    """resolve_model returns the vendored copy and never hits the network."""
    calls = []
    monkeypatch.setattr(
        mediapipe_assets, "_download_file",
        lambda url, target: calls.append(url),
    )
    path = mediapipe_assets.resolve_model("gesture_recognizer.task")
    assert path is not None
    assert path.parent == mediapipe_assets.vendored_dir()
    assert calls == []


def test_resolve_model_rejects_corrupt_vendored_copy(monkeypatch, tmp_path):
    """A vendored file that fails its checksum is ignored (download off)."""
    vendored = tmp_path / "vendored"
    legacy = tmp_path / "legacy"
    vendored.mkdir()
    legacy.mkdir()
    (vendored / "fake.task").write_bytes(b"corrupt")
    monkeypatch.setattr(mediapipe_assets, "vendored_dir", lambda: vendored)
    monkeypatch.setattr(mediapipe_assets, "legacy_dir", lambda: legacy)
    monkeypatch.setattr(
        mediapipe_assets, "load_manifest",
        lambda: {"fake.task": {"url": "https://example.invalid/fake.task", "sha256": "0" * 64}},
    )
    assert mediapipe_assets.resolve_model("fake.task", download=False) is None


def test_resolve_model_downloads_when_missing(monkeypatch, tmp_path):
    """Missing everywhere + download on -> downloads to the legacy dir."""
    vendored = tmp_path / "vendored"
    legacy = tmp_path / "legacy"
    vendored.mkdir()
    legacy.mkdir()
    payload = b"fresh-model-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(mediapipe_assets, "vendored_dir", lambda: vendored)
    monkeypatch.setattr(mediapipe_assets, "legacy_dir", lambda: legacy)
    monkeypatch.setattr(
        mediapipe_assets, "load_manifest",
        lambda: {"fake.task": {"url": "https://example.invalid/fake.task", "sha256": digest}},
    )

    def fake_download(url, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    monkeypatch.setattr(mediapipe_assets, "_download_file", fake_download)
    path = mediapipe_assets.resolve_model("fake.task")
    assert path == legacy / "fake.task"
    assert path.read_bytes() == payload
