# Title: MediaPipe task-model assets - vendored files + checksum verification
# Path: backend/mediapipe_assets.py
# Purpose: Resolve the MediaPipe task models the camera loop needs, preferring
# the vendored copies in assets/mediapipe/ (checksum-checked against the
# manifest) and falling back to a pinned-URL download into the legacy
# modules/models/mediapipe/ dir. No heavy imports: start.py uses this for
# its preflight check without pulling in cv2/mediapipe.

import hashlib
import json
import logging
import urllib.request
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Pinned download URLs for every task model (also recorded in the manifest).
MEDIAPIPE_MODELS: Dict[str, str] = {
    "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    "mobilenet_v3_small.tflite": "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def vendored_dir() -> Path:
    """Vendored models, tracked in git."""
    return project_root() / "assets" / "mediapipe"


def legacy_dir() -> Path:
    """Pre-vendoring download location (gitignored, kept as fallback)."""
    return project_root() / "modules" / "models" / "mediapipe"


def load_manifest() -> Dict[str, Dict[str, str]]:
    """Read assets/mediapipe/manifest.json ({filename: {url, sha256}})."""
    manifest_path = vendored_dir() / "manifest.json"
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("models", {})
    except (OSError, ValueError) as exc:
        logger.warning("Could not read MediaPipe manifest %s: %s", manifest_path, exc)
        return {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: Optional[str]) -> bool:
    """True when the file exists and matches the expected checksum.

    With no expected checksum the check is existence only.
    """
    if not path.exists():
        return False
    if not expected_sha256:
        return True
    try:
        return sha256_file(path) == expected_sha256.lower()
    except OSError as exc:
        logger.warning("Could not checksum %s: %s", path, exc)
        return False


def expected_sha256(filename: str) -> Optional[str]:
    entry = load_manifest().get(filename, {})
    return entry.get("sha256")


def _download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response, open(target, "wb") as out:
        out.write(response.read())


def resolve_model(filename: str, *, download: bool = True) -> Optional[Path]:
    """Return a verified local path for a MediaPipe task model.

    Order: vendored copy (checksum-checked against the manifest) first,
    then the legacy download dir, then a fresh download from the pinned
    URL (also checksum-checked). Returns None when no verified copy can
    be produced — callers must degrade gracefully (detector stays off).
    """
    manifest = load_manifest()
    expected = manifest.get(filename, {}).get("sha256")
    url = manifest.get(filename, {}).get("url") or MEDIAPIPE_MODELS.get(filename)
    if url is None:
        logger.warning("Unknown MediaPipe model: %s", filename)
        return None

    vendored = vendored_dir() / filename
    if verify_file(vendored, expected):
        return vendored
    if vendored.exists():
        logger.warning("Vendored model %s failed its checksum; ignoring it", filename)

    legacy = legacy_dir() / filename
    if verify_file(legacy, expected):
        return legacy
    if legacy.exists():
        logger.warning("Downloaded model %s failed its checksum; re-downloading", filename)

    if not download:
        return None
    try:
        logger.info("Downloading MediaPipe model %s ...", filename)
        _download_file(url, legacy)
    except Exception as exc:
        logger.warning("Could not download MediaPipe model %s: %s", filename, exc)
        return None
    if verify_file(legacy, expected):
        return legacy
    logger.warning("Downloaded model %s failed its checksum; ignoring it", filename)
    try:
        legacy.unlink()
    except OSError:
        pass
    return None


def verify_all_vendored() -> Dict[str, bool]:
    """Checksum every manifest entry against its vendored copy.

    Used by the start.py preflight and by tests. Missing/corrupt entries
    report False; a missing manifest reports every known model False.
    """
    manifest = load_manifest()
    names = manifest or MEDIAPIPE_MODELS
    return {
        filename: verify_file(vendored_dir() / filename, manifest.get(filename, {}).get("sha256"))
        for filename in names
    }
