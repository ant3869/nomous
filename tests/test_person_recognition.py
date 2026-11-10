#!/usr/bin/env python3
"""Tests for real person recognition via face embeddings (finding 4).

Covers: cosine similarity, embedding-dominated matching (a face that
moves across the frame still matches its person), fallback to the
size/position heuristic below the threshold, and embedding persistence.
"""

import math
import sys
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.backend.person_tracker import (
    PersonTracker,
    TrackedPerson,
    VisualSignature,
    _cosine_similarity,
)


def _unit_vector(seed: int, dim: int = 8) -> list:
    """Deterministic pseudo-random unit vector."""
    import random

    rng = random.Random(seed)
    vec = [rng.uniform(-1, 1) for _ in range(dim)]
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec]


def _near(vec: list, noise: float = 0.01) -> list:
    """A slightly perturbed copy of a unit vector (still ~1.0 similar)."""
    import random

    rng = random.Random(999)
    out = [v + rng.uniform(-noise, noise) for v in vec]
    norm = math.sqrt(sum(v * v for v in out))
    return [v / norm for v in out]


def test_cosine_similarity():
    assert _cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert _cosine_similarity([1, 0], [1, 0, 0]) is None
    assert _cosine_similarity([0, 0], [1, 0]) is None
    assert _cosine_similarity([], [1, 0]) is None


def test_embedding_similarity_needs_both_sides():
    a = VisualSignature(face_embedding=[1.0, 0.0])
    b = VisualSignature(face_embedding=[1.0, 0.0])
    c = VisualSignature()
    assert a.embedding_similarity(b) == pytest.approx(1.0)
    assert a.embedding_similarity(c) is None


def test_similarity_score_embedding_dominates():
    # Same position + size, but very different faces -> must NOT match.
    emb_a = _unit_vector(1)
    emb_b = _unit_vector(2)
    assert _cosine_similarity(emb_a, emb_b) < 0.5
    sig_a = VisualSignature(typical_position="left", face_size_avg=1000, face_embedding=emb_a)
    sig_b = VisualSignature(typical_position="left", face_size_avg=1000, face_embedding=emb_b)
    assert sig_a.similarity_score(sig_b) < 0.5

    # Different position + size, but (near-)identical faces -> must match.
    sig_c = VisualSignature(typical_position="right", face_size_avg=5000, face_embedding=_near(emb_a))
    assert sig_a.similarity_score(sig_c) > 0.7


@pytest.mark.asyncio
async def test_process_frame_matches_by_embedding_across_positions():
    tracker = PersonTracker(embedding_match_threshold=0.7)
    emb_a = _unit_vector(1)
    emb_b = _unit_vector(2)

    # Frame 1: person A on the left.
    first = await tracker.process_frame([(10, 10, 50, 50)], 640, 360, [emb_a])
    assert first["person_count"] == 1
    assert len(first["new_arrivals"]) == 1
    person_a = first["present"][0]["id"]

    # Frame 2: the same face on the RIGHT. Without embeddings this would
    # be a new person; the embedding must win over position.
    second = await tracker.process_frame([(500, 10, 50, 50)], 640, 360, [_near(emb_a)])
    assert second["new_arrivals"] == []
    assert second["present"][0]["id"] == person_a

    # Frame 3: a different face on the left -> a new person.
    third = await tracker.process_frame([(10, 10, 50, 50)], 640, 360, [emb_b])
    assert len(third["new_arrivals"]) == 1
    assert third["present"][0]["id"] != person_a


@pytest.mark.asyncio
async def test_process_frame_falls_back_without_embeddings():
    tracker = PersonTracker()
    # No embeddings at all: the old size/position heuristic still works.
    first = await tracker.process_frame([(10, 10, 50, 50)], 640, 360)
    second = await tracker.process_frame([(12, 10, 50, 50)], 640, 360)
    assert second["new_arrivals"] == []
    assert second["present"][0]["id"] == first["present"][0]["id"]


@pytest.mark.asyncio
async def test_embedding_veto_and_ambiguous_band():
    """Clearly-different faces veto a heuristic match; ambiguous ones don't.

    Vectors are crafted, not random, so the cosines are exact: a=[1,0,...]
    vs b=[0.6,0.8,...] is 0.6 (ambiguous band), vs c=[0,1,...] is 0.0 (veto).
    """
    tracker = PersonTracker(embedding_match_threshold=0.7)
    dim = 8
    emb_a = [1.0] + [0.0] * (dim - 1)
    emb_ambiguous = [0.6, 0.8] + [0.0] * (dim - 2)
    emb_other = [0.0, 1.0] + [0.0] * (dim - 2)
    assert _cosine_similarity(emb_a, emb_ambiguous) == pytest.approx(0.6)
    assert _cosine_similarity(emb_a, emb_other) == pytest.approx(0.0)

    await tracker.process_frame([(10, 10, 50, 50)], 640, 360, [emb_a])
    # Ambiguous embedding (0.6) in the same spot: heuristic decides -> match.
    second = await tracker.process_frame([(10, 10, 50, 50)], 640, 360, [emb_ambiguous])
    assert second["new_arrivals"] == []
    # Clearly different embedding (0.0) in the same spot: veto -> new person.
    third = await tracker.process_frame([(10, 10, 50, 50)], 640, 360, [emb_other])
    assert len(third["new_arrivals"]) == 1


@pytest.mark.asyncio
async def test_veto_applies_per_candidate_not_globally():
    """One ambiguous similarity must not shield a clearly-different face.

    A (left, [1,0,...]) and B (right, [0,1,...]) are tracked. An incoming
    left-hand face with [0,0.6,0.8,...] has cosine 0.0 to A (vetoed) and
    0.6 to B (ambiguous). The old global veto let the best similarity
    (0.6) disable the veto for everyone, and the heuristic then matched
    A on position + size. Now A is excluded and the face is a new person.
    """
    tracker = PersonTracker(embedding_match_threshold=0.7)
    dim = 8
    emb_a = [1.0] + [0.0] * (dim - 1)
    emb_b = [0.0, 1.0] + [0.0] * (dim - 2)
    emb_c = [0.0, 0.6, 0.8] + [0.0] * (dim - 3)
    assert _cosine_similarity(emb_c, emb_a) == pytest.approx(0.0)
    assert _cosine_similarity(emb_c, emb_b) == pytest.approx(0.6)

    first = await tracker.process_frame(
        [(10, 10, 50, 50), (500, 10, 50, 50)], 640, 360, [emb_a, emb_b]
    )
    assert len(first["new_arrivals"]) == 2
    known_ids = {p["id"] for p in first["present"]}

    second = await tracker.process_frame([(10, 10, 100, 100)], 640, 360, [emb_c])
    assert len(second["new_arrivals"]) == 1
    assert second["present"][0]["id"] not in known_ids


def test_embedding_persists_through_memory_dict():
    emb = _unit_vector(3)
    person = TrackedPerson(person_id="person_1")
    person.visual_signature.face_embedding = emb
    data = person.to_memory_dict()
    assert data["face_embedding"] is not None
    assert len(data["face_embedding"]) == len(emb)

    restored = TrackedPerson.from_memory_dict(data)
    assert restored.visual_signature.face_embedding is not None
    assert _cosine_similarity(
        restored.visual_signature.face_embedding, emb
    ) == pytest.approx(1.0, abs=1e-3)


def test_embed_face_crop_end_to_end():
    """Real ImageEmbedder on the vendored model: a crop embeds to 1024 dims.

    Skipped when the MediaPipe native library cannot load (environmental).
    """
    import numpy as np

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python import BaseOptions as MpBaseOptions
        from src.backend.video import CameraLoop
    except Exception as exc:  # pragma: no cover - environmental
        pytest.skip(f"MediaPipe unavailable: {exc}")

    model_path = project_root / "assets" / "mediapipe" / "mobilenet_v3_small.tflite"
    try:
        embedder = mp_vision.ImageEmbedder.create_from_options(
            mp_vision.ImageEmbedderOptions(
                base_options=MpBaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                l2_normalize=True,
                quantize=False,
            )
        )
    except Exception as exc:  # pragma: no cover - environmental
        pytest.skip(f"ImageEmbedder could not start: {exc}")

    loop = CameraLoop.__new__(CameraLoop)
    loop._face_embedder = embedder
    try:
        rng = np.random.default_rng(7)
        frame_rgb = (rng.random((360, 640, 3)) * 255).astype(np.uint8)
        vec = loop._embed_face_crop(frame_rgb, (100, 100, 80, 80))
        assert vec is not None
        assert len(vec) == 1024
        # Deterministic: the same crop embeds (near-)identically.
        vec2 = loop._embed_face_crop(frame_rgb, (100, 100, 80, 80))
        assert _cosine_similarity(vec, vec2) == pytest.approx(1.0, abs=1e-4)
        # A crop outside the frame is rejected, not crashed on.
        assert loop._embed_face_crop(frame_rgb, (10000, 10000, 10, 10)) is None
    finally:
        embedder.close()
