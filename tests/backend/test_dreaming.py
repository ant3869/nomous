import asyncio
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.backend.dreaming import Dreamer, seconds_until_next_run
from src.backend.memory import MemoryStore


class FakeBridge:
    def __init__(self):
        self.posts = []

    async def post(self, payload):
        self.posts.append(payload)


class FakeLLM:
    """Minimal LLM stand-in: themes + autonomous lane."""

    def __init__(self, themes_response=""):
        self.model = object()
        self._themes_response = themes_response
        self._lanes = {"chat": False, "vision": False, "autonomous": False}
        self._lane_lock = asyncio.Lock()
        self.generate_calls = []

    async def _acquire_lane(self, lane):
        async with self._lane_lock:
            if self._lanes.get(lane):
                return False
            self._lanes[lane] = True
            return True

    def _release_lane(self, lane):
        self._lanes[lane] = False

    async def _generate(self, prompt, max_tokens=200, allow_tool_calls=False):
        self.generate_calls.append(prompt)
        return self._themes_response


def _memory(tmp_path):
    bridge = FakeBridge()
    cfg = {
        "memory": {"enable": True, "db_path": str(tmp_path / "test.sqlite")},
        "paths": {},
    }
    return MemoryStore(cfg, bridge), bridge


def _dreamer(llm, store, bridge, tmp_path, **overrides):
    cfg = {
        "dreaming": {
            "enabled": True,
            "time": "03:00",
            "lookback_hours": 72,
            "min_score": 0.55,
            "min_occurrences": 2,
            "min_unique_days": 2,
            "decay_factor": 0.9,
            "diary_path": str(tmp_path / "DREAMS.md"),
            **overrides,
        }
    }
    return Dreamer(llm, store, bridge, cfg)


async def _seed(store, items):
    """items: (modality, stimulus_text, tags). Timestamps are 'now'."""
    for modality, text, tags in items:
        await store.record_interaction(modality, text, "noted.", tags=tags)


def _retimestamp(store, timestamps):
    """Assign timestamps (naive UTC ISO) to stimulus nodes, in seed order."""
    conn = store._conn
    rows = conn.execute(
        "SELECT id FROM memory_nodes WHERE kind = 'stimulus' ORDER BY rowid ASC"
    ).fetchall()
    assert len(rows) == len(timestamps), f"{len(rows)} stimulus nodes, {len(timestamps)} timestamps"
    for row, ts in zip(rows, timestamps):
        conn.execute("UPDATE memory_nodes SET timestamp = ? WHERE id = ?", (ts, row[0]))
    conn.commit()


def _node(store, node_id):
    row = store._conn.execute(
        "SELECT * FROM memory_nodes WHERE id = ?", (node_id,)
    ).fetchone()
    return dict(row) if row else None


def _all_nodes(store, kind=None):
    sql = "SELECT * FROM memory_nodes"
    args = []
    if kind:
        sql += " WHERE kind = ?"
        args.append(kind)
    return [dict(r) for r in store._conn.execute(sql, args).fetchall()]


def test_seconds_until_next_run():
    assert seconds_until_next_run("03:00", datetime(2026, 11, 10, 10, 0)) == 17 * 3600
    assert seconds_until_next_run("03:00", datetime(2026, 11, 10, 2, 0)) == 3600
    assert seconds_until_next_run("03:00", datetime(2026, 11, 10, 3, 0)) == 24 * 3600
    # Garbage time falls back to 03:00.
    assert seconds_until_next_run("nope", datetime(2026, 11, 10, 2, 0)) == 3600


@pytest.mark.asyncio
async def test_light_clusters_near_duplicates(tmp_path):
    store, bridge = _memory(tmp_path)
    await _seed(
        store,
        [
            ("autonomous", "Maya arrived home", ["importance_8"]),
            ("vision", "Maya arrived home again", ["importance_8"]),
            ("autonomous", "Maya arrived home late", ["importance_8"]),
            ("vision", "The dog slept on the rug all afternoon", ["importance_4"]),
        ],
    )
    dreamer = _dreamer(FakeLLM(), store, bridge, tmp_path)
    clusters = await dreamer._light(datetime.now(timezone.utc))
    assert len(clusters) == 2
    sizes = sorted(len(c["nodes"]) for c in clusters)
    assert sizes == [1, 3]


@pytest.mark.asyncio
async def test_recent_nodes_returns_window(tmp_path):
    store, bridge = _memory(tmp_path)
    await _seed(
        store,
        [
            ("chat", "Something recent happened", []),
            ("chat", "Something old happened", []),
        ],
    )
    now = datetime.now(timezone.utc)
    _retimestamp(
        store,
        [
            now.replace(tzinfo=None).isoformat(timespec="seconds"),
            (now - timedelta(days=5)).replace(tzinfo=None).isoformat(timespec="seconds"),
        ],
    )
    since = (now - timedelta(hours=72)).replace(tzinfo=None).isoformat(timespec="seconds")
    nodes = await store.recent_nodes(since, kinds=("stimulus",))
    assert len(nodes) == 1
    assert "recent" in nodes[0]["description"]


def _cluster(nodes, representative="Maya arrived home"):
    return {"representative": representative, "nodes": nodes}


def _node_dict(**overrides):
    base = {
        "id": "n1",
        "description": "Maya arrived home",
        "source": "autonomous",
        "timestamp": "2026-11-10T10:00:00",
        "importance": 0.8,
        "strength": 1.0,
    }
    base.update(overrides)
    return base


def test_score_gates_all_must_pass(tmp_path):
    store, bridge = _memory(tmp_path)
    dreamer = _dreamer(FakeLLM(), store, bridge, tmp_path)
    now = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)

    strong = _cluster(
        [
            _node_dict(id="a", timestamp="2026-11-09T10:00:00", source="autonomous"),
            _node_dict(id="b", timestamp="2026-11-10T10:00:00", source="vision"),
            _node_dict(id="c", timestamp="2026-11-10T11:00:00", source="autonomous"),
        ]
    )
    result = dreamer._score_candidate(strong, ["Maya arriving home"], now)
    assert result["score"] >= 0.55
    assert result["occurrences"] == 3
    assert result["unique_days"] == 2
    assert dreamer._gate_reason(result) is None

    # Single occurrence fails the frequency gate even with a good score.
    once = _cluster(
        [_node_dict(timestamp="2026-11-10T10:00:00", importance=1.0)]
    )
    result_once = dreamer._score_candidate(once, [], now)
    assert result_once["score"] >= 0.55  # score gate passes; frequency gate fires
    assert "seen 1x" in (dreamer._gate_reason(result_once) or "")

    # Single day fails the consolidation gate.
    one_day = _cluster(
        [
            _node_dict(id="a", timestamp="2026-11-10T09:00:00", source="autonomous"),
            _node_dict(id="b", timestamp="2026-11-10T10:00:00", source="vision"),
            _node_dict(id="c", timestamp="2026-11-10T11:00:00", source="chat"),
        ]
    )
    result_day = dreamer._score_candidate(one_day, [], now)
    assert "day(s)" in (dreamer._gate_reason(result_day) or "")

    # Low importance + old + thin fails the score gate.
    weak = _cluster(
        [
            _node_dict(
                id="a",
                description="A leaf fell",
                timestamp="2026-11-08T10:00:00",
                importance=0.1,
                source="vision",
            ),
            _node_dict(
                id="b",
                description="A leaf fell again",
                timestamp="2026-11-09T10:00:00",
                importance=0.1,
                source="vision",
            ),
        ]
    )
    result_weak = dreamer._score_candidate(weak, [], now)
    assert result_weak["score"] < 0.55
    assert "score" in (dreamer._gate_reason(result_weak) or "")


@pytest.mark.asyncio
async def test_run_once_promotes_durable_and_decays_rest(tmp_path):
    store, bridge = _memory(tmp_path)
    await _seed(
        store,
        [
            ("autonomous", "Maya arrived home", ["importance_8"]),
            ("vision", "Maya arrived home again", ["importance_8"]),
            ("autonomous", "Maya arrived home late", ["importance_8"]),
            ("vision", "The dog slept on the rug all afternoon", ["importance_4"]),
        ],
    )
    now = datetime.now(timezone.utc)
    _retimestamp(
        store,
        [
            (now - timedelta(days=1)).replace(tzinfo=None).isoformat(timespec="seconds"),
            (now - timedelta(hours=2)).replace(tzinfo=None).isoformat(timespec="seconds"),
            now.replace(tzinfo=None).isoformat(timespec="seconds"),
            now.replace(tzinfo=None).isoformat(timespec="seconds"),
        ],
    )
    llm = FakeLLM("- Maya arriving home\n- Dog sleeping\n")
    dreamer = _dreamer(llm, store, bridge, tmp_path)

    summary = await dreamer.run_once(now)

    assert summary["reviewed"] == 4
    assert summary["candidates"] == 2
    assert summary["promoted"] == 1
    assert summary["rejected"] == 1
    assert summary["decayed"] == 1
    assert summary["themes"] == ["Maya arriving home", "Dog sleeping"]
    assert not summary["skipped"]

    # Promotion wrote a consolidated milestone node linked to its sources.
    consolidated = [
        n for n in _all_nodes(store, "concept") if n.get("category") == "dream_consolidated"
    ]
    assert len(consolidated) == 1
    assert consolidated[0]["milestone"] == 1
    assert consolidated[0]["source"] == "dreaming"
    edges = store._conn.execute(
        "SELECT * FROM memory_edges WHERE from_id = ?", (consolidated[0]["id"],)
    ).fetchall()
    assert len(edges) == 3

    # The rejected one-off decayed (never deleted).
    dog = [n for n in _all_nodes(store, "stimulus") if "dog slept" in n["description"]]
    assert len(dog) == 1
    assert dog[0]["strength"] == pytest.approx(0.9)

    # Diary audit trail on disk + dashboard lines.
    diary = (tmp_path / "DREAMS.md").read_text(encoding="utf-8")
    assert diary.startswith("# Dream Diary")
    assert "### Promoted" in diary
    assert "### Rejected" in diary
    assert "Maya arrived home" in diary
    diary_posts = [p for p in bridge.posts if p.get("type") == "dream_diary"]
    assert any("promoted" in p.get("text", "") for p in diary_posts)
    assert any("rejected" in p.get("text", "") for p in diary_posts)
    assert any(
        p.get("type") == "thought" and "Dreaming done" in p.get("text", "")
        for p in bridge.posts
    )


@pytest.mark.asyncio
async def test_run_once_without_llm_still_promotes(tmp_path):
    store, bridge = _memory(tmp_path)
    await _seed(
        store,
        [
            ("autonomous", "Maya arrived home", ["importance_8"]),
            ("vision", "Maya arrived home again", ["importance_8"]),
        ],
    )
    now = datetime.now(timezone.utc)
    _retimestamp(
        store,
        [
            (now - timedelta(days=1)).replace(tzinfo=None).isoformat(timespec="seconds"),
            now.replace(tzinfo=None).isoformat(timespec="seconds"),
        ],
    )
    dreamer = _dreamer(None, store, bridge, tmp_path)
    summary = await dreamer.run_once(now)
    assert summary["themes"] == []
    assert summary["promoted"] == 1


@pytest.mark.asyncio
async def test_run_once_skips_without_memory(tmp_path):
    bridge = FakeBridge()
    dreamer = _dreamer(FakeLLM(), None, bridge, tmp_path)
    summary = await dreamer.run_once()
    assert summary["skipped"] == "memory unavailable"
    assert not (tmp_path / "DREAMS.md").exists()


@pytest.mark.asyncio
async def test_run_once_skips_when_lane_busy(tmp_path):
    store, bridge = _memory(tmp_path)
    llm = FakeLLM()
    llm._lanes["autonomous"] = True
    dreamer = _dreamer(llm, store, bridge, tmp_path)
    summary = await dreamer.run_once()
    assert summary["skipped"] == "autonomous lane busy"


@pytest.mark.asyncio
async def test_rem_handles_llm_failure(tmp_path):
    store, bridge = _memory(tmp_path)
    llm = FakeLLM()
    llm._generate = AsyncMock(side_effect=RuntimeError("boom"))
    dreamer = _dreamer(llm, store, bridge, tmp_path)
    clusters = [{"representative": "Maya arrived home", "nodes": [_node_dict()]}]
    assert await dreamer._rem(clusters) == []


@pytest.mark.asyncio
async def test_diary_appends_across_runs(tmp_path):
    store, bridge = _memory(tmp_path)
    await _seed(store, [("autonomous", "Maya arrived home", ["importance_8"])])
    dreamer = _dreamer(FakeLLM(), store, bridge, tmp_path)
    await dreamer.run_once()
    await dreamer.run_once()
    diary = (tmp_path / "DREAMS.md").read_text(encoding="utf-8")
    assert diary.count("## Dream ") == 2
    assert diary.count("# Dream Diary") == 1


@pytest.mark.asyncio
async def test_light_uses_response_text_for_placeholder_stimuli(tmp_path):
    # Codex P1 on PR #88: autonomous thoughts record the observation as
    # the RESPONSE ("internal_reflection" stimulus). Candidates must use
    # the real text, never the placeholder.
    store, bridge = _memory(tmp_path)
    for text in ("Maya arrived home", "Maya arrived home again", "Maya arrived home late"):
        await store.record_interaction("autonomous", "internal_reflection", text)
    now = datetime.now(timezone.utc)
    _retimestamp(
        store,
        [
            (now - timedelta(days=1)).replace(tzinfo=None).isoformat(timespec="seconds"),
            (now - timedelta(hours=2)).replace(tzinfo=None).isoformat(timespec="seconds"),
            now.replace(tzinfo=None).isoformat(timespec="seconds"),
        ],
    )
    llm = FakeLLM("- Maya arriving home\n")
    dreamer = _dreamer(llm, store, bridge, tmp_path)
    summary = await dreamer.run_once(now)

    assert summary["promoted"] == 1
    consolidated = [
        n for n in _all_nodes(store, "concept") if n.get("category") == "dream_consolidated"
    ]
    assert len(consolidated) == 1
    assert "Maya arrived home" in consolidated[0]["description"]
    assert "internal_reflection" not in consolidated[0]["description"]
    # P2 metadata fix: the dict survives (a JSON string would store NULL).
    meta = json.loads(consolidated[0]["metadata"])
    assert meta["dream_score"] >= 0.55
    assert meta["occurrences"] == 3
    assert len(meta["sources"]) == 3


@pytest.mark.asyncio
async def test_light_prefers_stimulus_over_boilerplate_response(tmp_path):
    store, bridge = _memory(tmp_path)
    await store.record_interaction(
        "observation", "[PEOPLE] Maya likes tea",
        "Recorded observation (importance: 8/10)", tags=["importance_8"],
    )
    await store.record_interaction(
        "observation", "[PEOPLE] Maya likes tea a lot",
        "Recorded observation (importance: 8/10)", tags=["importance_8"],
    )
    dreamer = _dreamer(FakeLLM(), store, bridge, tmp_path)
    clusters = await dreamer._light(datetime.now(timezone.utc))
    assert len(clusters) == 1
    assert "Maya likes tea" in clusters[0]["representative"]
    assert "Recorded observation" not in clusters[0]["representative"]


@pytest.mark.asyncio
async def test_recent_nodes_paginates_past_500(tmp_path):
    # Codex P2 on PR #88: the old LIMIT 500 silently dropped the newest
    # material in a busy window.
    store, bridge = _memory(tmp_path)
    conn = store._conn
    now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")
    for i in range(600):
        conn.execute(
            "INSERT INTO memory_nodes(id, label, kind, strength, description, source, timestamp)"
            " VALUES (?, ?, 'stimulus', 1.0, ?, 'test', ?)",
            (f"n{i}", f"node {i}", f"observation number {i}", now),
        )
    conn.commit()
    nodes = await store.recent_nodes("2020-01-01T00:00:00", kinds=("stimulus",))
    assert len(nodes) == 600


@pytest.mark.asyncio
async def test_second_run_skips_already_consolidated(tmp_path):
    # Codex P2 on PR #88: the overlapping 72h lookback must not mint a
    # duplicate milestone node every night.
    store, bridge = _memory(tmp_path)
    await _seed(
        store,
        [
            ("autonomous", "Maya arrived home", ["importance_8"]),
            ("vision", "Maya arrived home again", ["importance_8"]),
            ("autonomous", "Maya arrived home late", ["importance_8"]),
            ("vision", "The dog slept on the rug all afternoon", ["importance_4"]),
        ],
    )
    now = datetime.now(timezone.utc)
    _retimestamp(
        store,
        [
            (now - timedelta(days=1)).replace(tzinfo=None).isoformat(timespec="seconds"),
            (now - timedelta(hours=2)).replace(tzinfo=None).isoformat(timespec="seconds"),
            now.replace(tzinfo=None).isoformat(timespec="seconds"),
            now.replace(tzinfo=None).isoformat(timespec="seconds"),
        ],
    )
    dreamer = _dreamer(FakeLLM("- Maya arriving home\n- Dog sleeping\n"), store, bridge, tmp_path)

    summary1 = await dreamer.run_once(now)
    assert summary1["promoted"] == 1
    summary2 = await dreamer.run_once(now)
    assert summary2["promoted"] == 0
    assert summary2["already_consolidated"] == 1
    assert summary2["rejected"] == 1  # the dog, rejected again
    consolidated = [
        n for n in _all_nodes(store, "concept") if n.get("category") == "dream_consolidated"
    ]
    assert len(consolidated) == 1  # no duplicate milestone node
    diary = (tmp_path / "DREAMS.md").read_text(encoding="utf-8")
    assert "### Already consolidated" in diary
