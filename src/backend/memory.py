# Title: MemoryStore - Persistent conversational graph
# Path: backend/memory.py
# Purpose: Persist AI memories to SQLite and broadcast them to the UI with diagnostics.

"""Memory persistence and broadcasting utilities."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from .protocol import msg_event, msg_memory

logger = logging.getLogger(__name__)


class MemoryStore:
    """Persist conversational memories in SQLite and broadcast updates."""

    def __init__(self, cfg: Dict[str, Any], bridge):
        memory_cfg = (cfg or {}).get("memory", {})
        self.enabled = bool(memory_cfg.get("enable", True))
        self.bridge = bridge
        self._lock = asyncio.Lock()
        self._conn: Optional[sqlite3.Connection] = None

        if not self.enabled:
            logger.info("Memory store disabled via configuration")
            return

        db_path = memory_cfg.get("db_path")
        if not db_path:
            raise ValueError("memory.db_path must be configured when memory is enabled")

        path = Path(db_path)
        if not path.is_absolute():
            # Resolve relative to repository root (two levels up from backend/)
            root = Path(__file__).resolve().parent.parent.parent
            path = root / path

        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Initialising memory database at %s", path)

        try:
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
        except Exception:
            logger.exception("Failed to connect to memory database")
            raise

        self._conn = conn

        try:
            self._setup()
            self._ensure_identity_node()
        except Exception:
            logger.exception("Failed to initialise memory schema")
            raise

        logger.info("Memory store ready")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _setup(self) -> None:
        assert self._conn is not None
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    description TEXT,
                    tags TEXT,
                    milestone INTEGER DEFAULT 0,
                    source TEXT,
                    timestamp TEXT,
                    confidence REAL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_edges (
                    id TEXT PRIMARY KEY,
                    from_id TEXT NOT NULL,
                    to_id TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    context TEXT,
                    last_strength_change TEXT,
                    FOREIGN KEY(from_id) REFERENCES memory_nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY(to_id) REFERENCES memory_nodes(id) ON DELETE CASCADE
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_edges_from ON memory_edges(from_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_edges_to ON memory_edges(to_id)")
            self._conn.commit()

    def _ensure_identity_node(self) -> None:
        assert self._conn is not None
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                INSERT INTO memory_nodes(id, label, kind, strength, milestone, description, timestamp)
                VALUES (?, ?, 'self', 1.0, 1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label=excluded.label,
                    description=excluded.description,
                    timestamp=excluded.timestamp
                """,
                (
                    "self",
                    "Nomous",
                    "Core system identity",
                    datetime.utcnow().isoformat(timespec="seconds"),
                ),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def stop(self) -> None:
        if self._conn is None:
            return
        logger.info("Closing memory database")
        conn, self._conn = self._conn, None
        await asyncio.to_thread(conn.close)

    async def publish_graph(self) -> None:
        if not self.enabled or self._conn is None:
            return
        nodes, edges = await asyncio.to_thread(self._load_graph_sync)
        logger.debug("Broadcasting memory graph (%d nodes, %d edges)", len(nodes), len(edges))
        await self.bridge.post(msg_memory(nodes, edges))
    
    async def load_graph(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Public interface to load the memory graph."""
        if not self.enabled or self._conn is None:
            return ([], [])
        return await asyncio.to_thread(self._load_graph_sync)

    async def record_interaction(
        self,
        modality: str,
        stimulus: str,
        response: Optional[str],
        *,
        confidence: Optional[float] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> None:
        if not self.enabled or self._conn is None:
            logger.debug("Memory disabled - skipping record for modality '%s'", modality)
            return

        payload = {
            "modality": modality,
            "stimulus": stimulus,
            "response": response,
            "confidence": confidence,
            "tags": list(tags or []),
        }

        logger.info("Recording memory interaction: %s", json.dumps(payload, ensure_ascii=False)[:200])

        try:
            async with self._lock:
                await asyncio.to_thread(self._record_interaction_sync, payload)
        except Exception:
            logger.exception("Failed to record memory interaction for modality '%s'", modality)
            await self.bridge.post(msg_event(f"memory error: failed to store {modality}"))
            return

        await self.publish_graph()

    # ------------------------------------------------------------------
    # Synchronous helpers (used via asyncio.to_thread)
    # ------------------------------------------------------------------
    def _record_interaction_sync(self, payload: Dict[str, Any]) -> None:
        assert self._conn is not None
        now = datetime.utcnow().isoformat(timespec="seconds")
        modality = payload["modality"]
        stimulus = payload["stimulus"].strip()
        response = (payload.get("response") or "").strip()
        tags = payload.get("tags") or []
        confidence = payload.get("confidence")

        stim_id = f"{modality}:stim:{uuid4().hex[:8]}"
        resp_id = f"reply:{uuid4().hex[:8]}"
        edge_id = f"edge:{uuid4().hex[:12]}"

        stim_label = stimulus[:80] if stimulus else f"{modality.title()} stimulus"
        resp_label = response[:80] if response else "Thought"

        stim_description = stimulus if len(stimulus) <= 500 else stimulus[:497] + "..."
        resp_description = response if len(response) <= 500 else response[:497] + "..."

        tags_json = json.dumps(tags) if tags else None

        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                INSERT INTO memory_nodes(id, label, kind, strength, description, tags, source, timestamp, confidence)
                VALUES (?, ?, 'stimulus', 1.0, ?, ?, ?, ?, ?)
                """,
                (
                    stim_id,
                    stim_label,
                    stim_description or stim_label,
                    tags_json,
                    modality,
                    now,
                    confidence,
                ),
            )

            cur.execute(
                """
                INSERT INTO memory_nodes(id, label, kind, strength, description, source, timestamp, confidence)
                VALUES (?, ?, 'concept', 1.0, ?, 'assistant', ?, ?)
                """,
                (
                    resp_id,
                    resp_label,
                    resp_description or resp_label,
                    now,
                    confidence,
                ),
            )

            cur.execute(
                """
                INSERT INTO memory_edges(id, from_id, to_id, weight, context, last_strength_change)
                VALUES (?, ?, ?, 1.0, ?, ?)
                """,
                (
                    edge_id,
                    stim_id,
                    resp_id,
                    json.dumps({"modality": modality}, ensure_ascii=False),
                    now,
                ),
            )

            self._conn.commit()

        logger.debug(
            "Persisted memory nodes (%s -> %s) with edge %s",
            stim_id,
            resp_id,
            edge_id,
        )

    def _load_graph_sync(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        assert self._conn is not None
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                SELECT id, label, kind, strength, description, tags, milestone, source, timestamp, confidence
                FROM memory_nodes
                ORDER BY datetime(COALESCE(timestamp, '1970-01-01T00:00:00')) DESC
                """
            )
            nodes = []
            for row in cur.fetchall():
                tags = json.loads(row["tags"]) if row["tags"] else []
                nodes.append(
                    {
                        "id": row["id"],
                        "label": row["label"],
                        "kind": row["kind"],
                        "strength": float(row["strength"] or 0.0),
                        "description": row["description"],
                        "tags": tags,
                        "milestone": bool(row["milestone"]),
                        "source": row["source"],
                        "timestamp": row["timestamp"],
                        "confidence": row["confidence"],
                    }
                )

            cur.execute(
                """
                SELECT id, from_id, to_id, weight, context, last_strength_change
                FROM memory_edges
                ORDER BY datetime(COALESCE(last_strength_change, '1970-01-01T00:00:00')) DESC
                """
            )
            edges = []
            for row in cur.fetchall():
                edges.append(
                    {
                        "id": row["id"],
                        "from": row["from_id"],
                        "to": row["to_id"],
                        "weight": float(row["weight"] or 0.0),
                        "context": row["context"],
                        "lastStrengthChange": row["last_strength_change"],
                    }
                )

        return nodes, edges


__all__ = ["MemoryStore"]
