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
from collections.abc import Iterable, Sequence
from typing import Any, Dict, List, Optional, Tuple, Set
from uuid import uuid4

from .behavior import BehaviorLearner, BehaviorDirective
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
        self._behavior = BehaviorLearner()

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

        self._migrate_schema()

    def _migrate_schema(self) -> None:
        assert self._conn is not None
        with closing(self._conn.cursor()) as cur:
            node_columns = self._table_columns(cur, "memory_nodes")
            self._ensure_column(cur, "memory_nodes", node_columns, "meaning", "TEXT")
            self._ensure_column(cur, "memory_nodes", node_columns, "category", "TEXT")
            self._ensure_column(cur, "memory_nodes", node_columns, "importance", "REAL DEFAULT 0.0")
            self._ensure_column(cur, "memory_nodes", node_columns, "valence", "TEXT")
            self._ensure_column(cur, "memory_nodes", node_columns, "created_at", "TEXT")
            self._ensure_column(cur, "memory_nodes", node_columns, "updated_at", "TEXT")
            self._ensure_column(cur, "memory_nodes", node_columns, "last_accessed", "TEXT")
            self._ensure_column(cur, "memory_nodes", node_columns, "metadata", "TEXT")

            edge_columns = self._table_columns(cur, "memory_edges")
            self._ensure_column(cur, "memory_edges", edge_columns, "relationship", "TEXT")

            self._conn.commit()

    @staticmethod
    def _table_columns(cur: sqlite3.Cursor, table: str) -> Set[str]:
        # Only allow known table names to prevent SQL injection
        allowed_tables = {"memory_nodes", "memory_edges"}
        if table not in allowed_tables:
            raise ValueError(f"Invalid table name: {table}")
        cur.execute(f"PRAGMA table_info({table})")
        return {str(row[1]) for row in cur.fetchall()}

    def _ensure_column(
        self,
        cur: sqlite3.Cursor,
        table: str,
        existing: Set[str],
        column: str,
        definition: str,
    ) -> None:
        if column in existing:
            return
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        existing.add(column)

    @staticmethod
    def _normalise_tags(tags: Optional[Iterable[str]]) -> List[str]:
        seen: Set[str] = set()
        ordered: List[str] = []
        if not tags:
            return ordered
        for tag in tags:
            if not tag:
                continue
            norm = str(tag).strip()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            ordered.append(norm)
        return ordered

    @staticmethod
    def _derive_importance(tags: Sequence[str], default: float = 0.4) -> float:
        for tag in tags:
            if tag.startswith("importance_"):
                try:
                    score = float(tag.split("_", 1)[1])
                    return max(0.0, min(1.0, score / 10.0))
                except (ValueError, TypeError):
                    continue
        return default

    def _ensure_edge(
        self,
        cur: sqlite3.Cursor,
        from_id: str,
        to_id: str,
        weight: float,
        relationship: Optional[str],
        context: Optional[Dict[str, Any]],
        now: str,
    ) -> str:
        context_json = json.dumps(context, ensure_ascii=False) if context else None
        rel_key = relationship or (context or {}).get("relation") if context else None

        cur.execute(
            """
            SELECT id, weight FROM memory_edges
            WHERE from_id = ? AND to_id = ? AND IFNULL(relationship, '') = IFNULL(?, '')
            """,
            (from_id, to_id, rel_key or ""),
        )
        row = cur.fetchone()

        target_weight = max(0.1, min(5.0, float(weight)))

        if row:
            current = float(row["weight"] or 0.0)
            new_weight = max(0.1, min(5.0, current * 0.6 + target_weight * 0.8))
            cur.execute(
                """
                UPDATE memory_edges
                SET weight = ?, context = ?, last_strength_change = ?, relationship = IFNULL(?, relationship)
                WHERE id = ?
                """,
                (new_weight, context_json, now, rel_key, row["id"]),
            )
            return str(row["id"])

        edge_id = f"edge:{uuid4().hex[:12]}"
        cur.execute(
            """
            INSERT INTO memory_edges(id, from_id, to_id, weight, context, last_strength_change, relationship)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (edge_id, from_id, to_id, target_weight, context_json, now, rel_key),
        )
        return edge_id

    def _apply_behaviors(
        self,
        cur: sqlite3.Cursor,
        behaviors: Sequence[BehaviorDirective],
        stim_id: str,
        resp_id: str,
        now: str,
    ) -> List[Dict[str, Any]]:
        if not behaviors:
            return []

        results: List[Dict[str, Any]] = []
        for directive in behaviors:
            metadata = {
                "cues": directive.cues,
                "expectation": directive.expectation,
                "source": "user_instruction",
                "persona": directive.persona,
                "summary": directive.summary,
            }
            tags = self._normalise_tags([*directive.tags, "behavior", "social_learning"])
            importance = max(0.0, min(1.0, directive.importance))
            label = directive.label

            cur.execute("SELECT id, strength FROM memory_nodes WHERE id = ?", (directive.key,))
            row = cur.fetchone()
            if row:
                strength = float(row["strength"] or 1.0)
                new_strength = max(0.1, min(5.0, strength + importance))
                cur.execute(
                    """
                    UPDATE memory_nodes
                    SET label = ?, description = ?, meaning = ?, tags = ?, importance = ?, strength = ?,
                        confidence = ?, category = 'behavior', valence = COALESCE(valence, 'neutral'),
                        updated_at = ?, last_accessed = ?, metadata = ?, milestone = 1
                    WHERE id = ?
                    """,
                    (
                        label,
                        directive.instruction,
                        directive.summary or directive.expectation,
                        json.dumps(tags, ensure_ascii=False) if tags else None,
                        importance,
                        new_strength,
                        directive.confidence,
                        now,
                        now,
                        json.dumps(metadata, ensure_ascii=False),
                        directive.key,
                    ),
                )
                status = "reinforced"
            else:
                cur.execute(
                    """
                    INSERT INTO memory_nodes(
                        id, label, kind, strength, description, tags, source, timestamp,
                        confidence, meaning, category, importance, valence,
                        created_at, updated_at, last_accessed, metadata, milestone
                    )
                    VALUES (?, ?, 'behavior', ?, ?, ?, 'behavior_learner', ?, ?, ?, 'behavior', ?, 'neutral', ?, ?, ?, ?, 1)
                    """,
                    (
                        directive.key,
                        label,
                        max(0.6, min(5.0, 1.2 + directive.importance * 2)),
                        directive.instruction,
                        json.dumps(tags, ensure_ascii=False) if tags else None,
                        now,
                        directive.confidence,
                        directive.summary or directive.expectation,
                        importance,
                        now,
                        now,
                        now,
                        json.dumps(metadata, ensure_ascii=False),
                    ),
                )
                status = "learned"

            self._ensure_edge(
                cur,
                "self",
                directive.key,
                1.5 + importance,
                "identity_behavior",
                {"relation": "identity_behavior", "importance": importance},
                now,
            )
            self._ensure_edge(
                cur,
                stim_id,
                directive.key,
                1.0 + importance,
                "behavior_trigger",
                {"relation": "trigger", "instruction": directive.instruction[:200]},
                now,
            )
            self._ensure_edge(
                cur,
                directive.key,
                resp_id,
                0.8 + directive.confidence,
                "behavior_response",
                {"relation": "response_alignment", "expectation": directive.expectation[:200]},
                now,
            )

            results.append({"id": directive.key, "label": label, "status": status})

        return results

    def _update_node_sync(self, node_id: str, changes: Dict[str, Any]) -> bool:
        assert self._conn is not None
        if not changes:
            return False

        allowed_keys = {
            "label",
            "description",
            "meaning",
            "tags",
            "milestone",
            "strength",
            "importance",
            "category",
            "valence",
            "confidence",
            "metadata",
        }

        updates: List[str] = []
        values: List[Any] = []

        for key, value in changes.items():
            if key not in allowed_keys:
                continue
            if key == "tags":
                if isinstance(value, str):
                    raw_tags = [part.strip() for part in value.split(",")]
                elif isinstance(value, Iterable):
                    raw_tags = list(value)
                else:
                    raw_tags = []
                normalised = self._normalise_tags(raw_tags)
                updates.append("tags = ?")
                values.append(json.dumps(normalised, ensure_ascii=False) if normalised else None)
            elif key == "milestone":
                updates.append("milestone = ?")
                values.append(1 if bool(value) else 0)
            elif key == "strength":
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    continue
                updates.append("strength = ?")
                values.append(max(0.1, min(5.0, val)))
            elif key == "importance":
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    continue
                updates.append("importance = ?")
                values.append(max(0.0, min(1.0, val)))
            elif key == "confidence":
                try:
                    val = float(value)
                except (TypeError, ValueError):
                    continue
                updates.append("confidence = ?")
                values.append(max(0.0, min(1.0, val)))
            elif key == "metadata":
                meta_json = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else None
                updates.append("metadata = ?")
                values.append(meta_json)
            else:
                updates.append(f"{key} = ?")
                values.append(value)

        if not updates:
            return False

        now = datetime.utcnow().isoformat(timespec="seconds")
        updates.extend(["updated_at = ?", "last_accessed = ?"])
        values.extend([now, now, node_id])

        with closing(self._conn.cursor()) as cur:
            cur.execute(
                f"UPDATE memory_nodes SET {', '.join(updates)} WHERE id = ?",
                values,
            )
            self._conn.commit()
            return cur.rowcount > 0

    def _delete_node_sync(self, node_id: str) -> bool:
        assert self._conn is not None
        with closing(self._conn.cursor()) as cur:
            cur.execute("DELETE FROM memory_nodes WHERE id = ?", (node_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def _create_edge_sync(
        self,
        from_id: str,
        to_id: str,
        weight: float,
        relationship: Optional[str],
        context: Optional[Dict[str, Any]],
    ) -> bool:
        assert self._conn is not None
        now = datetime.utcnow().isoformat(timespec="seconds")
        with closing(self._conn.cursor()) as cur:
            edge_id = self._ensure_edge(cur, from_id, to_id, weight, relationship, context, now)
            self._conn.commit()
            return bool(edge_id)

    def _delete_edge_sync(self, edge_id: str) -> bool:
        assert self._conn is not None
        with closing(self._conn.cursor()) as cur:
            cur.execute("DELETE FROM memory_edges WHERE id = ?", (edge_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def _ensure_identity_node(self) -> None:
        assert self._conn is not None
        now = datetime.utcnow().isoformat(timespec="seconds")
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                INSERT INTO memory_nodes(
                    id, label, kind, strength, milestone, description, timestamp,
                    meaning, category, importance, valence, created_at, updated_at, last_accessed, metadata
                )
                VALUES (?, ?, 'self', 1.0, 1, ?, ?, ?, ?, 1.0, 'positive', ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label=excluded.label,
                    description=excluded.description,
                    timestamp=excluded.timestamp,
                    meaning=excluded.meaning,
                    category=excluded.category,
                    importance=excluded.importance,
                    valence=excluded.valence,
                    updated_at=excluded.updated_at,
                    last_accessed=excluded.last_accessed,
                    metadata=excluded.metadata
                """,
                (
                    "self",
                    "Nomous",
                    "Core system identity",
                    now,
                    "Core identity anchor for Nomous",
                    "identity",
                    now,
                    now,
                    now,
                    json.dumps({"cues": ["self"], "expectation": "Preserve continuity"})
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

        behaviors: List[BehaviorDirective] = []
        if stimulus:
            behaviors = self._behavior.extract(stimulus, response)

        result: Dict[str, Any] | None = None
        try:
            async with self._lock:
                result = await asyncio.to_thread(self._record_interaction_sync, payload, behaviors)
        except Exception:
            logger.exception("Failed to record memory interaction for modality '%s'", modality)
            await self.bridge.post(msg_event(f"memory error: failed to store {modality}"))
            return

        await self.publish_graph()

        if result and result.get("behaviors"):
            for item in result["behaviors"]:
                status = item.get("status", "learned")
                label = item.get("label", item.get("id", "behavior"))
                await self.bridge.post(
                    msg_event(f"behavior {status}: {label}")
                )

    async def update_node(self, node_id: str, changes: Dict[str, Any]) -> bool:
        if not self.enabled or self._conn is None or not node_id:
            return False
        async with self._lock:
            updated = await asyncio.to_thread(self._update_node_sync, node_id, changes)
        if updated:
            await self.publish_graph()
        return updated

    async def delete_node(self, node_id: str) -> bool:
        if not self.enabled or self._conn is None or not node_id:
            return False
        async with self._lock:
            removed = await asyncio.to_thread(self._delete_node_sync, node_id)
        if removed:
            await self.publish_graph()
        return removed

    async def create_edge(
        self,
        from_id: str,
        to_id: str,
        *,
        weight: float = 1.0,
        relationship: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.enabled or self._conn is None or not from_id or not to_id:
            return False
        async with self._lock:
            created = await asyncio.to_thread(
                self._create_edge_sync, from_id, to_id, weight, relationship, context
            )
        if created:
            await self.publish_graph()
        return created

    async def delete_edge(self, edge_id: str) -> bool:
        if not self.enabled or self._conn is None or not edge_id:
            return False
        async with self._lock:
            removed = await asyncio.to_thread(self._delete_edge_sync, edge_id)
        if removed:
            await self.publish_graph()
        return removed

    # ------------------------------------------------------------------
    # Synchronous helpers (used via asyncio.to_thread)
    # ------------------------------------------------------------------
    def _record_interaction_sync(
        self,
        payload: Dict[str, Any],
        behaviors: Sequence[BehaviorDirective],
    ) -> Dict[str, Any]:
        assert self._conn is not None
        now = datetime.utcnow().isoformat(timespec="seconds")
        modality = payload["modality"]
        stimulus = payload["stimulus"].strip()
        response = (payload.get("response") or "").strip()
        tags = payload.get("tags") or []
        confidence = payload.get("confidence")

        stim_id = f"{modality}:stim:{uuid4().hex[:8]}"
        resp_id = f"reply:{uuid4().hex[:8]}"

        stim_label = stimulus[:80] if stimulus else f"{modality.title()} stimulus"
        resp_label = response[:80] if response else "Thought"

        stim_description = stimulus if len(stimulus) <= 500 else stimulus[:497] + "..."
        resp_description = response if len(response) <= 500 else response[:497] + "..."

        normalised_tags = self._normalise_tags(tags)
        tags_json = json.dumps(normalised_tags, ensure_ascii=False) if normalised_tags else None
        importance = self._derive_importance(normalised_tags)

        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                INSERT INTO memory_nodes(
                    id, label, kind, strength, description, tags, source, timestamp, confidence,
                    meaning, category, importance, valence, created_at, updated_at, last_accessed, metadata
                )
                VALUES (?, ?, 'stimulus', 1.0, ?, ?, ?, ?, ?, ?, ?, ?, 'neutral', ?, ?, ?, ?)
                """,
                (
                    stim_id,
                    stim_label,
                    stim_description or stim_label,
                    tags_json,
                    modality,
                    now,
                    confidence,
                    stim_description or stim_label,
                    f"stimulus:{modality}",
                    importance,
                    now,
                    now,
                    now,
                    json.dumps(
                        {
                            "stimulus": stimulus,
                            "modality": modality,
                            "tags": normalised_tags,
                        },
                        ensure_ascii=False,
                    ),
                ),
            )

            cur.execute(
                """
                INSERT INTO memory_nodes(
                    id, label, kind, strength, description, source, timestamp, confidence,
                    meaning, category, importance, valence, created_at, updated_at, last_accessed, metadata
                )
                VALUES (?, ?, 'concept', 1.0, ?, 'assistant', ?, ?, ?, 'assistant_response', 0.5, 'neutral', ?, ?, ?, ?)
                """,
                (
                    resp_id,
                    resp_label,
                    resp_description or resp_label,
                    now,
                    confidence,
                    resp_description or resp_label,
                    now,
                    now,
                    now,
                    json.dumps(
                        {
                            "response": response,
                            "origin": "assistant",
                        },
                        ensure_ascii=False,
                    ),
                ),
            )

            interaction_context = {"modality": modality, "relation": "interaction"}
            self._ensure_edge(cur, stim_id, resp_id, 1.0, "interaction", interaction_context, now)

            behavior_results = self._apply_behaviors(cur, behaviors, stim_id, resp_id, now)

            self._conn.commit()

        logger.debug(
            "Persisted memory nodes (%s -> %s) with edge %s",
            stim_id,
            resp_id,
            "interaction",
        )

        return {"behaviors": behavior_results}

    def _load_graph_sync(self) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        assert self._conn is not None
        with closing(self._conn.cursor()) as cur:
            cur.execute(
                """
                SELECT id, label, kind, strength, description, tags, milestone, source, timestamp, confidence,
                       meaning, category, importance, valence, created_at, updated_at, last_accessed, metadata
                FROM memory_nodes
                ORDER BY datetime(COALESCE(timestamp, '1970-01-01T00:00:00')) DESC
                """
            )
            nodes = []
            for row in cur.fetchall():
                tags = json.loads(row["tags"]) if row["tags"] else []
                metadata = json.loads(row["metadata"]) if row["metadata"] else None
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
                        "meaning": row["meaning"],
                        "category": row["category"],
                        "importance": row["importance"],
                        "valence": row["valence"],
                        "createdAt": row["created_at"],
                        "updatedAt": row["updated_at"],
                        "lastAccessed": row["last_accessed"],
                        "metadata": metadata,
                    }
                )

            cur.execute(
                """
                SELECT id, from_id, to_id, weight, context, last_strength_change, relationship
                FROM memory_edges
                ORDER BY datetime(COALESCE(last_strength_change, '1970-01-01T00:00:00')) DESC
                """
            )
            edges = []
            for row in cur.fetchall():
                context = row["context"]
                context_data = None
                if context:
                    try:
                        context_data = json.loads(context)
                    except json.JSONDecodeError:
                        context_data = None
                edges.append(
                    {
                        "id": row["id"],
                        "from": row["from_id"],
                        "to": row["to_id"],
                        "weight": float(row["weight"] or 0.0),
                        "context": context,
                        "contextData": context_data,
                        "relationship": row["relationship"],
                        "lastStrengthChange": row["last_strength_change"],
                    }
                )

        return nodes, edges


__all__ = ["MemoryStore"]
