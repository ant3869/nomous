"""Nightly memory consolidation ("dreaming") for Nomous.

Adapted from OpenClaw's dreaming (experimental, opt-in, in memory-core):
three phases run in order on a schedule — Light (ingest recent
short-term material + recall traces, dedupe, stage candidates — NEVER
writes long-term), REM (reflect: extract patterns/themes from recent
traces), Deep (score candidates and promote durable ones into long-term
memory — the ONLY phase that writes long-term).

Nomous adaptation notes:
- Long-term memory is the SQLite memory graph, not a markdown file.
  Promotion creates a consolidated concept node (milestone=1) linked to
  its source nodes; it does not rewrite existing nodes.
- Short-term material is the day's interaction nodes (kind 'stimulus':
  chat, vision, autonomous ticks, and observations jotted down by the
  idle loop's act phase via record_observation). Responses ('concept'
  nodes) are not candidates on their own — each stimulus carries its
  response via the interaction edge, and the record_observation
  boilerplate reply would otherwise cluster into junk. Exception:
  autonomous thoughts store their content IN the response node
  (stimulus 'internal_reflection' / 'scene_digest'), so Light resolves
  those via the edge instead of promoting the placeholder. Vision is
  the same shape in reverse: the stimulus is the coarse frame
  description, so Light uses the linked VLM observation (the response)
  instead, or different activities in the same lighting collapse into
  one generic candidate.
- A second run never re-promotes already-consolidated sources: the
  overlapping lookback would otherwise mint a duplicate milestone
  node every night.
- Unpromoted memories decay (strength *= 0.9, floor 0.1); they are
  NEVER deleted.
- The Dream Diary (DREAMS.md) logs every consolidation decision —
  promoted / rejected + why — as the audit trail. Decisions are also
  posted to the dashboard as `dream_diary` lines (Dreams stream).
- Silent by definition: the pass posts thoughts/events only, never
  speaks, and runs on the autonomous lane (never over a conversation).
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Deep-phase signal weights, in the OpenClaw spirit.
W_RELEVANCE = 0.30
W_FREQUENCY = 0.24
W_DIVERSITY = 0.15
W_RECENCY = 0.15
W_CONSOLIDATION = 0.10
W_RICHNESS = 0.06

# Near-duplicate threshold for clustering (word-overlap / longest text),
# same shape as the vision similarity check in llm.py.
SIMILARITY_THRESHOLD = 0.5

_STOPWORDS = frozenset(
    "a an the and or of to in on at for with is was are were be been it its "
    "this that these those i you he she we they my your his her our their "
    "me him us them as by from".split()
)


def _words(text: str) -> set:
    # \w is Unicode-aware: CJK text is one word per unspaced run, so
    # distinct CJK observations share no words and never falsely merge,
    # while identical ones still overlap fully.
    return {
        w
        for w in re.findall(r"[\w']+", (text or "").lower())
        if w not in _STOPWORDS and len(w) > 1
    }


def _similar(a: str, b: str) -> bool:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return False
    overlap = len(wa & wb)
    return (overlap / max(len(wa), len(wb))) > SIMILARITY_THRESHOLD


# Stimulus labels that carry no content of their own: autonomous thoughts
# and scene digests record the observation as the RESPONSE node
# (record_interaction("autonomous", "internal_reflection", <thought>)).
# The candidate text comes from the linked response instead.
_PLACEHOLDER_STIMULI = frozenset({"internal_reflection", "scene_digest", "internal reflection"})


def _effective_text(node: Dict[str, Any]) -> str:
    """Candidate text for a stimulus node.

    Real stimuli speak for themselves — except vision: the stimulus is
    the coarse frame description and the substantive observation lives
    in the linked response. Placeholder stimuli resolve to their linked
    interaction response; boilerplate responses ("Thought",
    "Recorded observation...") resolve to "" so the node is skipped.
    """
    text = (node.get("description") or node.get("label") or "").strip()
    response = (node.get("response_description") or "").strip()
    if not text or text.lower() in _PLACEHOLDER_STIMULI:
        if not response or response.lower() == "thought":
            return ""
        return response
    if (node.get("source") or "") == "vision" and response:
        # The stimulus is the coarse _describe_frame() output; the VLM's
        # actual observation is the response. Different activities in the
        # same lighting collapse into one generic candidate otherwise.
        if response.lower() == "thought":
            return text
        return response
    return text


def _normalize_text(text: str) -> str:
    # \w is Unicode-aware, so CJK text survives normalization instead of
    # collapsing to an empty grouping key. A text with no word characters
    # at all keeps its raw form, so unrelated punctuation-only texts
    # don't group together under "".
    normalized = re.sub(r"\s+", " ", re.sub(r"[^\w ]+", "", text.lower())).strip()
    return normalized or (text or "").lower().strip()


def seconds_until_next_run(time_str: str, now: Optional[datetime] = None) -> float:
    """Seconds until the next daily run at HH:MM (local time)."""
    now = now or datetime.now()
    try:
        hour, minute = (int(p) for p in str(time_str).strip().split(":", 1))
    except (ValueError, AttributeError):
        hour, minute = 3, 0
    target = now.replace(hour=hour % 24, minute=minute % 60, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return (target - now).total_seconds()


class Dreamer:
    """Nightly consolidation pass: light -> REM -> deep, then diary."""

    def __init__(self, llm, memory, bridge, cfg: Optional[Dict[str, Any]] = None):
        dreaming_cfg = (cfg or {}).get("dreaming", cfg or {})
        self.llm = llm
        self.memory = memory
        self.bridge = bridge
        self.enabled = bool(dreaming_cfg.get("enabled", True))
        self.time = str(dreaming_cfg.get("time", "03:00"))
        self.lookback_hours = max(1, int(dreaming_cfg.get("lookback_hours", 72) or 72))
        self.min_score = float(dreaming_cfg.get("min_score", 0.55))
        self.min_occurrences = max(1, int(dreaming_cfg.get("min_occurrences", 2) or 2))
        self.min_unique_days = max(1, int(dreaming_cfg.get("min_unique_days", 2) or 2))
        self.decay_factor = float(dreaming_cfg.get("decay_factor", 0.9))
        diary_path = str(dreaming_cfg.get("diary_path", "./data/dreams/DREAMS.md"))
        path = Path(diary_path)
        if not path.is_absolute():
            root = Path(__file__).resolve().parent.parent.parent
            path = root / path
        self.diary_path = path

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    async def run_once(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Run one full dreaming pass. Returns a summary dict.

        Silent by definition: posts thoughts/events/diary lines only.
        Never speaks. Skips cleanly when memory is unavailable or the
        autonomous lane is busy.
        """
        now = now or datetime.now(timezone.utc)
        summary: Dict[str, Any] = {
            "reviewed": 0,
            "candidates": 0,
            "promoted": 0,
            "rejected": 0,
            "decayed": 0,
            "already_consolidated": 0,
            "themes": [],
            "decisions": [],
            "skipped": None,
        }
        if not self.enabled:
            summary["skipped"] = "dreaming disabled"
            return summary
        if self.memory is None or not getattr(self.memory, "enabled", False):
            summary["skipped"] = "memory unavailable"
            logger.info("Dreaming skipped: memory unavailable")
            return summary

        acquired = False
        acquire = getattr(self.llm, "_acquire_lane", None) if self.llm else None
        if acquire is not None:
            try:
                acquired = await acquire("autonomous")
            except Exception:
                acquired = False
            if not acquired:
                summary["skipped"] = "autonomous lane busy"
                logger.info("Dreaming skipped: autonomous lane busy")
                return summary

        try:
            # Chat lane busy — dreaming yields to conversations, as the
            # ordinary autonomous loop does; its REM generation would
            # otherwise hold the shared model lock behind a live reply.
            lane_busy = getattr(self.llm, "_lane_busy", None) if self.llm else None
            if lane_busy is not None and lane_busy("chat"):
                summary["skipped"] = "chat lane busy"
                logger.info("Dreaming skipped: chat lane busy")
                return summary
            await self.bridge.post(
                {"type": "thought", "text": "🌙 Dreaming: reviewing the day's notes…"}
            )
            clusters = await self._light(now)
            summary["reviewed"] = sum(len(c["nodes"]) for c in clusters)
            summary["candidates"] = len(clusters)

            themes = await self._rem(clusters)
            summary["themes"] = themes

            decisions = await self._deep(clusters, themes, now)
            summary["decisions"] = decisions
            summary["promoted"] = sum(1 for d in decisions if d["outcome"] == "promoted")
            summary["rejected"] = sum(1 for d in decisions if d["outcome"] == "rejected")
            summary["already_consolidated"] = sum(
                1 for d in decisions if d["outcome"] == "skipped"
            )
            summary["decayed"] = sum(d.get("decayed", 0) for d in decisions)

            await self._write_diary(now, summary)
            await self._post_summary(summary)
            return summary
        finally:
            if acquired:
                release = getattr(self.llm, "_release_lane", None)
                if release is not None:
                    try:
                        release("autonomous")
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Light: ingest + dedupe, never writes long-term
    # ------------------------------------------------------------------

    async def _light(self, now: datetime) -> List[Dict[str, Any]]:
        """Stage candidate clusters from recent stimulus nodes."""
        since = (
            (now - timedelta(hours=self.lookback_hours))
            .replace(tzinfo=None)
            .isoformat(timespec="seconds")
        )  # stored timestamps are naive UTC ISO; keep the comparison naive too
        nodes = await self.memory.recent_nodes(since, kinds=("stimulus",))
        # Resolve each node to its effective (candidate) text, deduping by id.
        effective: List[Dict[str, Any]] = []
        seen_ids = set()
        for node in nodes:
            node_id = node.get("id")
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            text = _effective_text(node)
            if not text:
                continue
            node["effective_text"] = text
            effective.append(node)
        # Cheap exact grouping first (most repeats are near-identical), so
        # the pairwise similarity merge only runs over distinct texts.
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for node in effective:
            groups.setdefault(_normalize_text(node["effective_text"]), []).append(node)
        clusters: List[Dict[str, Any]] = []
        for group_nodes in groups.values():
            representative = group_nodes[0]["effective_text"]
            placed = False
            for cluster in clusters:
                if _similar(representative, cluster["representative"]):
                    cluster["nodes"].extend(group_nodes)
                    placed = True
                    break
            if not placed:
                clusters.append({"representative": representative, "nodes": list(group_nodes)})
        logger.info("Dreaming light: %d nodes -> %d candidates", len(nodes), len(clusters))
        return clusters

    # ------------------------------------------------------------------
    # REM: reflect — extract patterns/themes from recent traces
    # ------------------------------------------------------------------

    async def _rem(self, clusters: Sequence[Dict[str, Any]]) -> List[str]:
        """Ask the model for recurring themes. Best-effort: [] when the
        model is unavailable — deep scoring falls back to importance."""
        if not clusters or self.llm is None or getattr(self.llm, "model", None) is None:
            return []
        lines = []
        for cluster in list(clusters)[:15]:
            lines.append(f"- {cluster['representative'][:120]} (x{len(cluster['nodes'])})")
        prompt = (
            "These are things observed or said over the last few days. "
            "List the recurring patterns or themes (one short phrase per line, "
            "max 5 lines, no preamble):\n\n" + "\n".join(lines)
        )
        try:
            response = await self.llm._generate(prompt, max_tokens=200, allow_tool_calls=False)
        except Exception as exc:
            logger.warning("Dreaming REM failed, continuing without themes: %s", exc)
            return []
        themes = []
        for raw in (response or "").splitlines():
            theme = raw.strip().lstrip("-•* ").strip()
            if theme and len(theme) > 2:
                themes.append(theme[:100])
        return themes[:5]

    # ------------------------------------------------------------------
    # Deep: score + promote (the ONLY phase that writes long-term)
    # ------------------------------------------------------------------

    def _score_candidate(
        self, cluster: Dict[str, Any], themes: Sequence[str], now: datetime
    ) -> Dict[str, Any]:
        """Weighted signals + per-signal values for one candidate."""
        nodes = cluster["nodes"]
        occurrences = len(nodes)
        modalities = {str(n.get("source") or "unknown") for n in nodes}
        days = set()
        latest = None
        importances = []
        lengths = []
        for node in nodes:
            ts = str(node.get("timestamp") or "")
            days.add(ts[:10])
            try:
                parsed = datetime.fromisoformat(ts)
            except ValueError:
                parsed = None
            if parsed is not None:
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)  # stored naive = UTC
                if latest is None or parsed > latest:
                    latest = parsed
            importances.append(float(node.get("importance") or 0.0))
            lengths.append(
                len(str(node.get("effective_text") or node.get("description") or ""))
            )

        relevance = max(importances) if importances else 0.0
        if themes:
            theme_words = _words(" ".join(themes))
            overlap = len(_words(cluster["representative"]) & theme_words)
            if theme_words and overlap:
                relevance = min(1.0, relevance + 0.2)
        frequency = min(1.0, occurrences / 5.0)
        diversity = min(1.0, len(modalities) / 3.0)
        if latest is not None:
            age_hours = max(0.0, (now - latest).total_seconds() / 3600.0)
            recency = max(0.0, 1.0 - age_hours / max(1, self.lookback_hours))
        else:
            recency = 0.0
        consolidation = min(1.0, len(days) / 3.0)
        richness = min(1.0, (sum(lengths) / max(1, len(lengths))) / 200.0)

        score = (
            W_RELEVANCE * relevance
            + W_FREQUENCY * frequency
            + W_DIVERSITY * diversity
            + W_RECENCY * recency
            + W_CONSOLIDATION * consolidation
            + W_RICHNESS * richness
        )
        return {
            "score": round(score, 3),
            "signals": {
                "relevance": round(relevance, 3),
                "frequency": round(frequency, 3),
                "diversity": round(diversity, 3),
                "recency": round(recency, 3),
                "consolidation": round(consolidation, 3),
                "richness": round(richness, 3),
            },
            "occurrences": occurrences,
            "modalities": sorted(modalities),
            "unique_days": len(days),
        }

    def _gate_reason(
        self, result: Dict[str, Any]
    ) -> Optional[str]:
        """First failing gate, or None when ALL gates pass."""
        if result["score"] < self.min_score:
            return f"score {result['score']} < {self.min_score}"
        if result["occurrences"] < self.min_occurrences:
            return f"seen {result['occurrences']}x (min {self.min_occurrences})"
        if result["unique_days"] < self.min_unique_days:
            return f"only {result['unique_days']} day(s) (min {self.min_unique_days})"
        return None

    async def _deep(
        self, clusters: Sequence[Dict[str, Any]], themes: Sequence[str], now: datetime
    ) -> List[Dict[str, Any]]:
        """Score every candidate; promote the durable ones, decay the rest.

        Graph writes are batched: all promotion edges and all decay
        strengths are applied in two bulk calls at the end, so the pass
        publishes the graph once per batch instead of once per node/edge.
        """
        decisions = []
        # Already-consolidated sources (from past runs) are skipped, not
        # re-promoted: the overlapping lookback would otherwise mint a
        # duplicate milestone node every night.
        all_sources = [
            n.get("id") for c in clusters for n in c["nodes"] if n.get("id")
        ]
        consolidated = await self.memory.consolidated_source_ids(all_sources)
        edge_batch: List[Dict[str, Any]] = []
        strength_updates: Dict[str, float] = {}
        for cluster in clusters:
            result = self._score_candidate(cluster, themes, now)
            label = cluster["representative"][:80]
            if any(n.get("id") in consolidated for n in cluster["nodes"]):
                decisions.append(
                    {
                        "outcome": "skipped",
                        "label": label,
                        "score": result["score"],
                        "occurrences": result["occurrences"],
                        "unique_days": result["unique_days"],
                        "modalities": result["modalities"],
                        "reason": "already consolidated in a past dream run",
                        "node_id": None,
                        "decayed": 0,
                    }
                )
                continue
            reason = self._gate_reason(result)
            if reason is None:
                node_id, edges = await self._promote(cluster, result, now)
                edge_batch.extend(edges)
                decisions.append(
                    {
                        "outcome": "promoted",
                        "label": label,
                        "score": result["score"],
                        "occurrences": result["occurrences"],
                        "unique_days": result["unique_days"],
                        "modalities": result["modalities"],
                        "reason": f"score {result['score']} passes all gates",
                        "node_id": node_id,
                        "decayed": 0,
                    }
                )
            else:
                strengths = self._decay_strengths(cluster)
                strength_updates.update(strengths)
                decisions.append(
                    {
                        "outcome": "rejected",
                        "label": label,
                        "score": result["score"],
                        "occurrences": result["occurrences"],
                        "unique_days": result["unique_days"],
                        "modalities": result["modalities"],
                        "reason": reason,
                        "node_id": None,
                        "decayed": len(strengths),
                    }
                )
        if edge_batch:
            await self.memory.create_edges(edge_batch)
        if strength_updates:
            await self.memory.update_node_strengths(strength_updates)
        return decisions

    async def _promote(
        self, cluster: Dict[str, Any], result: Dict[str, Any], now: datetime
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Write one consolidated long-term node, linked to its sources.

        Returns the node id plus the 'consolidates' edges to create — the
        caller applies all edges in one bulk call (one graph publish),
        rather than one publish per source."""
        sources = [n.get("id") for n in cluster["nodes"] if n.get("id")]
        node_id = await self.memory.create_node(
            {
                "label": cluster["representative"][:80],
                "kind": "concept",
                "category": "dream_consolidated",
                "description": cluster["representative"][:500],
                "source": "dreaming",
                "milestone": 1,
                "strength": min(5.0, 1.0 + result["score"]),
                "importance": result["score"],
                "tags": ["dream", "consolidated"],
                "timestamp": now.isoformat(timespec="seconds"),
                # Pass the dict itself: the store only serializes dict/list
                # metadata (a JSON string would be stored as NULL).
                "metadata": {
                    "dream_score": result["score"],
                    "signals": result["signals"],
                    "occurrences": result["occurrences"],
                    "unique_days": result["unique_days"],
                    "modalities": result["modalities"],
                    "sources": sources,
                },
            }
        )
        if node_id:
            edges = [
                {
                    "from_id": node_id,
                    "to_id": source_id,
                    "weight": 1.0,
                    "relationship": "consolidates",
                    "context": {"dream": now.isoformat(timespec="seconds")},
                }
                for source_id in sources
            ]
        else:
            edges = []
        logger.info("Dreaming promoted '%s' (score %s)", cluster["representative"][:60], result["score"])
        return node_id, edges

    def _decay_strengths(self, cluster: Dict[str, Any]) -> Dict[str, float]:
        """Compute decayed strengths for reviewed-but-rejected sources.

        Pure computation — the caller applies the whole batch with one
        graph publish. Never deletes."""
        strengths: Dict[str, float] = {}
        for node in cluster["nodes"]:
            node_id = node.get("id")
            if not node_id:
                continue
            try:
                strength = float(node.get("strength") or 1.0)
            except (TypeError, ValueError):
                strength = 1.0
            strengths[node_id] = max(0.1, round(strength * self.decay_factor, 3))
        return strengths

    # ------------------------------------------------------------------
    # Diary + dashboard
    # ------------------------------------------------------------------

    def _diary_text(self, now: datetime, summary: Dict[str, Any]) -> str:
        lines = [
            f"## Dream {now.strftime('%Y-%m-%d %H:%M')} — reviewed {summary['reviewed']}, "
            f"promoted {summary['promoted']}, rejected {summary['rejected']}, "
            f"decayed {summary['decayed']}",
            "",
        ]
        if summary["themes"]:
            lines.append("### Themes")
            lines.extend(f"- {t}" for t in summary["themes"])
            lines.append("")
        promoted = [d for d in summary["decisions"] if d["outcome"] == "promoted"]
        rejected = [d for d in summary["decisions"] if d["outcome"] == "rejected"]
        skipped = [d for d in summary["decisions"] if d["outcome"] == "skipped"]
        if promoted:
            lines.append("### Promoted")
            for d in promoted:
                lines.append(
                    f"- {d['label']} (score {d['score']}, {d['occurrences']}x, "
                    f"{d['unique_days']} day(s), {','.join(d['modalities'])}) — {d['reason']}"
                )
            lines.append("")
        if rejected:
            lines.append("### Rejected")
            for d in rejected:
                lines.append(
                    f"- {d['label']} (score {d['score']}, {d['occurrences']}x, "
                    f"{d['unique_days']} day(s), {','.join(d['modalities'])}) — {d['reason']}"
                )
            lines.append("")
        if skipped:
            lines.append("### Already consolidated")
            for d in skipped:
                lines.append(
                    f"- {d['label']} (score {d['score']}, {d['occurrences']}x, "
                    f"{d['unique_days']} day(s), {','.join(d['modalities'])}) — {d['reason']}"
                )
            lines.append("")
        return "\n".join(lines)

    def _append_diary_sync(self, text: str) -> None:
        self.diary_path.parent.mkdir(parents=True, exist_ok=True)
        header = "# Dream Diary\n\nEvery nightly consolidation decision — what was promoted, what was rejected and why.\n\n"
        if not self.diary_path.exists():
            self.diary_path.write_text(header, encoding="utf-8")
        with self.diary_path.open("a", encoding="utf-8") as fh:
            fh.write(text)
            fh.write("\n")

    async def _write_diary(self, now: datetime, summary: Dict[str, Any]) -> None:
        text = self._diary_text(now, summary)
        try:
            await asyncio.to_thread(self._append_diary_sync, text)
        except Exception as exc:
            logger.warning("Dreaming diary write failed: %s", exc)

    async def _post_summary(self, summary: Dict[str, Any]) -> None:
        for decision in summary["decisions"]:
            if decision["outcome"] == "promoted":
                line = f"🌙 promoted: {decision['label']} — {decision['reason']}"
            elif decision["outcome"] == "skipped":
                line = f"🌙 already consolidated: {decision['label']} — {decision['reason']}"
            else:
                line = f"💤 rejected: {decision['label']} — {decision['reason']}"
            try:
                await self.bridge.post({"type": "dream_diary", "text": line})
            except Exception:
                pass
        text = (
            f"🌙 Dreaming done: reviewed {summary['reviewed']}, "
            f"promoted {summary['promoted']}, rejected {summary['rejected']}, "
            f"decayed {summary['decayed']}."
        )
        try:
            await self.bridge.post({"type": "thought", "text": text})
        except Exception:
            pass
