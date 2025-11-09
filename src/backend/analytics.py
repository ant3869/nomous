"""Conversation analytics and scoring for Nomous responses."""
from __future__ import annotations

import math
import re
import statistics
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional

__all__ = ["ConversationAnalytics", "ResponseSample"]

_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "when",
    "is",
    "are",
    "was",
    "were",
    "be",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "at",
    "by",
    "about",
    "into",
    "over",
    "than",
    "that",
    "this",
    "it",
    "as",
    "from",
    "your",
    "you",
    "me",
    "my",
    "we",
    "our",
}

_POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "positive",
    "awesome",
    "helpful",
    "glad",
    "love",
    "like",
    "joy",
    "clear",
    "insight",
    "thanks",
    "appreciate",
}

_NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "negative",
    "hate",
    "angry",
    "confused",
    "sorry",
    "concern",
    "unclear",
    "worry",
    "issue",
    "problem",
    "error",
}

_TOXIC_TOKENS = {
    "idiot",
    "stupid",
    "dumb",
    "hate",
    "kill",
    "die",
    "violent",
    "racist",
    "sexist",
    "dumbass",
    "moron",
}


@dataclass
class ResponseSample:
    """Snapshot describing a single user/model interaction."""

    user_text: str
    model_text: str
    timestamp: float
    latency_ms: float
    tokens_in: int
    tokens_out: int
    metrics: Dict[str, float] = field(default_factory=dict)


def _tokenise(text: str) -> List[str]:
    return re.findall(r"[\w']+", text.lower(), flags=re.UNICODE)


def _keywords(text: str) -> Counter:
    words = [w for w in _tokenise(text) if len(w) > 2 and w not in _STOPWORDS]
    return Counter(words)


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    numerator = sum(a[w] * b[w] for w in shared)
    if numerator == 0:
        return 0.0
    sum_a = sum(v * v for v in a.values())
    sum_b = sum(v * v for v in b.values())
    denominator = math.sqrt(sum_a) * math.sqrt(sum_b)
    return numerator / denominator if denominator else 0.0


def _count_tokens(text: str) -> int:
    return max(1, len(_tokenise(text)))


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _sentiment_score(text: str) -> float:
    words = _tokenise(text)
    if not words:
        return 0.5
    pos = sum(1 for w in words if w in _POSITIVE_WORDS)
    neg = sum(1 for w in words if w in _NEGATIVE_WORDS)
    raw = 0.5 + (pos - neg) / max(1, len(words))
    return _clamp(raw)


def _toxicity_score(text: str) -> float:
    words = set(_tokenise(text))
    toxic_hits = sum(1 for w in words if w in _TOXIC_TOKENS)
    if not words:
        return 0.0
    return _clamp(toxic_hits / len(words))


def _lexical_richness(text: str) -> float:
    words = _tokenise(text)
    if not words:
        return 0.0
    unique = len(set(words))
    return _clamp(unique / len(words))


def _coherence_score(text: str) -> float:
    words = _tokenise(text)
    if not words:
        return 0.0
    def is_noise(w: str) -> bool:
        vowels = sum(ch in "aeiou" for ch in w)
        return vowels == 0 or len(w) > 12
    noise_ratio = sum(1 for w in words if is_noise(w)) / len(words)
    punctuation_penalty = min(0.4, text.count("??") * 0.05 + text.count("!!") * 0.05)
    return _clamp(1.0 - noise_ratio - punctuation_penalty)


def _brevity_score(tokens_out: int) -> float:
    # Encourage between 12 and 80 tokens
    if tokens_out <= 0:
        return 0.0
    target = 40
    spread = 35
    diff = abs(tokens_out - target)
    score = math.exp(-(diff ** 2) / (2 * (spread ** 2)))
    return _clamp(score)


def _responsiveness(user_vec: Counter, model_vec: Counter) -> float:
    if not model_vec:
        return 0.0
    overlap = sum(model_vec[w] for w in model_vec if w in user_vec)
    coverage = overlap / sum(model_vec.values())
    return _clamp(coverage)


def _stability(samples: Iterable[ResponseSample]) -> float:
    metrics = {
        "on_topic": [],
        "coherence": [],
        "responsiveness": [],
    }
    for sample in samples:
        metrics["on_topic"].append(sample.metrics.get("on_topic", 0.0))
        metrics["coherence"].append(sample.metrics.get("coherence", 0.0))
        metrics["responsiveness"].append(sample.metrics.get("responsiveness", 0.0))
    spreads = []
    for values in metrics.values():
        if len(values) >= 2:
            spreads.append(statistics.pstdev(values))
    if not spreads:
        return 1.0
    avg_spread = sum(spreads) / len(spreads)
    return _clamp(1.0 - avg_spread)


class ConversationAnalytics:
    """Track rolling behaviour metrics for model interactions."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.samples: Deque[ResponseSample] = deque(maxlen=window_size)
        self.reward_total = 0.0
        self._pending_user_text: Optional[str] = None
        self._pending_user_timestamp: Optional[float] = None
        self._global_topic = Counter()

    def reset(self) -> None:
        self.samples.clear()
        self.reward_total = 0.0
        self._pending_user_text = None
        self._pending_user_timestamp = None
        self._global_topic.clear()

    def observe_user_message(self, text: str) -> None:
        now = time.time()
        self._pending_user_text = text or ""
        self._pending_user_timestamp = now
        self._global_topic.update(_keywords(text))

    def observe_model_response(
        self,
        text: str,
        *,
        tokens_in: Optional[int] = None,
        tokens_out: Optional[int] = None,
    ) -> Dict[str, object]:
        if self._pending_user_text is None:
            # Treat as system initiated
            self._pending_user_text = ""
            self._pending_user_timestamp = time.time()

        now = time.time()
        latency_ms = 0.0
        if self._pending_user_timestamp is not None:
            latency_ms = max(0.0, (now - self._pending_user_timestamp) * 1000.0)

        tokens_in = tokens_in if tokens_in is not None else _count_tokens(self._pending_user_text or "")
        tokens_out = tokens_out if tokens_out is not None else _count_tokens(text)

        user_vector = _keywords(self._pending_user_text or "")
        response_vector = _keywords(text)

        on_topic = _clamp(_cosine(response_vector, self._global_topic or user_vector))
        responsiveness = _responsiveness(user_vector, response_vector)
        brevity = _brevity_score(tokens_out)
        coherence = _coherence_score(text)
        sentiment = _sentiment_score(text)
        toxicity = _toxicity_score(text)
        lexical = _lexical_richness(text)

        sample = ResponseSample(
            user_text=self._pending_user_text or "",
            model_text=text,
            timestamp=now,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            metrics={
                "on_topic": on_topic,
                "responsiveness": responsiveness,
                "brevity": brevity,
                "coherence": coherence,
                "sentiment": sentiment,
                "toxicity": toxicity,
                "lexical": lexical,
            },
        )
        self.samples.append(sample)

        self._pending_user_text = None
        self._pending_user_timestamp = None

        return self._build_payload(sample)

    def apply_reward(self, delta: float) -> Dict[str, object]:
        self.reward_total += float(delta)
        if self.samples:
            latest = self.samples[-1]
        else:
            latest = ResponseSample(
                user_text="",
                model_text="",
                timestamp=time.time(),
                latency_ms=0.0,
                tokens_in=0,
                tokens_out=0,
                metrics={"on_topic": 0.0, "responsiveness": 0.0, "brevity": 0.0, "coherence": 0.0, "sentiment": 0.5, "toxicity": 0.0, "lexical": 0.0},
            )
        return self._build_payload(latest)

    # ------------------------------------------------------------------
    # Payload helpers
    # ------------------------------------------------------------------

    def _build_payload(self, latest: ResponseSample) -> Dict[str, object]:
        history = list(self.samples)
        summary = self._aggregate_summary(history)
        signals = self._derive_signals(history, latest)
        anomalies = self._detect_anomalies(latest)
        trend = [
            {
                "timestamp": sample.timestamp,
                "onTopic": sample.metrics.get("on_topic", 0.0),
                "sentiment": sample.metrics.get("sentiment", 0.5),
                "coherence": sample.metrics.get("coherence", 0.0),
                "safety": 1.0 - sample.metrics.get("toxicity", 0.0),
            }
            for sample in history
        ]

        payload: Dict[str, object] = {
            "summary": summary,
            "signals": signals,
            "history": trend,
            "rewardTotal": self.reward_total,
            "anomalies": anomalies,
            "lastSample": {
                "userText": latest.user_text,
                "assistantText": latest.model_text,
                "latencyMs": latest.latency_ms,
                "tokensIn": latest.tokens_in,
                "tokensOut": latest.tokens_out,
                "scores": {
                    "onTopic": latest.metrics.get("on_topic", 0.0),
                    "responsiveness": latest.metrics.get("responsiveness", 0.0),
                    "brevity": latest.metrics.get("brevity", 0.0),
                    "coherence": latest.metrics.get("coherence", 0.0),
                    "sentiment": latest.metrics.get("sentiment", 0.5),
                    "lexicalRichness": latest.metrics.get("lexical", 0.0),
                    "safety": 1.0 - latest.metrics.get("toxicity", 0.0),
                    "toxicity": latest.metrics.get("toxicity", 0.0),
                },
            },
        }
        return payload

    def _aggregate_summary(self, history: List[ResponseSample]) -> Dict[str, float]:
        if not history:
            return {
                "onTopic": 0.0,
                "responsiveness": 0.0,
                "brevity": 0.0,
                "coherence": 0.0,
                "sentiment": 0.5,
                "lexicalRichness": 0.0,
                "safety": 1.0,
                "stability": 1.0,
            }

        def mean(key: str) -> float:
            return sum(sample.metrics.get(key, 0.0) for sample in history) / len(history)

        summary = {
            "onTopic": mean("on_topic"),
            "responsiveness": mean("responsiveness"),
            "brevity": mean("brevity"),
            "coherence": mean("coherence"),
            "sentiment": mean("sentiment"),
            "lexicalRichness": mean("lexical"),
            "safety": 1.0 - mean("toxicity"),
            "stability": _stability(history),
        }
        for key, value in list(summary.items()):
            summary[key] = _clamp(value)
        return summary

    def _derive_signals(self, history: List[ResponseSample], latest: ResponseSample) -> Dict[str, float]:
        if history:
            span = history[-1].timestamp - history[0].timestamp
        else:
            span = 0.0
        pace = 0.0
        if span > 1:
            pace = (len(history) / span) * 60.0
        avg_length = sum(sample.tokens_out for sample in history) / len(history) if history else 0.0
        return {
            "latencyMs": latest.latency_ms,
            "tokensIn": latest.tokens_in,
            "tokensOut": latest.tokens_out,
            "conversationPace": pace,
            "avgResponseLength": avg_length,
        }

    def _detect_anomalies(self, latest: ResponseSample) -> List[Dict[str, str]]:
        anomalies: List[Dict[str, str]] = []
        metrics = latest.metrics
        user_text = latest.user_text.strip()
        if user_text and metrics.get("on_topic", 0.0) < 0.35:
            anomalies.append({
                "label": "Drift",
                "severity": "warn",
                "detail": "Response drifted away from conversation focus.",
            })
        if metrics.get("coherence", 0.0) < 0.3:
            anomalies.append({
                "label": "Coherence",
                "severity": "critical",
                "detail": "Low coherence detected in the latest reply.",
            })
        if metrics.get("toxicity", 0.0) > 0.15:
            anomalies.append({
                "label": "Safety",
                "severity": "critical",
                "detail": "Potentially unsafe language detected.",
            })
        if latest.latency_ms > 4500:
            anomalies.append({
                "label": "Latency",
                "severity": "warn",
                "detail": "Slow model response time.",
            })
        return anomalies

