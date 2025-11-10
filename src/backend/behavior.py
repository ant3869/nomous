"""Behavioral learning heuristics for conversational memory."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class BehaviorDirective:
    """Structured representation of a social/behavioral rule."""

    key: str
    label: str
    instruction: str
    cues: List[str]
    expectation: str
    importance: float
    confidence: float
    tags: List[str]
    persona: str | None = None
    summary: str | None = None


class BehaviorLearner:
    """Detect user behavioral preferences from natural language."""

    _RE_NORMALISE = re.compile(r"\s+")

    def __init__(self) -> None:
        self._busy_keywords = ("doing something", "busy", "working", "focused", "in the middle")
        self._wait_keywords = ("wait", "hold on", "let me finish", "finish")
        self._silence_keywords = ("talk so much", "talk less", "keep quiet", "stay quiet", "be quiet")
        self._night_keywords = ("night", "evening", "late", "dark")
        self._human_convo_keywords = ("talking to another human", "talking to someone", "conversation", "another person")

    @staticmethod
    def _normalise(text: str) -> str:
        return BehaviorLearner._RE_NORMALISE.sub(" ", text.strip().lower())

    @staticmethod
    def _hash_key(*parts: Sequence[str]) -> str:
        joined = "|".join(part.strip().lower() for part in parts if part)
        digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:20]
        return f"behavior:{digest}"

    def extract(self, stimulus: str, response: str | None = None) -> List[BehaviorDirective]:
        """Return a list of directives inferred from the latest interaction."""

        text = self._normalise(stimulus)
        if not text:
            return []

        directives: List[BehaviorDirective] = []
        sentences = re.split(r"[.!?]+", text)

        for raw_sentence in sentences:
            sentence = self._normalise(raw_sentence)
            if len(sentence) < 6:
                continue

            if not sentence.startswith("please") and "please" not in sentence and "kindly" not in sentence and "don't" not in sentence and "do not" not in sentence:
                # If the sentence is not clearly instructive, skip unless it carries strong cues.
                strong_cue = any(keyword in sentence for keyword in self._night_keywords + self._human_convo_keywords)
                if not strong_cue:
                    continue

            handlers = (
                self._handle_busy_focus,
                self._handle_wait_to_speak,
                self._handle_turn_taking,
                self._handle_night_quiet,
                self._handle_generic_preference,
            )

            for handler in handlers:
                directive = handler(sentence, stimulus)
                if directive is None:
                    continue
                if directive.key in {existing.key for existing in directives}:
                    continue
                directives.append(directive)
                break

        return directives

    def _handle_busy_focus(self, sentence: str, original: str) -> BehaviorDirective | None:
        if not any(keyword in sentence for keyword in self._silence_keywords):
            return None
        if not any(keyword in sentence for keyword in self._busy_keywords):
            return None

        label = "Respect focused work"
        expectation = "Minimize conversation when the user appears busy or concentrating."
        cues = ["user_state:busy", "visual:task_focus"]
        importance = 0.9
        tags = ["behavior", "social", "quiet_mode", "respect_focus"]

        return BehaviorDirective(
            key=self._hash_key(label, expectation),
            label=label,
            instruction=original.strip(),
            cues=cues,
            expectation=expectation,
            importance=importance,
            confidence=0.85,
            tags=tags,
            persona="considerate",
            summary="Back off conversationally when the user is busy."
        )

    def _handle_wait_to_speak(self, sentence: str, original: str) -> BehaviorDirective | None:
        if not any(keyword in sentence for keyword in self._wait_keywords):
            return None
        if "before" not in sentence and "until" not in sentence:
            return None

        label = "Wait for completion"
        expectation = "Pause responses until the user finishes their action or speech."
        cues = ["conversation:pause", "user_state:occupied"]
        importance = 0.85
        tags = ["behavior", "turn_taking", "patience"]

        return BehaviorDirective(
            key=self._hash_key(label, expectation),
            label=label,
            instruction=original.strip(),
            cues=cues,
            expectation=expectation,
            importance=importance,
            confidence=0.82,
            tags=tags,
            persona="patient",
            summary="Hold replies until the user signals they are ready."
        )

    def _handle_turn_taking(self, sentence: str, original: str) -> BehaviorDirective | None:
        if not any(keyword in sentence for keyword in self._human_convo_keywords):
            return None

        label = "Respect human conversations"
        expectation = "Do not interrupt when the user is talking with someone else; wait to be invited."
        cues = ["audio:multiple_speakers", "user_state:conversing"]
        importance = 0.88
        tags = ["behavior", "social", "turn_taking", "politeness"]

        return BehaviorDirective(
            key=self._hash_key(label, expectation),
            label=label,
            instruction=original.strip(),
            cues=cues,
            expectation=expectation,
            importance=importance,
            confidence=0.83,
            tags=tags,
            persona="courteous",
            summary="Yield the floor when the user is engaged with other people."
        )

    def _handle_night_quiet(self, sentence: str, original: str) -> BehaviorDirective | None:
        if not any(keyword in sentence for keyword in self._night_keywords):
            return None

        label = "Night quiet mode"
        expectation = "Reduce speech and volume when the environment is dark or it is late."
        cues = ["environment:low_light", "time:night"]
        importance = 0.8
        tags = ["behavior", "quiet_mode", "environmental"]

        return BehaviorDirective(
            key=self._hash_key(label, expectation),
            label=label,
            instruction=original.strip(),
            cues=cues,
            expectation=expectation,
            importance=importance,
            confidence=0.78,
            tags=tags,
            persona="calm",
            summary="Adopt a softer presence during nighttime or low-light conditions."
        )

    def _handle_generic_preference(self, sentence: str, original: str) -> BehaviorDirective | None:
        if "please" not in sentence and "kindly" not in sentence:
            return None

        words = sentence.split()
        if not words:
            return None
        # Build a short label from the first verbs/nouns.
        label_tokens = [token for token in words[:5] if token not in {"please", "kindly", "you"}]
        label = "User preference: " + " ".join(label_tokens).title()
        expectation = original.strip().capitalize()
        cues = ["conversation:general"]
        importance = 0.7
        tags = ["behavior", "preference"]

        return BehaviorDirective(
            key=self._hash_key(label, expectation),
            label=label,
            instruction=original.strip(),
            cues=cues,
            expectation=expectation,
            importance=importance,
            confidence=0.7,
            tags=tags,
            persona=None,
            summary="Learned explicit user preference."
        )


__all__ = ["BehaviorLearner", "BehaviorDirective"]
