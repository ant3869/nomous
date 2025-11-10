from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.behavior import BehaviorDirective, BehaviorLearner


def _extract_single(stimulus: str) -> BehaviorDirective:
    learner = BehaviorLearner()
    directives = learner.extract(stimulus)
    assert directives, "expected at least one directive to be produced"
    assert len(directives) == 1, "expected a single dominant directive"
    return directives[0]


def test_detects_focus_respect_directive() -> None:
    directive = _extract_single("Please don't talk so much when you see I'm doing something important.")
    assert directive.label == "Respect focused work"
    assert directive.persona == "considerate"
    assert "user_state:busy" in directive.cues
    assert directive.importance > 0.8


def test_detects_wait_before_talking() -> None:
    directive = _extract_single("Please wait until I'm finished before talking.")
    assert directive.label == "Wait for completion"
    assert directive.persona == "patient"
    assert directive.confidence >= 0.82


def test_detects_turn_taking_when_user_is_talking_to_human() -> None:
    directive = _extract_single("If you see or hear me talking to another human be sure to wait your turn.")
    assert directive.label == "Respect human conversations"
    assert "turn_taking" in directive.tags


def test_detects_night_quiet_mode() -> None:
    directive = _extract_single("If it's night time or dark in the room try to talk less, please.")
    assert directive.label == "Night quiet mode"
    assert "environment:low_light" in directive.cues


def test_ignores_non_directive_language() -> None:
    learner = BehaviorLearner()
    directives = learner.extract("It's a beautiful day and I'm happy to chat.")
    assert directives == []


def test_consistent_key_generation() -> None:
    learner = BehaviorLearner()
    first = learner.extract("Please don't talk so much when you see I'm doing something important.")[0]
    second = learner.extract("please don't talk so much when you see i'm doing something important")[0]
    assert first.key == second.key
