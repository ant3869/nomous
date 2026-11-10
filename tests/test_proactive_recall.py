#!/usr/bin/env python3
"""Tests for proactive memory recall (finding 11).

When a known face is present, the vision prompt gets a one-line
"last time you saw this person" recall.
"""

import asyncio
import sys
import types
from pathlib import Path

import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if "llama_cpp" not in sys.modules:
    llama_stub = types.ModuleType("llama_cpp")
    llama_stub.Llama = object
    sys.modules["llama_cpp"] = llama_stub

from src.backend.llm import LocalLLM
from src.backend.person_tracker import PersonTracker, TrackedPerson


def test_recall_empty_without_conversations():
    person = TrackedPerson(person_id="person_1")
    assert person.get_last_seen_recall() == ""


def test_recall_uses_topics_when_available():
    person = TrackedPerson(person_id="person_1", name="Maya")
    person.add_conversation("what should I cook?", "Try tacos.", ["food", "cooking"])
    recall = person.get_last_seen_recall()
    assert recall == "Last time you saw Maya (just now): talked about food, cooking."


def test_recall_falls_back_to_user_message():
    person = TrackedPerson(person_id="person_1", name="Maya")
    person.add_conversation("did you see the game last night?", "No, missed it.")
    recall = person.get_last_seen_recall()
    assert recall.startswith("Last time you saw Maya (just now): they said")
    assert "did you see the game last night?" in recall


def test_recall_uses_display_name_when_unnamed():
    person = TrackedPerson(person_id="person_1")
    person.visual_signature.typical_position = "left"
    person.add_conversation("hello", "hi there.", ["greeting"])
    recall = person.get_last_seen_recall()
    assert recall.startswith("Last time you saw person on the left (just now):")


@pytest.mark.asyncio
async def test_get_all_present_includes_recall():
    tracker = PersonTracker()
    await tracker.process_frame([(10, 10, 50, 50)], 640, 360)
    person = tracker.active_persons["person_1"]
    person.add_conversation("remember the project?", "Yes, the deck.", ["work"])
    present = await tracker.get_all_present()
    assert len(present) == 1
    assert present[0]["last_seen_recall"].startswith(
        "Last time you saw person on the left (just now): talked about work."
    )


@pytest.mark.asyncio
async def test_get_all_present_recall_empty_for_stranger():
    tracker = PersonTracker()
    await tracker.process_frame([(10, 10, 50, 50)], 640, 360)
    present = await tracker.get_all_present()
    assert present[0]["last_seen_recall"] == ""


@pytest.mark.asyncio
async def test_person_context_includes_recall_line():
    llm = LocalLLM.__new__(LocalLLM)
    llm.person_tracker = types.SimpleNamespace(
        get_all_present=lambda: _async_return([
            {
                "name": "Maya",
                "display_name": "Maya",
                "familiarity": 0.8,
                "should_ask_name": False,
                "last_seen_recall": "Last time you saw Maya (just now): talked about food.",
            },
            {
                "name": None,
                "display_name": "person_2",
                "familiarity": 0.0,
                "should_ask_name": False,
                "last_seen_recall": "",
            },
        ])
    )
    context = await llm._get_person_context()
    assert "People present: Maya (familiar), person_2" in context
    assert "Last time you saw Maya (just now): talked about food." in context
    # The stranger contributes no recall line.
    assert context.count("Last time you saw") == 1


async def _async_return(value):
    return value
