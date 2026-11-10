#!/usr/bin/env python3
"""Tests for the MCP-style tool registry (Part 1 architecture work).

Covers: MCP schema shape, OpenAI schema derived from the same definitions,
ToolContext decoupling (tools run with no LLM in scope), and the bridge's
tools/list + tools/call surface through ToolExecutor.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.backend.tools import ToolContext, ToolExecutor


class MockLLM:
    """Mock LLM instance for testing."""

    def __init__(self):
        self.memory = None
        self.bridge = Mock()
        self.bridge.post = AsyncMock()
        self.recent_context = []
        self._reinforcement = 0.0
        self.person_tracker = None


def test_mcp_schema_shape():
    executor = ToolExecutor(MockLLM())
    tools = executor.get_mcp_tools()

    assert len(tools) == len(executor.tools) > 0
    names = set()
    for schema in tools:
        assert set(schema) == {"name", "description", "inputSchema"}, schema
        assert schema["name"] not in names
        names.add(schema["name"])
        input_schema = schema["inputSchema"]
        assert input_schema["type"] == "object"
        assert isinstance(input_schema["properties"], dict)
        assert isinstance(input_schema["required"], list)
        # Every required param must be declared in properties.
        for required in input_schema["required"]:
            assert required in input_schema["properties"]
        for param_name, param_def in input_schema["properties"].items():
            assert param_def["type"] in {
                "string", "number", "boolean", "array", "object",
            }, (schema["name"], param_name)
            assert param_def["description"]


def test_openai_schema_derives_from_mcp_schema():
    executor = ToolExecutor(MockLLM())
    openai = {s["name"]: s for s in executor.get_tools_schema()}
    mcp = {s["name"]: s for s in executor.get_mcp_tools()}

    assert set(openai) == set(mcp)
    for name in openai:
        assert openai[name]["description"] == mcp[name]["description"]
        assert openai[name]["parameters"] == mcp[name]["inputSchema"]


def test_tool_context_from_llm_shares_live_context():
    llm = MockLLM()
    context = ToolContext.from_llm(llm)

    assert context.memory is None
    assert context.bridge is llm.bridge
    assert context.reinforcement == 0.0
    # The context holds the SAME list object the LLM appends to, not a copy.
    llm.recent_context.append({"type": "user_text", "content": "hello"})
    assert context.recent_context == [{"type": "user_text", "content": "hello"}]


def test_tool_context_from_none_is_empty():
    context = ToolContext.from_llm(None)
    assert context.memory is None
    assert context.bridge is None
    assert context.recent_context == []
    assert context.reinforcement == 0.0
    assert context.reinforce is None


@pytest.mark.asyncio
async def test_execute_tool_with_context_only_no_llm():
    """The registry works with an explicit context and no LLM instance."""
    context = ToolContext(recent_context=[
        {"type": "user_text", "content": "hello", "timestamp": 1.0},
    ])
    executor = ToolExecutor(context=context)

    result = await executor.execute_tool("recall_recent_context", {"count": 5})
    assert result["success"] is True
    assert result["result"]["count"] == 1

    listed = await executor.execute_tool("list_available_tools", {})
    assert listed["success"] is True
    assert listed["result"]["total"] == len(executor.tools)


@pytest.mark.asyncio
async def test_execute_tool_unknown_and_validation_errors():
    executor = ToolExecutor(MockLLM())

    unknown = await executor.execute_tool("no_such_tool", {})
    assert unknown["success"] is False
    assert "Unknown tool" in unknown["error"]

    # search_memory requires `query`.
    missing = await executor.execute_tool("search_memory", {})
    assert missing["success"] is False
    assert "Missing required parameter: query" in missing["error"]

    # Enum values are validated.
    bad_enum = await executor.execute_tool(
        "search_memory", {"query": "x", "modality": "bogus"})
    assert bad_enum["success"] is False
    assert "Must be one of" in bad_enum["error"]


@pytest.mark.asyncio
async def test_reinforcement_flows_through_context():
    llm = MockLLM()
    called = {}

    async def fake_reinforce(delta):
        called["delta"] = delta

    llm.reinforce = fake_reinforce
    llm._reinforcement = 3.5
    executor = ToolExecutor(llm)

    result = await executor.execute_tool(
        "evaluate_interaction",
        {"quality_score": 8, "what_worked": "fast", "what_to_improve": "none"},
    )
    assert result["success"] is True
    # (8 - 5) * 2 = +6 reinforcement.
    assert called["delta"] == 6

    caps = await executor.execute_tool("get_current_capabilities", {})
    assert caps["result"]["reinforcement_score"] == 3.5
