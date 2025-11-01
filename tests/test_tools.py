#!/usr/bin/env python3
"""
Test suite for LLM tool system
Tests tool registration, execution, and integration
"""

import asyncio
try:
    import pytest
except ImportError:
    # Define a dummy decorator if pytest is not available
    class _MockMark:
        @staticmethod
        def asyncio(func):
            return func
    
    class _MockPytest:
        mark = _MockMark()
    
    pytest = _MockPytest()

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from unittest.mock import Mock, AsyncMock, patch
from src.backend.tools import Tool, ToolParameter, ToolExecutor


class MockLLM:
    """Mock LLM instance for testing."""
    def __init__(self):
        self.memory = None
        self.bridge = Mock()
        self.bridge.post = AsyncMock()
        self.recent_context = []
        self._reinforcement = 0.0


class MockMemory:
    """Mock memory store for testing."""
    def __init__(self):
        self.enabled = True
        self.interactions = []
    
    async def record_interaction(self, modality, stimulus, response, confidence=None, tags=None):
        """Mock record interaction."""
        self.interactions.append({
            "modality": modality,
            "stimulus": stimulus,
            "response": response,
            "confidence": confidence,
            "tags": tags
        })
    
    async def load_graph(self):
        """Mock load graph."""
        nodes = [
            {
                "id": "1",
                "label": "Test memory",
                "description": "This is a test memory about Python",
                "kind": "stimulus",
                "source": "text",
                "timestamp": "2024-01-01T00:00:00"
            },
            {
                "id": "2",
                "label": "Test observation",
                "description": "User prefers casual conversation",
                "kind": "concept",
                "source": "observation",
                "timestamp": "2024-01-01T00:01:00",
                "tags": ["milestone"]
            }
        ]
        edges = []
        return nodes, edges


@pytest.mark.asyncio
async def test_tool_registration():
    """Test that tools are registered correctly."""
    llm = MockLLM()
    executor = ToolExecutor(llm)
    
    # Check that builtin tools are registered
    assert len(executor.tools) > 0
    assert "search_memory" in executor.tools
    assert "record_observation" in executor.tools
    assert "evaluate_interaction" in executor.tools
    assert "track_milestone" in executor.tools
    
    print(f"✅ Registered {len(executor.tools)} tools")


@pytest.mark.asyncio
async def test_tool_schema_generation():
    """Test OpenAI-compatible schema generation."""
    llm = MockLLM()
    executor = ToolExecutor(llm)
    
    schema = executor.get_tools_schema()
    
    assert isinstance(schema, list)
    assert len(schema) > 0
    
    # Check first tool has required fields
    first_tool = schema[0]
    assert "name" in first_tool
    assert "description" in first_tool
    assert "parameters" in first_tool
    
    print(f"✅ Generated schema for {len(schema)} tools")


@pytest.mark.asyncio
async def test_tool_prompt_generation():
    """Test tool prompt generation."""
    llm = MockLLM()
    executor = ToolExecutor(llm)
    
    prompt = executor.get_tools_prompt()
    
    assert isinstance(prompt, str)
    assert "search_memory" in prompt
    assert "TOOL_CALL:" in prompt
    
    print("✅ Tool prompt generated successfully")


@pytest.mark.asyncio
async def test_parse_tool_calls():
    """Test parsing tool calls from LLM output."""
    llm = MockLLM()
    executor = ToolExecutor(llm)
    
    # Test with valid tool call
    text = """Here's my response.
TOOL_CALL: {"tool": "search_memory", "args": {"query": "test", "limit": 3}}
And here's more text."""
    
    calls = executor.parse_tool_calls(text)
    
    assert len(calls) == 1
    assert calls[0]["tool"] == "search_memory"
    assert calls[0]["args"]["query"] == "test"
    assert calls[0]["args"]["limit"] == 3
    
    print("✅ Tool call parsing works correctly")


@pytest.mark.asyncio
async def test_parse_multiple_tool_calls():
    """Test parsing multiple tool calls."""
    llm = MockLLM()
    executor = ToolExecutor(llm)
    
    text = """Let me use some tools.
TOOL_CALL: {"tool": "recall_recent_context", "args": {"count": 5}}
And another one:
TOOL_CALL: {"tool": "analyze_sentiment", "args": {}}"""
    
    calls = executor.parse_tool_calls(text)
    
    assert len(calls) == 2
    assert calls[0]["tool"] == "recall_recent_context"
    assert calls[1]["tool"] == "analyze_sentiment"
    
    print("✅ Multiple tool call parsing works")


@pytest.mark.asyncio
async def test_recall_recent_context():
    """Test recall_recent_context tool."""
    llm = MockLLM()
    llm.recent_context = [
        {"type": "user_text", "content": "Hello", "timestamp": 1},
        {"type": "assistant", "content": "Hi there", "timestamp": 2},
        {"type": "user_text", "content": "How are you?", "timestamp": 3}
    ]
    
    executor = ToolExecutor(llm)
    result = await executor.execute_tool("recall_recent_context", {"count": 2})
    
    assert result["success"] is True
    assert "result" in result
    assert result["result"]["count"] == 2
    assert len(result["result"]["context"]) == 2
    
    print("✅ recall_recent_context works correctly")


@pytest.mark.asyncio
async def test_record_observation():
    """Test record_observation tool."""
    llm = MockLLM()
    llm.memory = MockMemory()
    
    executor = ToolExecutor(llm)
    result = await executor.execute_tool("record_observation", {
        "observation": "User likes technical details",
        "category": "user_preference",
        "importance": 8
    })
    
    assert result["success"] is True
    assert result["result"]["success"] is True
    assert len(llm.memory.interactions) == 1
    assert llm.memory.interactions[0]["modality"] == "observation"
    
    print("✅ record_observation works correctly")


@pytest.mark.asyncio
async def test_evaluate_interaction():
    """Test evaluate_interaction tool."""
    llm = MockLLM()
    llm.memory = MockMemory()
    llm.reinforce = AsyncMock()
    
    executor = ToolExecutor(llm)
    result = await executor.execute_tool("evaluate_interaction", {
        "quality_score": 8,
        "what_worked": "Good explanation",
        "what_to_improve": "Be more concise"
    })
    
    assert result["success"] is True
    assert result["result"]["quality_score"] == 8
    assert llm.reinforce.called
    
    print("✅ evaluate_interaction works correctly")


@pytest.mark.asyncio
async def test_search_memory():
    """Test search_memory tool."""
    llm = MockLLM()
    llm.memory = MockMemory()
    
    executor = ToolExecutor(llm)
    result = await executor.execute_tool("search_memory", {
        "query": "Python",
        "limit": 5
    })
    
    assert result["success"] is True
    assert result["result"]["found"] >= 0
    assert "results" in result["result"]
    
    print("✅ search_memory works correctly")


@pytest.mark.asyncio
async def test_identify_pattern():
    """Test identify_pattern tool."""
    llm = MockLLM()
    llm.memory = MockMemory()
    
    executor = ToolExecutor(llm)
    result = await executor.execute_tool("identify_pattern", {
        "pattern": "User asks technical questions in the morning",
        "occurrences": 5,
        "confidence": 0.85
    })
    
    assert result["success"] is True
    assert result["result"]["success"] is True
    assert len(llm.memory.interactions) == 1
    
    print("✅ identify_pattern works correctly")


@pytest.mark.asyncio
async def test_track_milestone():
    """Test track_milestone tool."""
    llm = MockLLM()
    llm.memory = MockMemory()
    llm.bridge = Mock()
    llm.bridge.post = AsyncMock()
    
    executor = ToolExecutor(llm)
    result = await executor.execute_tool("track_milestone", {
        "milestone": "First successful tool use",
        "category": "technical",
        "notes": "Used search_memory effectively"
    })
    
    assert result["success"] is True
    assert result["result"]["success"] is True
    assert llm.bridge.post.called
    
    print("✅ track_milestone works correctly")


@pytest.mark.asyncio
async def test_analyze_sentiment():
    """Test analyze_sentiment tool."""
    llm = MockLLM()
    llm.recent_context = [
        {"type": "user_text", "content": "This is great! I love it!"},
        {"type": "user_text", "content": "Thank you so much!"}
    ]
    
    executor = ToolExecutor(llm)
    result = await executor.execute_tool("analyze_sentiment", {})
    
    assert result["success"] is True
    assert result["result"]["sentiment"] in ["positive", "negative", "neutral"]
    assert "confidence" in result["result"]
    assert "suggestion" in result["result"]
    
    # Should be positive given the context
    assert result["result"]["sentiment"] == "positive"
    
    print("✅ analyze_sentiment works correctly")


@pytest.mark.asyncio
async def test_check_appropriate_response():
    """Test check_appropriate_response tool."""
    llm = MockLLM()
    
    executor = ToolExecutor(llm)
    
    # Test appropriate response
    result = await executor.execute_tool("check_appropriate_response", {
        "proposed_response": "That's a great question! Let me help."
    })
    
    assert result["success"] is True
    assert result["result"]["appropriate"] is True
    
    # Test inappropriate response (too long)
    long_response = "x" * 600
    result = await executor.execute_tool("check_appropriate_response", {
        "proposed_response": long_response
    })
    
    assert result["success"] is True
    assert result["result"]["appropriate"] is False
    assert "length_ok" in result["result"]["issues"]
    
    print("✅ check_appropriate_response works correctly")


@pytest.mark.asyncio
async def test_get_current_capabilities():
    """Test get_current_capabilities tool."""
    llm = MockLLM()
    llm.memory = MockMemory()
    llm._reinforcement = 15.5
    
    executor = ToolExecutor(llm)
    
    # Add some execution history
    executor.execution_history = [
        {"tool": "search_memory", "success": True},
        {"tool": "record_observation", "success": True}
    ]
    
    result = await executor.execute_tool("get_current_capabilities", {})
    
    assert result["success"] is True
    assert result["result"]["tools_available"] > 0
    assert result["result"]["reinforcement_score"] == 15.5
    assert "tool_categories" in result["result"]
    
    print("✅ get_current_capabilities works correctly")


@pytest.mark.asyncio
async def test_tool_execution_history():
    """Test that tool execution is tracked."""
    llm = MockLLM()
    llm.recent_context = []
    
    executor = ToolExecutor(llm)
    
    # Execute a tool
    await executor.execute_tool("recall_recent_context", {"count": 1})
    
    assert len(executor.execution_history) == 1
    assert executor.execution_history[0]["tool"] == "recall_recent_context"
    assert executor.execution_history[0]["success"] is True
    
    print("✅ Tool execution history tracked correctly")


@pytest.mark.asyncio
async def test_invalid_tool_call():
    """Test handling of invalid tool calls."""
    llm = MockLLM()
    executor = ToolExecutor(llm)
    
    # Test non-existent tool
    result = await executor.execute_tool("nonexistent_tool", {})
    
    assert "error" in result
    assert "Unknown tool" in result["error"]
    
    print("✅ Invalid tool calls handled correctly")


@pytest.mark.asyncio
async def test_missing_required_parameter():
    """Test handling of missing required parameters."""
    llm = MockLLM()
    executor = ToolExecutor(llm)
    
    # search_memory requires 'query' parameter
    result = await executor.execute_tool("search_memory", {})
    
    assert "error" in result
    assert "required parameter" in result["error"].lower()
    
    print("✅ Missing required parameters handled correctly")


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("  Testing LLM Tool System")
    print("=" * 60)
    print()
    
    # Get all test functions
    test_functions = [
        test_tool_registration,
        test_tool_schema_generation,
        test_tool_prompt_generation,
        test_parse_tool_calls,
        test_parse_multiple_tool_calls,
        test_recall_recent_context,
        test_record_observation,
        test_evaluate_interaction,
        test_search_memory,
        test_identify_pattern,
        test_track_milestone,
        test_analyze_sentiment,
        test_check_appropriate_response,
        test_get_current_capabilities,
        test_tool_execution_history,
        test_invalid_tool_call,
        test_missing_required_parameter
    ]
    
    async def run_all():
        for test_func in test_functions:
            try:
                await test_func()
            except Exception as e:
                print(f"❌ {test_func.__name__} failed: {e}")
                raise
    
    asyncio.run(run_all())
    
    print()
    print("=" * 60)
    print(f"  All {len(test_functions)} Tests Passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
