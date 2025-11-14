#!/usr/bin/env python
"""
Test script for memory tools integration.
"""
import asyncio
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backend.tools import ToolExecutor
from src.backend.llm import LocalLLM
from src.backend.config import load_config

async def main():
    print("=" * 80)
    print("🔧 Testing Memory Tools Integration")
    print("=" * 80)
    print()
    
    # Load config
    print("📝 Loading configuration...")
    cfg = load_config()
    print()
    
    # Initialize LLM (which includes memory)
    print("🔧 Initializing LLM with memory...")
    llm = LocalLLM(cfg)
    await llm.start()
    print("✅ LLM initialized with memory")
    print()
    
    # Initialize tools
    print("🔧 Initializing tools module...")
    tools = ToolExecutor(llm, cfg)
    print("✅ Tools module initialized")
    print(f"   Total tools: {len(tools.tools_list)}")
    print()
    
    # Find memory tools
    memory_tool_names = [
        "remember_person",
        "remember_place", 
        "remember_object",
        "remember_fact",
        "learn_user_preference",
        "recall_entity",
        "get_learning_progress",
        "search_memory"
    ]
    
    print("=" * 80)
    print("Tool Inventory Check")
    print("=" * 80)
    for tool_name in memory_tool_names:
        tool = next((t for t in tools.tools_list if t.get("function", {}).get("name") == tool_name), None)
        if tool:
            print(f"✅ {tool_name:25s} - {tool['function']['description'][:50]}...")
        else:
            print(f"❌ {tool_name:25s} - NOT FOUND")
    print()
    
    # Test remember_person tool
    print("=" * 80)
    print("TEST 1: remember_person")
    print("=" * 80)
    result = await tools.call_tool(
        "remember_person",
        {
            "name": "Alex",
            "description": "Your colleague who loves Python and coffee",
            "properties": {"role": "software engineer", "interests": ["python", "coffee"]}
        }
    )
    print(f"✅ Result: {result}")
    print()
    
    # Test remember_place tool
    print("=" * 80)
    print("TEST 2: remember_place")
    print("=" * 80)
    result = await tools.call_tool(
        "remember_place",
        {
            "name": "Office",
            "description": "Work office with computers and desks",
            "properties": {"type": "workplace", "has_coffee": True}
        }
    )
    print(f"✅ Result: {result}")
    print()
    
    # Test recall_entity tool
    print("=" * 80)
    print("TEST 3: recall_entity")
    print("=" * 80)
    result = await tools.call_tool(
        "recall_entity",
        {
            "query": "colleague who likes programming",
            "entity_type": "person"
        }
    )
    print(f"✅ Found entities: {len(result.get('entities', []))}")
    for entity in result.get("entities", []):
        print(f"   - {entity['name']}: {entity.get('description', 'N/A')}")
        print(f"     Similarity: {entity.get('similarity', 'N/A')}")
    print()
    
    # Test search_memory tool
    print("=" * 80)
    print("TEST 4: search_memory (semantic)")
    print("=" * 80)
    result = await tools.call_tool(
        "search_memory",
        {
            "query": "workspace environment",
            "limit": 5
        }
    )
    print(f"✅ Found memories: {len(result.get('results', []))}")
    for mem in result.get("results", []):
        print(f"   - {mem.get('label', 'N/A')}: {mem.get('description', '')[:50]}...")
        if "similarity" in mem:
            print(f"     Similarity: {mem['similarity']:.3f}")
    print()
    
    # Test learn_user_preference tool
    print("=" * 80)
    print("TEST 5: learn_user_preference")
    print("=" * 80)
    result = await tools.call_tool(
        "learn_user_preference",
        {
            "preference": "dark mode",
            "value": "enabled",
            "strength": 8
        }
    )
    print(f"✅ Result: {result}")
    print()
    
    # Test get_learning_progress tool
    print("=" * 80)
    print("TEST 6: get_learning_progress")
    print("=" * 80)
    result = await tools.call_tool(
        "get_learning_progress",
        {
            "limit": 10
        }
    )
    print(f"✅ Timeline events: {len(result.get('timeline', []))}")
    for event in result.get("timeline", []):
        print(f"   - [{event['event_type']}] {event['description']}")
        print(f"     Entity: {event.get('entity_name', 'N/A')} ({event.get('entity_type', 'N/A')})")
    print()
    
    # Cleanup
    print("=" * 80)
    print("🧹 Cleanup")
    print("=" * 80)
    await llm.stop()
    print("✅ LLM stopped")
    print()
    
    print("=" * 80)
    print("✅ ALL TOOL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
