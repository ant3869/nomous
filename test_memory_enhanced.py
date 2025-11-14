"""
Test script for enhanced memory system with vector embeddings and entity recognition.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backend.memory import MemoryStore
from src.backend.config import load_config


async def test_memory_system():
    """Test the enhanced memory system."""
    print("="*80)
    print("🧠 Testing Enhanced Memory System")
    print("="*80)
    
    # Load configuration
    print("\n📝 Loading configuration...")
    cfg = load_config()
    
    # Create a mock bridge
    class MockBridge:
        async def post(self, msg):
            print(f"   [Bridge] {msg.get('type', 'message')}: {msg.get('message', '')[:100]}")
    
    bridge = MockBridge()
    
    # Initialize memory store
    print("\n🔧 Initializing memory store...")
    memory = MemoryStore(cfg, bridge)
    
    if not memory.enabled:
        print("❌ Memory system not enabled!")
        return
    
    print("✅ Memory system initialized")
    print(f"   Embeddings: {'✅ Enabled' if memory._embed_model else '❌ Disabled'}")
    
    # Test 1: Store entities
    print("\n\n" + "="*80)
    print("TEST 1: Entity Storage")
    print("="*80)
    
    print("\n👤 Remembering person: 'Sarah'...")
    person_id = await memory.store_entity(
        entity_type="person",
        name="Sarah",
        description="User's friend, works in tech",
        properties={"relationship": "friend"}
    )
    print(f"✅ Stored person with ID: {person_id}")
    
    print("\n📍 Remembering place: 'Living Room'...")
    place_id = await memory.store_entity(
        entity_type="place",
        name="Living Room",
        description="Main living area with couch and TV",
        properties={"layout": "Open concept"}
    )
    print(f"✅ Stored place with ID: {place_id}")
    
    print("\n🔧 Remembering object: 'Coffee Mug'...")
    object_id = await memory.store_entity(
        entity_type="object",
        name="Coffee Mug",
        description="Blue ceramic mug",
        properties={"location": "kitchen counter"}
    )
    print(f"✅ Stored object with ID: {object_id}")
    
    # Test 2: Record interactions with embeddings
    print("\n\n" + "="*80)
    print("TEST 2: Interaction Recording")
    print("="*80)
    
    print("\n💬 Recording interaction about Sarah...")
    await memory.record_interaction(
        modality="text",
        stimulus="Tell me about Sarah",
        response="Sarah is your friend who works in tech. She's always interested in the latest technology trends.",
        tags=["person", "sarah"]
    )
    print("✅ Interaction recorded with embeddings")
    
    print("\n👁️ Recording vision observation...")
    await memory.record_interaction(
        modality="vision",
        stimulus="I see 1 person in a moderately lit living room",
        response="Acknowledged: One person present",
        confidence=0.85,
        tags=["vision", "person_detected"]
    )
    print("✅ Vision observation recorded")
    
    # Test 3: Semantic search
    print("\n\n" + "="*80)
    print("TEST 3: Semantic Search")
    print("="*80)
    
    if memory._embed_model:
        print("\n🔍 Searching for 'friend technology'...")
        results = await memory.semantic_search("friend technology", limit=5)
        print(f"✅ Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['label']} (similarity: {result['similarity']:.3f})")
            print(f"      {result.get('description', '')[:80]}...")
    else:
        print("⚠️ Semantic search unavailable (no embedding model)")
    
    # Test 4: Get entities
    print("\n\n" + "="*80)
    print("TEST 4: Entity Retrieval")
    print("="*80)
    
    print("\n📋 Getting all people...")
    people = await memory.get_entities(entity_type="person")
    print(f"✅ Found {len(people)} person/people:")
    for person in people:
        print(f"   - {person['name']}: {person.get('description', '')[:60]}...")
    
    print("\n📋 Getting all places...")
    places = await memory.get_entities(entity_type="place")
    print(f"✅ Found {len(places)} place(s):")
    for place in places:
        print(f"   - {place['name']}: {place.get('description', '')[:60]}...")
    
    # Test 5: Learning timeline
    print("\n\n" + "="*80)
    print("TEST 5: Learning Timeline")
    print("="*80)
    
    print("\n📅 Getting learning timeline...")
    timeline = await memory.get_learning_timeline(limit=10)
    print(f"✅ Found {len(timeline)} timeline events:")
    for event in timeline[:5]:
        print(f"   - [{event['event_type']}] {event['description']}")
        print(f"     Entity: {event.get('entity_name', 'N/A')} ({event.get('entity_type', 'N/A')})")
        print(f"     Time: {event['timestamp']}")
    
    # Test 6: Memory graph
    print("\n\n" + "="*80)
    print("TEST 6: Memory Graph")
    print("="*80)
    
    print("\n🕸️ Loading memory graph...")
    nodes, edges = await memory.load_graph()
    print(f"✅ Graph loaded:")
    print(f"   Nodes: {len(nodes)}")
    print(f"   Edges: {len(edges)}")
    print(f"\n   Node types:")
    from collections import Counter
    node_types = Counter(n['kind'] for n in nodes)
    for kind, count in node_types.items():
        print(f"     - {kind}: {count}")
    
    # Cleanup
    print("\n\n" + "="*80)
    print("🧹 Cleanup")
    print("="*80)
    await memory.stop()
    print("✅ Memory system stopped")
    
    print("\n\n" + "="*80)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    try:
        asyncio.run(test_memory_system())
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
