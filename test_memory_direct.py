#!/usr/bin/env python
"""
Direct test script for memory tools - testing without full LLM initialization.
"""
import asyncio
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.backend.memory import MemoryStore
from src.backend.config import load_config

async def main():
    print("=" * 80)
    print("🔧 Direct Memory Tools Test")
    print("=" * 80)
    print()
    
    # Load config
    print("📝 Loading configuration...")
    cfg = load_config()
    print()
    
    # Initialize memory store directly
    print("🔧 Initializing memory store...")
    memory = MemoryStore(cfg)
    await memory.start()
    print("✅ Memory store initialized")
    print(f"   Embeddings: {'✅ Enabled' if memory._embed_model else '❌ Disabled'}")
    print()
    
    # Test 1: Store entities
    print("=" * 80)
    print("TEST 1: Store Person Entity")
    print("=" * 80)
    person_id = await memory.store_entity(
        entity_type="person",
        name="Michael",
        description="Your brother who plays guitar",
        properties={"relation": "family", "hobbies": ["guitar", "music"]}
    )
    print(f"✅ Stored person: {person_id}")
    print()
    
    print("=" * 80)
    print("TEST 2: Store Place Entity")
    print("=" * 80)
    place_id = await memory.store_entity(
        entity_type="place",
        name="Kitchen",
        description="Kitchen with stove and fridge",
        properties={"type": "room", "appliances": ["stove", "fridge", "microwave"]}
    )
    print(f"✅ Stored place: {place_id}")
    print()
    
    print("=" * 80)
    print("TEST 3: Store Object Entity")
    print("=" * 80)
    object_id = await memory.store_entity(
        entity_type="object",
        name="Guitar",
        description="Acoustic guitar with 6 strings",
        properties={"owner": "Michael", "type": "instrument"}
    )
    print(f"✅ Stored object: {object_id}")
    print()
    
    # Test 2: Get entities
    print("=" * 80)
    print("TEST 4: Retrieve All People")
    print("=" * 80)
    people = await memory.get_entities(entity_type="person")
    print(f"✅ Found {len(people)} person/people:")
    for person in people:
        print(f"   - {person['name']}: {person.get('description', 'N/A')}")
        print(f"     First seen: {person['first_seen']}")
        print(f"     Occurrences: {person['occurrence_count']}")
    print()
    
    print("=" * 80)
    print("TEST 5: Retrieve All Entities")
    print("=" * 80)
    all_entities = await memory.get_entities()
    print(f"✅ Found {len(all_entities)} total entities:")
    for entity in all_entities:
        print(f"   - [{entity['entity_type']}] {entity['name']}: {entity.get('description', 'N/A')[:50]}...")
    print()
    
    # Test 3: Semantic search
    print("=" * 80)
    print("TEST 6: Semantic Search - 'musical instrument'")
    print("=" * 80)
    results = await memory.semantic_search(
        query="musical instrument",
        limit=5,
        similarity_threshold=0.5
    )
    print(f"✅ Found {len(results)} results:")
    for result in results:
        print(f"   - {result['label']} (similarity: {result['similarity']:.3f})")
        print(f"     {result.get('description', 'N/A')[:60]}...")
    print()
    
    print("=" * 80)
    print("TEST 7: Semantic Search - 'family member'")
    print("=" * 80)
    results = await memory.semantic_search(
        query="family member",
        limit=5,
        similarity_threshold=0.5
    )
    print(f"✅ Found {len(results)} results:")
    for result in results:
        print(f"   - {result['label']} (similarity: {result['similarity']:.3f})")
        print(f"     {result.get('description', 'N/A')[:60]}...")
    print()
    
    # Test 4: Learning timeline
    print("=" * 80)
    print("TEST 8: Learning Timeline")
    print("=" * 80)
    timeline = await memory.get_learning_timeline(limit=10)
    print(f"✅ Found {len(timeline)} timeline events:")
    for event in timeline:
        print(f"   - [{event['event_type']}] {event['description']}")
        print(f"     Entity: {event.get('entity_name', 'N/A')} ({event.get('entity_type', 'N/A')})")
        print(f"     Time: {event['timestamp']}")
    print()
    
    # Test 5: Store same entity again (reinforcement)
    print("=" * 80)
    print("TEST 9: Entity Reinforcement")
    print("=" * 80)
    print("Storing 'Michael' again to test occurrence counting...")
    person_id2 = await memory.store_entity(
        entity_type="person",
        name="Michael",
        description="Your brother who plays guitar and sings"
    )
    print(f"✅ Stored person again: {person_id2}")
    print()
    
    # Check occurrence count
    people = await memory.get_entities(entity_type="person")
    michael = next((p for p in people if p['name'] == "Michael"), None)
    if michael:
        print(f"✅ Michael's occurrence count: {michael['occurrence_count']}")
        print(f"   Description updated: {michael['description']}")
    print()
    
    # Check timeline for reinforcement
    print("📅 Checking timeline for reinforcement event...")
    timeline = await memory.get_learning_timeline(limit=5)
    reinforcement_events = [e for e in timeline if e['event_type'] == 'reinforcement']
    print(f"✅ Found {len(reinforcement_events)} reinforcement event(s)")
    for event in reinforcement_events:
        print(f"   - {event['description']}")
    print()
    
    # Cleanup
    print("=" * 80)
    print("🧹 Cleanup")
    print("=" * 80)
    await memory.stop()
    print("✅ Memory store stopped")
    print()
    
    print("=" * 80)
    print("✅ ALL DIRECT MEMORY TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
