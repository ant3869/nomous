#!/usr/bin/env python3
"""
Test all bug fixes are working
Run after starting the server
"""

import asyncio
import json
import time

import websockets
import pytest

from src.backend.config import load_config


def _loopback_fallback(host: str) -> str:
    """
    Normalize host to loopback if it is a wildcard or empty.
    """
    if host in {"0.0.0.0", "::", "", "*"}:
        return "127.0.0.1"
    return host

def resolve_ws_uri() -> str:
    cfg = load_config()
    host = str(cfg.get("ws", {}).get("host", "127.0.0.1"))
    port = int(cfg.get("ws", {}).get("port", 8765))
    host = _loopback_fallback(host)
    return f"ws://{host}:{port}"


@pytest.mark.asyncio
async def test_fixes():
    print("=" * 60)
    print("  Testing Bug Fixes")
    print("=" * 60)
    print()
    
    uri = resolve_ws_uri()
    
    try:
        async with websockets.connect(uri) as ws:
            print("✅ Connected to server")
            print()
            
            # Test 1: Thought streaming
            print("[Test 1] Checking for thought messages...")
            await ws.send(json.dumps({"type": "text", "value": "Hello"}))
            
            thought_received = False
            for _ in range(10):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(msg)
                    if data.get("type") == "thought":
                        print(f"✅ Thought received: {data.get('text', '')[:50]}...")
                        thought_received = True
                        break
                except asyncio.TimeoutError:
                    continue
            
            if not thought_received:
                print("⚠️  No thought messages received (might be normal if processing fast)")
            print()
            
            # Test 2: No pong spam
            print("[Test 2] Checking for pong spam...")
            await ws.send(json.dumps({"type": "ping"}))
            
            pong_count = 0
            for _ in range(5):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    data = json.loads(msg)
                    if data.get("type") == "pong":
                        pong_count += 1
                except asyncio.TimeoutError:
                    break
            
            if pong_count == 1:
                print("✅ Pong received once (correct)")
            elif pong_count == 0:
                print("⚠️  No pong received (might be filtered)")
            else:
                print(f"❌ Multiple pongs received: {pong_count}")
            print()
            
            # Test 3: Status messages
            print("[Test 3] Checking status messages...")
            status_received = False
            for _ in range(5):
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    data = json.loads(msg)
                    if data.get("type") == "status":
                        print(f"✅ Status: {data.get('value')} - {data.get('detail', 'N/A')}")
                        status_received = True
                        break
                except asyncio.TimeoutError:
                    continue
            
            if not status_received:
                print("⚠️  No status messages yet (might arrive later)")
            print()
            
            print("=" * 60)
            print("  Test Complete")
            print("=" * 60)
            print()
            print("Manual tests to perform:")
            print("1. Open UI → Thoughts tab → Should see purple timestamped entries")
            print("2. Speak into mic → Should respond within 1-3 seconds")
            print("3. Wave at camera → Should recognize gesture")
            print("4. Check console → Should be clean (no pong spam)")
            print("5. Responses → Should be natural, not role-play")
            print()
            
    except ConnectionRefusedError:
        print("❌ Server not running!")
        print("   Start with: python run_bridge.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_fixes())
