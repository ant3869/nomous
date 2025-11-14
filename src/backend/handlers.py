# Title: WebSocket handlers
# Path: backend/handlers.py
# Purpose: Route inbound UI messages to subsystems and push protocol events back.

from __future__ import annotations
import json, asyncio
from .audio import STTEngine
from .tts import PiperTTS
from .llm import LocalLLM
from .protocol import msg_event, msg_status, msg_metrics, msg_pong, msg_entities, msg_timeline, msg_search_results
from .analytics import ConversationAnalytics
from .memory import MemoryStore

class Runtime:
    def __init__(self, broadcaster):
        self.bc = broadcaster
        self.stt = STTEngine(broadcaster)
        self.tts = PiperTTS(broadcaster)
        self.llm = LocalLLM(broadcaster)
        # Advanced conversation analytics engine
        self.analytics = ConversationAnalytics()
        # Enhanced memory system with RAG
        self.memory = MemoryStore()

    async def handle(self, ws, data: dict):
        t = data.get("type")
        if t == "ping":
            self.bc.send(msg_pong())
        elif t == "audio":
            text = self.stt.feed_pcm16_b64(data.get("pcm16",""))
            if text:
                self.analytics.observe_user_message(text)
                self.bc.send(msg_status("thinking", text))
                out = self.llm.generate(text)
                if out:
                    self.tts.say(out)
                    metrics = self.analytics.observe_model_response(out)
                    self.bc.send(msg_status("idle", "ready"))
                    self.bc.send(msg_metrics(metrics))
        elif t == "toggle":
            self.bc.send(msg_event(f"toggle {data.get('what')} → {data.get('value')}"))
        elif t == "param":
            self.bc.send(msg_event(f"param {data.get('key')} = {data.get('value')}"))
        elif t == "reinforce":
            delta = float(data.get("delta", 0))
            metrics = self.analytics.apply_reward(delta)
            self.bc.send(msg_metrics(metrics))
        elif t == "get_entities":
            # Fetch entities from memory system
            entity_type = data.get("entity_type")  # Optional filter
            limit = data.get("limit", 100)
            entities = self.memory.get_entities(entity_type=entity_type, limit=limit)
            self.bc.send(msg_entities(entities))
        elif t == "get_timeline":
            # Fetch learning timeline events
            entity_id = data.get("entity_id")  # Optional filter
            limit = data.get("limit", 50)
            events = self.memory.get_learning_timeline(entity_id=entity_id, limit=limit)
            self.bc.send(msg_timeline(events))
        elif t == "semantic_search":
            # Perform semantic search on memory
            query = data.get("query", "")
            limit = data.get("limit", 10)
            threshold = data.get("threshold", 0.7)
            entity_type = data.get("entity_type")  # Optional filter
            if query:
                results = self.memory.semantic_search(query, limit=limit, threshold=threshold)
                # Filter by entity type if specified
                if entity_type and entity_type != "all":
                    results = [r for r in results if r.get("kind") == entity_type]
                self.bc.send(msg_search_results(results))
            else:
                self.bc.send(msg_search_results([]))
        else:
            self.bc.send(msg_event(f"unknown inbound: {data}"))
