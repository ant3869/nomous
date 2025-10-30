# Title: WebSocket handlers
# Path: backend/handlers.py
# Purpose: Route inbound UI messages to subsystems and push protocol events back.

from __future__ import annotations
import json, asyncio
from .audio import STTEngine
from .tts import PiperTTS
from .llm import LocalLLM
from .protocol import msg_event, msg_status, msg_metrics, msg_pong

class Runtime:
    def __init__(self, broadcaster):
        self.bc = broadcaster
        self.stt = STTEngine(broadcaster)
        self.tts = PiperTTS(broadcaster)
        self.llm = LocalLLM(broadcaster)
        # simple behavior metrics cache
        self.metrics = dict(onTopic=0.9, brevity=0.5, responsiveness=0.9, nonsenseRate=0.05, rewardTotal=0.0)

    async def handle(self, ws, data: dict):
        t = data.get("type")
        if t == "ping":
            self.bc.send(msg_pong())
        elif t == "audio":
            text = self.stt.feed_pcm16_b64(data.get("pcm16",""))
            if text:
                self.bc.send(msg_status("thinking", text))
                out = self.llm.generate(text)
                if out:
                    self.tts.say(out)
                    self.bc.send(msg_status("idle", "ready"))
        elif t == "toggle":
            self.bc.send(msg_event(f"toggle {data.get('what')} → {data.get('value')}"))
        elif t == "param":
            self.bc.send(msg_event(f"param {data.get('key')} = {data.get('value')}"))
        elif t == "reinforce":
            delta = float(data.get("delta", 0))
            self.metrics["rewardTotal"] = self.metrics.get("rewardTotal", 0) + delta
            self.bc.send(msg_metrics(self.metrics))
        else:
            self.bc.send(msg_event(f"unknown inbound: {data}"))
