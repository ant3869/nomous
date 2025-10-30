# Title: AudioSTT (Vosk streaming) - Auto-trigger LLM
# Path: backend/audio.py
# Purpose: Convert base64 PCM16 chunks to text using Vosk; automatically send to LLM

import base64
import json
import asyncio
import logging
from vosk import Model, KaldiRecognizer
from .utils import msg_event

logger = logging.getLogger(__name__)


class AudioSTT:
    def __init__(self, cfg, bridge):
        self.bridge = bridge
        self.rate = int(cfg["audio"]["sample_rate"])
        
        logger.info(f"Loading Vosk model from {cfg['paths']['vosk_model_dir']}")
        self.model = Model(cfg["paths"]["vosk_model_dir"])
        self.rec = KaldiRecognizer(self.model, self.rate)
        
        self.enabled = True
        self._lock = asyncio.Lock()
        self.llm = None  # Will be set by server after initialization
        
        logger.info("AudioSTT initialized successfully")

    def set_llm(self, llm):
        """Set LLM reference for automatic triggering."""
        self.llm = llm
        logger.info("LLM reference set for AudioSTT")

    async def stop(self):
        logger.info("AudioSTT stopping...")

    async def feed_base64_pcm(self, b64: str, rate: int):
        """Process audio chunk and automatically trigger LLM on speech."""
        if not self.enabled:
            return
        
        if rate != self.rate:
            logger.warning(f"Sample rate mismatch: expected {self.rate}, got {rate}")
            return
        
        if not b64:
            return
        
        try:
            raw = base64.b64decode(b64)
        except Exception as e:
            logger.error(f"Error decoding base64 audio: {e}")
            return
        
        async with self._lock:
            try:
                # Check if we have a complete utterance
                if self.rec.AcceptWaveform(raw):
                    result = json.loads(self.rec.Result())
                    text = (result.get("text") or "").strip()
                    
                    if text:
                        logger.info(f"STT FINAL: '{text}'")
                        await self.bridge.post(msg_event(f"ðŸŽ¤ Heard: {text}"))
                        
                        # AUTOMATICALLY TRIGGER LLM
                        if self.llm:
                            logger.info(f"Triggering LLM with audio: {text}")
                            asyncio.create_task(self.llm.process_audio(text))
                        else:
                            logger.error("LLM not set! Cannot process audio")
                            await self.bridge.post(msg_event("ERROR: LLM not connected"))
                
                # Show partial results
                else:
                    partial_result = json.loads(self.rec.PartialResult())
                    partial = partial_result.get("partial", "").strip()
                    
                    if partial:
                        # Only log significant partial results
                        if len(partial) > 3:
                            await self.bridge.post(msg_event(f"ðŸŽ¤ Listening: {partial}"))
                            
            except Exception as e:
                logger.error(f"Error processing audio: {e}", exc_info=True)
