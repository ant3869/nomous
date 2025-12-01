# Title: AudioSTT (Vosk streaming) - Auto-trigger LLM
# Path: backend/audio.py
# Purpose: Convert base64 PCM16 chunks to text using Vosk; automatically send to LLM

import base64
import json
import os
import asyncio
import logging
from typing import Optional
from vosk import Model, KaldiRecognizer
from .utils import msg_event

logger = logging.getLogger(__name__)


class AudioSTT:
    def __init__(self, cfg, bridge):
        self.bridge = bridge
        self.rate = int(cfg["audio"]["sample_rate"])

        self.model_path = cfg["paths"]["vosk_model_dir"]
        self.model: Optional[Model] = None
        self.rec: Optional[KaldiRecognizer] = None

        self.enabled = True
        self._lock = asyncio.Lock()
        self.llm: Optional[object] = None  # Will be set by server after initialization

        self.sensitivity = 60
        self._min_partial_chars = 3
        self._min_final_chars = 3
        self._update_thresholds()

        self._load_model()

    def set_llm(self, llm):
        """Set LLM reference for automatic triggering."""
        self.llm = llm
        logger.info("LLM reference set for AudioSTT")

    def set_enabled(self, value: bool):
        self.enabled = bool(value)
        state = "enabled" if self.enabled else "disabled"
        logger.info(f"AudioSTT {state}")

    def set_sensitivity(self, percent: int):
        self.sensitivity = max(0, min(100, int(percent)))
        self._update_thresholds()
        logger.info(f"AudioSTT sensitivity set to {self.sensitivity}% (final≥{self._min_final_chars} chars)")

    def _update_thresholds(self):
        # Higher sensitivity -> accept shorter snippets
        if self.sensitivity >= 85:
            self._min_partial_chars = 1
            self._min_final_chars = 1
        elif self.sensitivity >= 60:
            self._min_partial_chars = 2
            self._min_final_chars = 2
        elif self.sensitivity >= 35:
            self._min_partial_chars = 3
            self._min_final_chars = 3
        else:
            self._min_partial_chars = 4
            self._min_final_chars = 4

    def _load_model(self):
        """Load the default Vosk model, disabling STT if it fails."""
        model_path = self.model_path

        if not os.path.isdir(model_path):
            logger.error(f"Vosk model directory not found: {model_path}")
            self.enabled = False
            return

        try:
            logger.info(f"Loading Vosk model from {model_path}")
            self.model = Model(model_path)
            self.rec = KaldiRecognizer(self.model, self.rate)
            logger.info("AudioSTT initialized successfully")
        except Exception as exc:
            logger.error(f"Failed to load Vosk model: {exc}")
            self.model = None
            self.rec = None
            self.enabled = False

    async def reload_model(self, model_path: str):
        """Hot-reload the STT acoustic model."""
        model_path = model_path.strip()
        if not model_path:
            raise ValueError("Model path cannot be empty")

        async with self._lock:
            if not os.path.isdir(model_path):
                logger.error(f"Vosk model directory not found: {model_path}")
                self.enabled = False
                self.model = None
                self.rec = None
                return

            logger.info(f"Reloading Vosk model from {model_path}")
            new_model = await asyncio.to_thread(Model, model_path)
            new_recognizer = KaldiRecognizer(new_model, self.rate)
            self.model = new_model
            self.rec = new_recognizer
            self.model_path = model_path
            logger.info("Vosk model reloaded successfully")

    async def stop(self):
        logger.info("AudioSTT stopping...")

    async def feed_base64_pcm(self, b64: str, rate: int):
        """Process audio chunk and automatically trigger LLM on speech."""
        if not self.enabled:
            logger.debug("STT is disabled, skipping audio chunk")
            return

        if not self.model or not self.rec:
            logger.warning("STT feed skipped: Vosk model not loaded. Check model path in config.")
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
                recognizer = self.rec
                if recognizer.AcceptWaveform(raw):
                    result = json.loads(recognizer.Result())
                    text = (result.get("text") or "").strip()

                    words = [w for w in text.split() if w]
                    if text and (len(text) >= self._min_final_chars or len(words) >= 1):
                        logger.info(f"STT FINAL: '{text}'")
                        await self.bridge.post({"type": "stt", "phase": "final", "text": text, "forwarded": False})
                        await self.bridge.post(msg_event(f"🎤 You said: {text}"))

                        if self.llm:
                            logger.info(f"Triggering LLM with audio: {text}")
                            asyncio.create_task(self.llm.process_audio(text))
                            await self.bridge.post({"type": "stt", "phase": "forwarded", "text": text, "forwarded": True})
                        else:
                            logger.warning("LLM not set – skipping audio hand-off but keeping transcription visible")
                            await self.bridge.post(msg_event("⚠️ STT captured speech but no LLM is configured. Configure an LLM to enable spoken follow-ups."))
                    elif text:
                        logger.debug(
                            "Dropping STT final '%s' below threshold %d",
                            text,
                            self._min_final_chars,
                        )
                else:
                    partial_result = json.loads(recognizer.PartialResult())
                    partial = partial_result.get("partial", "").strip()

                    if partial and len(partial) >= self._min_partial_chars:
                        await self.bridge.post({"type": "stt", "phase": "partial", "text": partial})
                        await self.bridge.post(msg_event(f"🎤 Listening: {partial}"))
                            
            except Exception as e:
                logger.error(f"Error processing audio: {e}", exc_info=True)
