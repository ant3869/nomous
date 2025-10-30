# Title: LocalLLM (llama.cpp) - Autonomous with Vision
# Path: backend/llm.py
# Purpose: Autonomous LLM that processes text, vision, and speaks unprompted

import asyncio
import logging
import time
from pathlib import Path
from llama_cpp import Llama
from .utils import msg_event, msg_token, msg_speak, msg_status

logger = logging.getLogger(__name__)


class LocalLLM:
    def __init__(self, cfg, bridge, tts):
        self.bridge = bridge
        self.tts = tts
        self._cfg = cfg
        p = cfg["paths"]

        logger.info(f"Loading LLM from {p['gguf_path']}")

        n_gpu_layers = int(cfg["llm"].get("n_gpu_layers", 0))
        if n_gpu_layers > 0:
            logger.info(f"GPU acceleration enabled: {n_gpu_layers} layers on GPU")
        
        self.model_path = p["gguf_path"]
        self.model = Llama(
            model_path=self.model_path,
            n_ctx=cfg["llm"]["n_ctx"],
            n_threads=cfg["llm"]["n_threads"],
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )

        self.temperature = float(cfg["llm"]["temperature"])
        self.top_p = float(cfg["llm"]["top_p"])
        self.max_tokens = int(cfg["llm"].get("max_tokens", 256))
        self._reinforcement = 0.0

        # Autonomous behavior
        self.autonomous_mode = True
        self.last_vision_analysis = 0
        self.vision_cooldown = 15  # seconds between autonomous vision checks
        self.last_thought = 0
        self.thought_cooldown = 30  # seconds between unprompted thoughts
        
        # Context memory
        self.recent_context = []
        self.max_context_items = 5
        
        # Processing lock
        self._processing = False
        self._lock = asyncio.Lock()

        logger.info("LLM initialized successfully")

    async def stop(self):
        logger.info("LLM stopping...")

    def update_sampling(self, temperature: float | None = None, max_tokens: int | None = None):
        if temperature is not None:
            self.temperature = float(temperature)
            logger.info(f"LLM temperature set to {self.temperature:.2f}")
        if max_tokens is not None:
            self.max_tokens = max(32, int(max_tokens))
            logger.info(f"LLM max tokens set to {self.max_tokens}")

    async def reload_model(self, model_path: str):
        model_path = model_path.strip()
        if not model_path:
            raise ValueError("Model path cannot be empty")

        async with self._lock:
            if self._processing:
                raise RuntimeError("Cannot reload LLM while processing")

            logger.info(f"Reloading LLM from {model_path}")
            await self.bridge.post(msg_event("Reloading LLM model..."))

            def _load():
                return Llama(
                    model_path=model_path,
                    n_ctx=self._cfg["llm"]["n_ctx"],
                    n_threads=self._cfg["llm"]["n_threads"],
                    n_gpu_layers=int(self._cfg["llm"].get("n_gpu_layers", 0)),
                    verbose=False,
                )

            new_model = await asyncio.to_thread(_load)
            self.model = new_model
            self.model_path = model_path
            await self.bridge.post(msg_event(f"LLM model → {Path(model_path).name}"))
            logger.info("LLM model reloaded successfully")

    async def reinforce(self, delta: float):
        """Apply reinforcement learning signal."""
        self._reinforcement += float(delta)
        logger.info(f"Reinforcement applied: {delta:+.1f} (total: {self._reinforcement:+.1f})")
        await self.bridge.post(msg_event(f"reinforcement: {self._reinforcement:+.1f}"))

    def _add_context(self, context_type: str, content: str):
        """Add to rolling context memory."""
        self.recent_context.append({
            "type": context_type,
            "content": content,
            "timestamp": time.time()
        })
        if len(self.recent_context) > self.max_context_items:
            self.recent_context.pop(0)

    def _build_prompt(self, user_input: str, context_type: str = "text") -> str:
        """Build prompt with context."""
        # Add recent context
        context_str = ""
        if self.recent_context:
            for ctx in self.recent_context[-2:]:
                context_str += f"{ctx['type']}: {ctx['content'][:80]}\n"
        
        if context_type == "vision":
            prompt = f"""You're an AI that can see and speak. You notice: {user_input}

Previous: {context_str if context_str else 'Just started observing'}

Comment naturally in 1 sentence (be casual, like a friend):"""
        elif context_type == "audio":
            prompt = f"""Previous: {context_str if context_str else 'Just started'}

Person: "{user_input}"

You (reply naturally, 1-2 sentences):"""
        elif context_type == "autonomous":
            prompt = f"""You're observing your environment silently.

Recent: {context_str if context_str else 'Quiet so far'}

Think out loud briefly (1 short sentence, casual):"""
        else:
            prompt = f"""Previous: {context_str if context_str else 'New conversation'}

Person: {user_input}

You (reply naturally):"""
        
        return prompt

    async def _generate(self, prompt: str, max_tokens: int | None = None, min_tokens: int = 10) -> str:
        """Generate response from model with token streaming."""
        try:
            await self.bridge.post(msg_status("thinking", "Processing..."))

            # Send thinking process to UI
            await self.bridge.post({"type": "thought", "text": f"Prompt: {prompt[:200]}..."})

            max_tok = max_tokens or self.max_tokens
            stream = self.model(
                prompt,
                max_tokens=max_tok,
                temperature=self.temperature,
                top_p=self.top_p,
                stream=True,
                stop=["Person:", "You:", "\n\n\n", "###", "User:"],
                repeat_penalty=1.1
            )
            
            tokens = []
            total_tokens = 0
            
            for chunk in stream:
                if chunk and "choices" in chunk and len(chunk["choices"]) > 0:
                    choice = chunk["choices"][0]
                    text = choice.get("text", "")
                    
                    if choice.get("finish_reason") == "length":
                        logger.warning("Hit max token limit, response may be truncated")
                    
                    if text:
                        tokens.append(text)
                        total_tokens += max(1, len(text) // 4)
                        await self.bridge.post(msg_token(total_tokens))
                        
                        # Stream thinking chunks to UI
                        if len(tokens) % 3 == 0:  # Every 3 tokens
                            await self.bridge.post({"type": "thought", "text": f"Generating: {''.join(tokens[-9:])}"})
            
            response = "".join(tokens).strip()
            
            # Remove any role-play artifacts
            response = response.replace("You:", "").replace("Person:", "").strip()
            
            if len(response) < 3 and total_tokens < min_tokens:
                logger.warning(f"Response too short ({len(response)} chars, {total_tokens} tokens), regenerating...")
                return await self._generate(prompt, max_tokens=max_tok, min_tokens=0)
            
            logger.info(f"Generated ({total_tokens} tokens): {response[:100]}")
            await self.bridge.post({"type": "thought", "text": f"Final: {response}"})
            
            await self.bridge.post(msg_status("idle", "Ready"))
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            await self.bridge.post(msg_event(f"llm error: {str(e)}"))
            await self.bridge.post(msg_status("idle", "Error"))
            return ""

    async def chat(self, user_text: str):
        """Process user text input (manual or from UI)."""
        async with self._lock:
            if self._processing:
                logger.warning("Already processing, skipping")
                return
            self._processing = True
        
        try:
            logger.info(f"Chat input: {user_text[:100]}")
            self._add_context("user_text", user_text)
            
            prompt = self._build_prompt(user_text, "text")
            response = await self._generate(prompt)
            
            if response:
                self._add_context("assistant", response)
                yield msg_speak(response)
                await self.tts.speak(response)
        finally:
            self._processing = False

    async def process_audio(self, transcribed_text: str):
        """Process speech-to-text input (autonomous trigger)."""
        if not transcribed_text.strip():
            return
        
        async with self._lock:
            if self._processing:
                logger.warning("Already processing, skipping audio")
                return
            self._processing = True
        
        try:
            logger.info(f"Processing audio: {transcribed_text}")
            self._add_context("audio", transcribed_text)
            
            prompt = self._build_prompt(transcribed_text, "audio")
            response = await self._generate(prompt)
            
            if response:
                self._add_context("assistant", response)
                await self.bridge.post(msg_speak(response))
                await self.tts.speak(response)
                
        finally:
            self._processing = False

    async def process_vision(self, description: str):
        """Process vision input (what the camera sees)."""
        now = time.time()
        if now - self.last_vision_analysis < self.vision_cooldown:
            return  # Skip if too soon
        
        async with self._lock:
            if self._processing:
                return
            self._processing = True
        
        try:
            self.last_vision_analysis = now
            
            # DECISION: Should we comment on this?
            # Only speak if something interesting (person, gesture, or significant change)
            interesting = any(word in description.lower() for word in 
                            ['person', 'people', 'waving', 'gesture', 'pointing', 'thumbs', 'peace'])
            
            # Random chance to stay quiet (20% of the time, even if interesting)
            import random
            if not interesting or (random.random() < 0.2):
                logger.info(f"Vision (quiet): {description[:100]}")
                self._add_context("vision_quiet", description)
                await self.bridge.post({"type": "thought", "text": f"Observing: {description}"})
                return
            
            logger.info(f"Vision (speaking): {description[:100]}")
            self._add_context("vision", description)
            
            prompt = self._build_prompt(description, "vision")
            response = await self._generate(prompt, max_tokens=80)
            
            if response:
                self._add_context("assistant", response)
                await self.bridge.post(msg_speak(response))
                await self.tts.speak(response)
                
        finally:
            self._processing = False

    async def autonomous_thought(self):
        """Generate unprompted thought based on context."""
        now = time.time()
        if now - self.last_thought < self.thought_cooldown:
            return
        
        if not self.autonomous_mode:
            return
        
        async with self._lock:
            if self._processing:
                return
            self._processing = True
        
        try:
            self.last_thought = now
            
            # Random chance to skip (stay quiet 70% of the time)
            import random
            if random.random() < 0.7:
                logger.info("Autonomous thought: choosing silence")
                await self.bridge.post({"type": "thought", "text": "Quietly observing..."})
                self._processing = False
                return
            
            logger.info("Autonomous thought: speaking")
            
            prompt = self._build_prompt("", "autonomous")
            response = await self._generate(prompt, max_tokens=50)
            
            if response and len(response) > 3:
                self._add_context("assistant_autonomous", response)
                await self.bridge.post(msg_speak(response))
                await self.tts.speak(response)
            else:
                logger.info("Autonomous thought: no output")

        finally:
            self._processing = False
