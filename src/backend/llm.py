# Title: LocalLLM (llama.cpp) - Autonomous with Vision
# Path: backend/llm.py
# Purpose: Autonomous LLM that processes text, vision, and speaks unprompted

import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Optional

from llama_cpp import Llama

from .memory import MemoryStore
from .utils import msg_event, msg_token, msg_speak, msg_status
from .tools import ToolExecutor

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are Nomous, an autonomous multimodal AI orchestrator. Support the "
    "operator with concise, confident guidance, coordinate sensors and tools, "
    "and narrate decisions with a collaborative tone."
)

DEFAULT_THINKING_PROMPT = (
    "Think in small, verifiable steps. Reference available tools and memories, "
    "note uncertainties, and decide on an action plan before committing to a "
    "response. Keep thoughts structured and actionable."
)


class LocalLLM:
    def __init__(
        self,
        cfg,
        bridge,
        tts,
        memory: Optional[MemoryStore] = None,
        loop: asyncio.AbstractEventLoop | None = None,
        *,
        load_immediately: bool = True,
    ):
        self.bridge = bridge
        self.tts = tts
        self.memory = memory
        self._cfg = cfg
        self._loop = loop or asyncio.get_event_loop()
        self._last_progress_sent = -1
        self._last_progress_detail: str | None = None

        self.model_path = cfg["paths"]["gguf_path"]
        self.model: Optional[Llama] = None

        llm_cfg = cfg.get("llm", {})
        self.system_prompt = str(llm_cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT).strip()
        self.thinking_prompt = str(llm_cfg.get("thinking_prompt") or DEFAULT_THINKING_PROMPT).strip()

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
        
        # Tool executor for function calling
        self.tools = ToolExecutor(self)
        self.tools_enabled = cfg["llm"].get("tools_enabled", True)
        logger.info(f"Tool system initialized with {len(self.tools.tools)} tools")

        if load_immediately:
            self._load_model_sync(self.model_path)

    @classmethod
    async def create(cls, cfg, bridge, tts, memory: Optional[MemoryStore], loop: asyncio.AbstractEventLoop | None = None):
        instance = cls(cfg, bridge, tts, memory, loop, load_immediately=False)
        await instance._load_model_async(instance.model_path, announce=True)
        logger.info("LLM initialized successfully")
        return instance

    def _schedule_bridge_post(self, payload):
        if not self.bridge:
            return

        try:
            asyncio.run_coroutine_threadsafe(self.bridge.post(payload), self._loop)
        except RuntimeError:
            # Event loop not running (shutdown), ignore
            pass

    def _emit_load_progress(self, progress: int, detail: str | None = None):
        if not self.bridge:
            return

        if progress == self._last_progress_sent and detail == self._last_progress_detail:
            return

        self._last_progress_sent = progress
        self._last_progress_detail = detail

        payload = {
            "type": "load_progress",
            "target": "llm",
            "label": "Loading offline language model",
            "progress": max(0, min(100, int(progress)))
        }
        if detail:
            payload["detail"] = detail

        self._schedule_bridge_post(payload)

    def _load_model_sync(self, model_path: str, announce: bool = False):
        model_path = str(model_path)
        logger.info(f"Loading LLM from {model_path} (sync)")

        model_name = Path(model_path).name

        if announce:
            self._schedule_bridge_post(msg_event(f"Loading language model: {model_name}"))

        self._last_progress_sent = -1
        self._last_progress_detail = None
        self._emit_load_progress(0, "Initializing…")

        model = self._create_model(model_path)
        self.model = model
        self.model_path = model_path

        self._emit_load_progress(100, "Model ready")

        if announce:
            self._schedule_bridge_post(msg_event(f"LLM model → {model_name}"))

    def _create_model(self, model_path: str) -> Llama:
        llm_cfg = self._cfg.get("llm", {})
        n_gpu_layers_cfg = llm_cfg.get("n_gpu_layers", 0)

        try:
            n_gpu_layers = int(n_gpu_layers_cfg)
        except (TypeError, ValueError):
            logger.warning("Invalid n_gpu_layers value %r; defaulting to CPU", n_gpu_layers_cfg)
            n_gpu_layers = 0

        if n_gpu_layers > 0:
            logger.info(f"GPU acceleration enabled: {n_gpu_layers} layers on GPU")

        def _progress(current: int, total: int) -> bool:
            percent = int((current / total) * 100) if total else 0
            detail = f"{percent}% loaded" if total else "Initializing..."
            self._emit_load_progress(percent, detail)
            return True

        return Llama(
            model_path=model_path,
            n_ctx=llm_cfg["n_ctx"],
            n_threads=llm_cfg["n_threads"],
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            progress_callback=_progress
        )

    async def _load_model_async(self, model_path: str, announce: bool = False):
        model_path = str(model_path)
        logger.info(f"Loading LLM from {model_path}")

        model_name = Path(model_path).name

        if announce:
            await self.bridge.post(msg_event(f"Loading language model: {model_name}"))

        self._last_progress_sent = -1
        self._last_progress_detail = None
        self._emit_load_progress(0, "Initializing…")

        model = await asyncio.to_thread(self._create_model, model_path)
        self.model = model
        self.model_path = model_path

        self._emit_load_progress(100, "Model ready")

        if announce:
            await self.bridge.post(msg_event(f"LLM model → {model_name}"))

    def _sanitize_response(self, text: str) -> str:
        """Remove stage directions/emotes and keep output brief."""
        if not text:
            return ""

        def _strip_stage(match) -> str:
            inner = match.group(1).strip()
            # Keep if it looks like a full sentence or URL
            if not inner:
                return " "
            if any(punct in inner for punct in ".!?"):
                return match.group(0)
            if len(inner.split()) > 6:
                return match.group(0)
            return " "

        cleaned = text
        cleaned = re.sub(r"\*([^*\n]{0,80})\*", _strip_stage, cleaned)
        cleaned = re.sub(r"\(([^\)\n]{0,80})\)", _strip_stage, cleaned)
        cleaned = re.sub(r"\[([^\]\n]{0,80})\]", _strip_stage, cleaned)
        cleaned = re.sub(r"(?i)\b(?:assistant|nomous|ai|system|bot)\s*:\s*", "", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

        # Remove meta-instruction sentences that echo internal guidance
        raw_sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        filtered_sentences = []
        meta_keywords = [
            "tool", "instruction", "decision", "milestone", "memory",
            "system prompt", "thinking prompt", "tool_call", "respond with",
            "visual observation", "available tools", "use them",
        ]
        for sentence in raw_sentences:
            stripped = sentence.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            # Filter meta keywords by substring
            if any(keyword in lower for keyword in meta_keywords):
                continue
            # Filter sentences starting with "okay," (case-insensitive, at sentence start)
            if re.match(r"^(okay,)\b", stripped, re.IGNORECASE):
                continue
            if re.match(r"^[A-Z\s]{3,}:", stripped):
                continue
            filtered_sentences.append(stripped)

        if not filtered_sentences:
            for sentence in raw_sentences:
                stripped = sentence.strip()
                if not stripped:
                    continue
                lower = stripped.lower()
                if ":" in stripped:
                    continue
                if any(keyword in lower for keyword in ("tool", "instruction", "system")):
                    continue
                filtered_sentences.append(stripped)
                break

        if not filtered_sentences:
            filtered_sentences = [cleaned]

        # Limit to at most two sentences to avoid rambling
        if len(filtered_sentences) > 2:
            filtered_sentences = filtered_sentences[:2]

        return " ".join(filtered_sentences).strip()

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

            await self._load_model_async(model_path, announce=True)
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

    async def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = str(prompt or "").strip() or DEFAULT_SYSTEM_PROMPT
        await self.bridge.post(
            {
                "type": "prompt_state",
                "system_prompt": self.system_prompt,
                "thinking_prompt": self.thinking_prompt,
            }
        )
        await self.bridge.post(
            msg_event("system prompt updated – reload the language model to bake in new instructions")
        )

    async def set_thinking_prompt(self, prompt: str) -> None:
        self.thinking_prompt = str(prompt or "").strip() or DEFAULT_THINKING_PROMPT
        await self.bridge.post(
            {
                "type": "prompt_state",
                "system_prompt": self.system_prompt,
                "thinking_prompt": self.thinking_prompt,
            }
        )
        await self.bridge.post(
            msg_event("thinking prompt updated – reload the language model to apply internal guidance")
        )

    def get_prompts(self) -> dict[str, str]:
        return {
            "system_prompt": self.system_prompt,
            "thinking_prompt": self.thinking_prompt,
        }

    def _build_prompt(self, user_input: str, context_type: str = "text", include_tools: bool = True) -> str:
        """Build prompt with context, system persona, and tool instructions."""

        context_lines = []
        if self.recent_context:
            for ctx in self.recent_context[-2:]:
                context_lines.append(f"{ctx['type']}: {ctx['content'][:80]}")
        context_summary = "\n".join(context_lines) if context_lines else "No recent context recorded."

        tools_instructions = ""
        if include_tools and self.tools_enabled:
            tools_instructions = (
                "AVAILABLE TOOLS:\n"
                "You can use tools to enhance your capabilities. Available tools:\n"
                "- search_memory: Search past interactions and memories\n"
                "- recall_recent_context: Get recent conversation history\n"
                "- record_observation: Save important observations\n"
                "- evaluate_interaction: Self-evaluate your responses\n"
                "- identify_pattern: Record patterns you notice\n"
                "- analyze_sentiment: Understand emotional tone\n"
                "- check_appropriate_response: Verify response appropriateness\n"
                "- track_milestone: Record achievements\n"
                "- get_current_capabilities: Review your abilities\n\n"
                "To use a tool, include: TOOL_CALL: {\"tool\": \"tool_name\", \"args\": {\"param\": \"value\"}}\n"
                "Use tools when they help you be more helpful, remember better, or improve yourself."
            )

        if context_type == "vision":
            scenario = (
                f"Recent context:\n{context_summary}\n\n"
                f"Visual observation:\n{user_input}\n\n"
                "Respond with a single natural sentence describing what you notice."
            )
        elif context_type == "audio":
            scenario = (
                f"Recent context:\n{context_summary}\n\n"
                f"Transcript:\n{user_input}\n\n"
                "Reply in one or two warm, collaborative sentences."
            )
        elif context_type == "autonomous":
            scenario = (
                f"Recent context:\n{context_summary}\n\n"
                "Produce a short internal reflection or tool call that keeps momentum going."
            )
        else:
            scenario = (
                f"Recent context:\n{context_summary}\n\n"
                f"User message:\n{user_input}\n\n"
                "Respond as an empathetic collaborator with concise, high-signal language."
            )

        if tools_instructions:
            scenario = f"{scenario}\n\n{tools_instructions}"

        sections = []
        if self.system_prompt:
            sections.append(self.system_prompt)
        if self.thinking_prompt:
            sections.append(f"THOUGHT PROCESS GUIDANCE:\n{self.thinking_prompt}")
        sections.append(scenario)

        return "\n\n".join(part.strip() for part in sections if part and part.strip())

    async def _generate(self, prompt: str, max_tokens: int | None = None, min_tokens: int = 10) -> str:
        """Generate response from model with token streaming."""
        if self.model is None:
            raise RuntimeError("LLM model not loaded")

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

            # Process tool calls if enabled
            if self.tools_enabled and "TOOL_CALL:" in response:
                tool_calls = self.tools.parse_tool_calls(response)
                if tool_calls:
                    logger.info(f"Found {len(tool_calls)} tool call(s) in response")
                    await self.bridge.post({"type": "thought", "text": f"Executing {len(tool_calls)} tool(s)..."})
                    
                    tool_results = []
                    for call in tool_calls:
                        tool_name = call.get("tool")
                        args = call.get("args", {})
                        result = await self.tools.execute_tool(tool_name, args)
                        tool_results.append({
                            "tool": tool_name,
                            "result": result
                        })
                        
                        # Send tool result to UI
                        await self.bridge.post({
                            "type": "tool_result",
                            "tool": tool_name,
                            "result": result
                        })
                    
                    # Remove tool calls from response for speaking
                    response_lines = []
                    for line in response.split('\n'):
                        if 'TOOL_CALL:' not in line:
                            response_lines.append(line)
                    response = '\n'.join(response_lines).strip()

            sanitized = self._sanitize_response(response)
            if sanitized:
                response = sanitized

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
                if self.memory:
                    await self.memory.record_interaction("text", user_text, response)
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
                if self.memory:
                    await self.memory.record_interaction("audio", transcribed_text, response)
                
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
                if self.memory:
                    await self.memory.record_interaction("vision", description, response, tags=["vision"])
                
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
                if self.memory:
                    await self.memory.record_interaction("autonomous", "internal_reflection", response)
            else:
                logger.info("Autonomous thought: no output")

        finally:
            self._processing = False
