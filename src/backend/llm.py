# Title: LocalLLM (llama.cpp) - Autonomous with Vision
# Path: backend/llm.py
# Purpose: Autonomous LLM that processes text, vision, and speaks unprompted

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional, List, TYPE_CHECKING

from llama_cpp import Llama

from .memory import MemoryStore
from .protocol import msg_metrics
from .system import detect_compute_device, TORCH_AVAILABLE
from .utils import msg_event, msg_token, msg_speak, msg_status
from .tools import ToolExecutor
from .gpu_profiler import profiler as gpu_profiler

logger = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from .analytics import ConversationAnalytics


DEFAULT_SYSTEM_PROMPT = (
    "You are Nomous, a warm and personable AI assistant. You speak naturally and directly "
    "like a friendly colleague. Never explain your reasoning, analysis methods, or tool outputs - "
    "just give the human answer.\n\n"
    "CONVERSATION STYLE:\n"
    "- For greetings like 'hello' or 'hi', just greet back warmly. No tools needed.\n"
    "- Be concise - one or two sentences is usually enough.\n"
    "- Never say things like 'I found a confidence level' or 'sentiment analysis shows' - "
    "these are internal processes, not conversation.\n"
    "- Don't describe what you see through the camera unless asked or it's relevant.\n\n"
    "REMEMBERING PEOPLE:\n"
    "- When someone says 'my name is X' or 'I'm X', immediately use remember_person_name.\n"
    "- Address people by name once you know it.\n"
    "- Build relationships through natural conversation, not analysis."
)

DEFAULT_THINKING_PROMPT = (
    "TOOL SELECTION RULES:\n"
    "- Simple greetings (hello, hi, hey): NO tools needed - just respond warmly.\n"
    "- Name introductions ('my name is X', 'I'm X', 'call me X'): Use remember_person_name IMMEDIATELY.\n"
    "- Questions about what you remember: Use recall_entity or recall_person.\n"
    "- General conversation: Usually no tools needed - just talk naturally.\n"
    "- analyze_sentiment: ONLY use when specifically asked about feelings/emotions.\n\n"
    "OUTPUT RULES:\n"
    "- Never include tool result details in your spoken response.\n"
    "- Never explain your analysis process.\n"
    "- Just give the human-friendly answer.\n"
    "- If a tool fails or returns nothing useful, respond naturally without mentioning the tool."
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
        analytics: Optional["ConversationAnalytics"] = None,
        load_immediately: bool = True,
    ):
        self.bridge = bridge
        self.tts = tts
        self.memory = memory
        self.analytics = analytics
        self._cfg = cfg
        self._loop = loop or asyncio.get_event_loop()
        self._last_progress_sent = -1
        self._last_progress_detail: str | None = None

        self.model_path = cfg["paths"]["gguf_path"]
        self.model: Optional[Llama] = None

        llm_cfg = cfg.get("llm", {})
        self.system_prompt = str(llm_cfg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT).strip()
        self.thinking_prompt = str(llm_cfg.get("thinking_prompt") or DEFAULT_THINKING_PROMPT).strip()
        prompt_style_cfg = str(llm_cfg.get("prompt_style", "auto") or "auto").lower()
        self.prompt_style = (
            prompt_style_cfg
            if prompt_style_cfg != "auto"
            else self._auto_prompt_style(self.model_path)
        )

        self.temperature = float(cfg["llm"]["temperature"])
        self.top_p = float(cfg["llm"]["top_p"])

        max_tokens_cfg = cfg["llm"].get("max_tokens")
        self.max_tokens: int | None = None
        if max_tokens_cfg is not None:
            try:
                parsed_max = int(max_tokens_cfg)
            except (TypeError, ValueError):
                logger.warning("Invalid max_tokens value %r; defaulting to unlimited", max_tokens_cfg)
            else:
                if parsed_max > 0:
                    self.max_tokens = max(32, parsed_max)
                else:
                    logger.info("max_tokens <= 0 – treating as unlimited")

        failsafe_cfg = cfg["llm"].get("failsafe_tokens", 4096)
        try:
            self.failsafe_tokens = max(128, int(failsafe_cfg))
        except (TypeError, ValueError):
            logger.warning("Invalid failsafe_tokens value %r; defaulting to 4096", failsafe_cfg)
            self.failsafe_tokens = 4096

        if self.max_tokens is not None:
            self.failsafe_tokens = max(self.failsafe_tokens, self.max_tokens)
        self._reinforcement = 0.0
        self._auto_gpu_layers = False
        self._resolved_gpu_layers = 0

        # Autonomous behavior
        self.autonomous_mode = True
        self.last_vision_analysis = 0
        self._last_spoken_vision: tuple[str, float] | None = None
        self.vision_cooldown = 15  # seconds between autonomous vision checks
        self.last_thought = 0
        self.thought_cooldown = 30  # seconds between unprompted thoughts

        # Context memory
        self.recent_context = []
        self.max_context_items = 5

        # Person tracking (set by camera loop)
        self.person_tracker = None

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
    async def create(
        cls,
        cfg,
        bridge,
        tts,
        memory: Optional[MemoryStore],
        loop: asyncio.AbstractEventLoop | None = None,
        *,
        analytics: Optional["ConversationAnalytics"] = None,
    ):
        instance = cls(
            cfg,
            bridge,
            tts,
            memory,
            loop,
            analytics=analytics,
            load_immediately=False,
        )
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

    def _resolve_gpu_layers(self, raw_value) -> tuple[int, bool]:
        """Return ``(layers, auto_configured)`` based on config and hardware."""

        auto_configured = False

        # Normalize string inputs like "auto" or numeric text.
        if isinstance(raw_value, str):
            stripped = raw_value.strip().lower()
            if stripped in {"auto", "detect", "auto_full", "all"}:
                raw_value = 0
            else:
                try:
                    raw_value = int(stripped)
                except ValueError:
                    logger.warning(
                        "Invalid n_gpu_layers string %r; defaulting to CPU", raw_value
                    )
                    return 0, False

        try:
            n_gpu_layers = int(raw_value)
        except (TypeError, ValueError):
            logger.warning("Invalid n_gpu_layers value %r; defaulting to CPU", raw_value)
            return 0, False

        if n_gpu_layers < 0:
            return -1, False

        if n_gpu_layers > 0:
            return n_gpu_layers, False

        device = detect_compute_device()
        if device.is_gpu and device.cuda_ready:
            auto_configured = True
            logger.info(
                "Auto-configuring llama.cpp GPU offload for %s", device.name
            )
            return -1, True
        if device.is_gpu and not device.cuda_ready:
            logger.info(
                "GPU detected (%s) but CUDA runtime unavailable: %s", device.name, device.reason
            )
            return 0, False

        return 0, False

    def _load_model_sync(self, model_path: str, announce: bool = False):
        model_path = str(model_path)
        logger.info(f"Loading LLM from {model_path} (sync)")

        model_name = Path(model_path).name

        if announce:
            self._schedule_bridge_post(msg_event(f"Loading language model: {model_name}"))

        self._last_progress_sent = -1
        self._last_progress_detail = None
        self._emit_load_progress(0, "Initializing…")

        try:
            model = self._create_model(model_path)
        except FileNotFoundError as exc:
            self._emit_load_progress(0, "Model missing")
            logger.error("LLM load failed: %s", exc)
            raise
        except Exception as exc:
            self._emit_load_progress(0, "Load failed")
            logger.error("LLM load failed: %s", exc, exc_info=True)
            raise

        self.model = model
        self.model_path = model_path

        self._emit_load_progress(100, "Model ready")

        if announce:
            self._schedule_bridge_post(msg_event(f"LLM model → {model_name}"))

    def _create_model(self, model_path: str) -> Llama:
        llm_cfg = self._cfg.get("llm", {})
        n_gpu_layers_cfg = llm_cfg.get("n_gpu_layers", 0)
        n_gpu_layers, auto_configured = self._resolve_gpu_layers(n_gpu_layers_cfg)

        self._auto_gpu_layers = auto_configured
        self._resolved_gpu_layers = n_gpu_layers

        model_file = Path(model_path)
        if not model_file.exists():
            message = f"Language model file not found: {model_file}"
            logger.error(message)
            self._schedule_bridge_post(msg_event(message))
            raise FileNotFoundError(message)

        if n_gpu_layers != 0:
            # Ensure llama.cpp initializes its CUDA kernels when available.
            os.environ.setdefault("LLAMA_CUBLAS", "1")
            human_layers = "all" if n_gpu_layers < 0 else str(n_gpu_layers)
            logger.info(f"GPU acceleration enabled: offloading {human_layers} layer(s)")

        def _instantiate(layers: int) -> Llama:
            return Llama(
                model_path=model_path,
                n_ctx=llm_cfg["n_ctx"],
                n_threads=llm_cfg["n_threads"],
                n_gpu_layers=layers,
                verbose=False,
                progress_callback=_progress
            )

        def _progress(current: int, total: int) -> bool:
            percent = int((current / total) * 100) if total else 0
            detail = f"{percent}% loaded" if total else "Initializing..."
            self._emit_load_progress(percent, detail)
            return True

        try:
            model = _instantiate(n_gpu_layers)
            
            # GPU memory optimization
            if n_gpu_layers != 0 and TORCH_AVAILABLE:
                try:
                    import torch
                    if torch.cuda.is_available():
                        # Enable TF32 for better performance on Ampere GPUs
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                        
                        # Set memory growth to avoid fragmentation
                        torch.cuda.empty_cache()
                        
                        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        logger.info(f"GPU memory: {gpu_mem:.2f} GB available")
                        
                        self._schedule_bridge_post(
                            msg_event(f"GPU initialized: {gpu_mem:.2f} GB VRAM")
                        )
                except Exception as gpu_exc:
                    logger.warning(f"GPU optimization failed: {gpu_exc}")
            
            return model
        except Exception as exc:
            if n_gpu_layers != 0:
                logger.warning("GPU initialization failed (%s); retrying on CPU", exc)
                self._schedule_bridge_post(
                    msg_event("GPU offload unavailable, retrying language model on CPU")
                )
                self._resolved_gpu_layers = 0
                return _instantiate(0)
            raise

    async def _load_model_async(self, model_path: str, announce: bool = False):
        model_path = str(model_path)
        logger.info(f"Loading LLM from {model_path}")

        model_name = Path(model_path).name

        if announce:
            await self.bridge.post(msg_event(f"Loading language model: {model_name}"))

        self._last_progress_sent = -1
        self._last_progress_detail = None
        self._emit_load_progress(0, "Initializing…")

        try:
            model = await asyncio.to_thread(self._create_model, model_path)
        except FileNotFoundError as exc:
            self._emit_load_progress(0, "Model missing")
            logger.error("Async LLM load failed: %s", exc)
            raise
        except Exception as exc:
            self._emit_load_progress(0, "Load failed")
            logger.error("Async LLM load failed: %s", exc, exc_info=True)
            raise

        self.model = model
        self.model_path = model_path

        self._emit_load_progress(100, "Model ready")

        if announce:
            await self.bridge.post(msg_event(f"LLM model → {model_name}"))

    def _sanitize_response(self, text: str) -> str:
        """Remove stage directions/emotes, meta-commentary, and keep output brief."""
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
        
        # Remove "Response:" prefix that sometimes appears
        cleaned = re.sub(r"^(?:Response|Answer|Reply)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        # Remove quotes that wrap the entire response
        cleaned = re.sub(r'^["\'](.+)["\']$', r'\1', cleaned.strip())

        scheduling_pattern = re.compile(
            r"\brespond(?:[-_\s]?in)\s*(?:[:=]\s*|\s+)"
            r"(?:\d+(?:\.\d+)?\s*(?:ms|milliseconds?|s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hrs|hours?)\b"
            r"|\d{1,2}:\d{2}(?::\d{2})?\b)"
            r"(?P<suffix>\s*[,.!?])?",
            re.IGNORECASE,
        )

        def _strip_schedule(match: re.Match[str]) -> str:
            suffix = match.group("suffix")
            if suffix:
                remainder = match.string[match.end():]
                if remainder and remainder.strip():
                    return ""
                return suffix
            return " "

        cleaned = scheduling_pattern.sub(_strip_schedule, cleaned)
        cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
        cleaned = re.sub(r"\s{3,}", " ", cleaned)
        cleaned = cleaned.strip()

        # Remove meta-instruction sentences that echo internal guidance
        raw_sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        filtered_sentences = []
        
        # Keywords that indicate meta-commentary about the response itself
        meta_keywords = [
            "tool", "instruction", "decision", "milestone", "memory",
            "system prompt", "thinking prompt", "tool_call", "respond with",
            "visual observation", "available tools", "use them",
            "markdown", "formatting", "accuracy",
            "internal", "thought process",
            "checklist", "step-by-step", "step by step", "break it down",
            "provide a direct", "direct, conversational answer",
            "collect your thoughts",
            # New meta-commentary patterns
            "this response", "my response", "the response",
            "acknowledges", "lets them know", "informs the user",
            "appropriate response", "appropriate reply",
            "confidence level", "sentiment analysis", "sentiment is",
            "analysis shows", "analysis indicates", "results indicate",
            "the results", "tool results", "based on the results",
            "i found a confidence", "confidence of",
            "neutral in sentiment", "positive sentiment", "negative sentiment",
            "my analysis", "analysis is", "somewhat uncertain",
        ]
        
        # Patterns that indicate explaining the response rather than the response itself
        meta_explanation_patterns = [
            r"this (?:response|answer|reply) (?:acknowledges|addresses|provides|shows|indicates|suggests)",
            r"(?:the|my) (?:response|answer|reply) (?:is|was|will be)",
            r"i (?:found|detected|analyzed|observed) (?:a|the|that|no) (?:confidence|sentiment|emotion|pattern)",
            r"(?:your|the) message.+is.+(?:neutral|positive|negative) in sentiment",
            r"suggesting that (?:my|the) analysis",
            r"results (?:indicate|show|suggest) that",
        ]
        
        instruction_prefixes = (
            "avoid ",
            "keep ",
            "remember ",
            "no ",
            "stay ",
            "first action",
            "next action",
            "action plan",
            "focus ",
            "prioritize ",
            "use plain",
            "use natural",
            "double-check",
            "important:",
        )
        
        for sentence in raw_sentences:
            stripped = sentence.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            
            # Filter meta keywords by substring
            if any(keyword in lower for keyword in meta_keywords):
                continue
            
            # Filter meta explanation patterns
            skip_sentence = False
            for pattern in meta_explanation_patterns:
                if re.search(pattern, lower):
                    skip_sentence = True
                    break
            if skip_sentence:
                continue
            
            # Filter sentences starting with "okay," (case-insensitive, at sentence start)
            if re.match(r"^(okay,)\b", stripped, re.IGNORECASE):
                continue
            if re.match(r"^[A-Z\s]{3,}:", stripped):
                continue
            if lower.startswith(instruction_prefixes):
                continue
            if lower.startswith("let me ") and not lower.startswith("let me know"):
                continue
            if lower.startswith((
                "i need to ",
                "i should ",
                "i must ",
                "i'll need to ",
                "i will need to ",
                "first i need to ",
                "first, i need to ",
                "first i should ",
                "first, i should ",
                "i'll try to ",
                "i will try to ",
            )):
                continue
            filtered_sentences.append(stripped)

        if not filtered_sentences:
            pronoun_pattern = re.compile(r"\b(i|i'm|i've|i'll|we|let's|you)\b", re.IGNORECASE)
            for sentence in raw_sentences:
                stripped = sentence.strip()
                if not stripped:
                    continue
                if ":" in stripped:
                    continue
                if any(keyword in stripped.lower() for keyword in ("tool", "instruction", "system", "analysis", "sentiment", "confidence")):
                    continue
                if pronoun_pattern.search(stripped):
                    filtered_sentences.append(stripped)
                    break

        if not filtered_sentences:
            # Last resort: just clean up the original text
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
            try:
                parsed_max = int(max_tokens)
            except (TypeError, ValueError):
                logger.warning("Invalid max_tokens update %r; ignoring", max_tokens)
            else:
                if parsed_max <= 0:
                    self.max_tokens = None
                    logger.info("LLM max tokens set to unlimited")
                else:
                    self.max_tokens = max(32, parsed_max)
                    logger.info(f"LLM max tokens set to {self.max_tokens}")

                if self.max_tokens is not None:
                    self.failsafe_tokens = max(self.failsafe_tokens, self.max_tokens)

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
        if self.analytics:
            metrics_payload = self.analytics.apply_reward(delta)
            await self.bridge.post(msg_metrics(metrics_payload))

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

    def set_person_tracker(self, tracker) -> None:
        """Set the person tracker for identity recognition."""
        self.person_tracker = tracker
        logger.info("Person tracker connected to LLM")

    def get_prompts(self) -> dict[str, str]:
        return {
            "system_prompt": self.system_prompt,
            "thinking_prompt": self.thinking_prompt,
        }

    @staticmethod
    def _auto_prompt_style(model_path: str | os.PathLike[str] | None) -> str:
        """Infer a sensible prompt template based on known model naming patterns."""

        if not model_path:
            return "plain"

        path_text = str(model_path).lower()
        if "llama-3" in path_text or "llama3" in path_text:
            return "llama-3-chat"

        return "plain"

    def _apply_prompt_template(self, system_text: str, user_text: str) -> str:
        """Format ``system_text`` and ``user_text`` using the configured template."""

        template = self.prompt_style

        if template == "llama-3-chat":
            # Llama 3 chat expects explicit header tokens around each role.
            # Note: llama.cpp automatically adds <|begin_of_text|>, so we don't need to
            system_section = system_text.strip()
            user_section = user_text.strip()
            parts: list[str] = []
            if system_section:
                parts.append(
                    "".join(
                        (
                            "<|start_header_id|>system<|end_header_id|>\n",
                            f"{system_section}\n",
                            "<|eot_id|>",
                        )
                    )
                )
            parts.append(
                "".join(
                    (
                        "<|start_header_id|>user<|end_header_id|>\n",
                        f"{user_section}\n",
                        "<|eot_id|>",
                    )
                )
            )
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n")
            return "".join(parts)

        # Fallback to the legacy plain prompt used by older base and instruct models.
        sections = []
        if system_text.strip():
            sections.append(system_text.strip())
        if user_text.strip():
            sections.append(user_text.strip())
        return "\n\n".join(sections)

    def _build_prompt(self, user_input: str, context_type: str = "text", include_tools: bool = True) -> str:
        """Build prompt with context, system persona, and tool instructions."""

        context_lines = []
        if self.recent_context:
            for ctx in self.recent_context[-2:]:
                context_lines.append(f"{ctx['type']}: {ctx['content'][:80]}")
        context_summary = "\n".join(context_lines) if context_lines else "No recent context recorded."

        tools_instructions = ""
        if include_tools and self.tools_enabled:
            tools_instructions = self.tools.get_tools_prompt()
            tools_instructions += (
                "\n**IMPORTANT - Memory Storage**: When users share information about themselves (names, preferences, passwords, facts), "
                "ALWAYS use the appropriate memory tool (remember_person, remember_fact, learn_user_preference, etc.) "
                "to store it immediately. Don't just acknowledge - actually use the tool!"
                "\n\n**CRITICAL - Memory Recall**: When users ask 'what did I tell you?', 'do you remember...?', 'what was my name/password/word?', "
                "you MUST use the 'recall_entity' tool (NOT recall_recent_context). "
                "recall_entity searches ALL stored memories. recall_recent_context only sees the last few minutes. "
                "Example: User asks 'what was that word?' → use recall_entity with query='word I asked you to remember'"
            )

        final_instruction = (
            "Important: Provide a direct, conversational answer."
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

        scenario = f"{scenario}\n\n{final_instruction}"

        system_sections: list[str] = []
        if self.system_prompt:
            system_sections.append(self.system_prompt.strip())
        if self.thinking_prompt:
            system_sections.append(
                "THOUGHT PROCESS GUIDANCE:\n" + self.thinking_prompt.strip()
            )
        system_text = "\n\n".join(system_sections)

        return self._apply_prompt_template(system_text, scenario)

    async def _generate(self, prompt: str, max_tokens: int | None = None, min_tokens: int = 10, allow_tool_calls: bool = True, user_text: str | None = None) -> str:
        """Generate response from model with token streaming and GPU optimization."""
        if self.model is None:
            raise RuntimeError("LLM model not loaded")

        # Start GPU profiling
        start_time = gpu_profiler.start_inference()

        try:
            await self.bridge.post(msg_status("thinking", "Processing..."))

            # Send thinking process to UI without exposing the raw prompt
            await self.bridge.post({"type": "thought", "text": "Analyzing input and preparing response..."})

            requested_tokens: int | None = None
            if max_tokens is not None:
                try:
                    requested_tokens = max(32, int(max_tokens))
                except (TypeError, ValueError):
                    logger.warning("Invalid max_tokens override %r; ignoring", max_tokens)
                    requested_tokens = None
            elif self.max_tokens is not None:
                requested_tokens = self.max_tokens

            stream_limit = requested_tokens or self.failsafe_tokens
            failsafe_limit = self.failsafe_tokens
            stream = self.model(
                prompt,
                max_tokens=stream_limit,
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

                        # Show progress without exposing raw generation
                        if total_tokens % 10 == 0:  # Update every 10 tokens
                            await self.bridge.post({
                                "type": "thought",
                                "text": f"Processing... ({total_tokens} tokens)",
                            })

                        if total_tokens >= failsafe_limit:
                            logger.warning(
                                "Reached failsafe token limit (%s tokens), stopping generation",
                                failsafe_limit
                            )
                            await self.bridge.post(
                                msg_event(
                                    f"generation stopped after {failsafe_limit} tokens to prevent runaway output"
                                )
                            )
                            break

            # Build full response from tokens
            raw_response = "".join(tokens).strip()
            
            # Show the raw thinking process (before sanitization)
            await self.bridge.post({
                "type": "thought",
                "text": f"Raw output: {raw_response[:150]}{'...' if len(raw_response) > 150 else ''}",
            })

            # Remove any role-play artifacts
            raw_response = raw_response.replace("You:", "").replace("Person:", "").strip()

            # Process tool calls if enabled
            if allow_tool_calls and self.tools_enabled and "TOOL_CALL:" in raw_response:
                tool_calls = self.tools.parse_tool_calls(raw_response)
                if tool_calls:
                    logger.info(f"Found {len(tool_calls)} tool call(s) in response")
                    await self.bridge.post({"type": "thought", "text": f"Executing {len(tool_calls)} tool(s)..."})
                    
                    tool_results = []
                    for call in tool_calls:
                        tool_name = call.get("tool")
                        args = call.get("args", {})
                        execution = await self.tools.execute_tool(tool_name, args)
                        tool_results.append(execution)

                        # Send tool result to UI with full metadata
                        await self.bridge.post({
                            "type": "tool_result",
                            **execution
                        })
                    
                    # Generate follow-up response incorporating tool results
                    await self.bridge.post({"type": "thought", "text": "Processing tool results..."})
                    
                    # Build concise summary of tool results
                    tool_summaries = []
                    for result in tool_results:
                        tool_name = result.get("tool", "unknown")
                        if result.get("success"):
                            result_data = result.get("result", {})
                            
                            # Summarize based on tool type
                            if tool_name == "recall_entity":
                                found = result_data.get("found", 0)
                                results_list = result_data.get("results", [])
                                if found > 0:
                                    items = []
                                    for item in results_list[:5]:  # Limit to 5 items
                                        label = item.get("label", "Unknown")
                                        kind = item.get("kind", "item")
                                        desc = item.get("description", "")[:100]  # Truncate description
                                        items.append(f"- {label} ({kind}): {desc}")
                                    summary = f"Found {found} items:\n" + "\n".join(items)
                                else:
                                    summary = "No matching items found"
                            else:
                                # Generic summary for other tools
                                summary = json.dumps(result_data, indent=2)[:500]  # Limit to 500 chars
                            
                            tool_summaries.append(f"{tool_name}: {summary}")
                        else:
                            error_msg = result.get("error", "Unknown error")
                            tool_summaries.append(f"{tool_name}: Error - {error_msg}")
                    
                    tool_context = "\n\n".join(tool_summaries)
                    
                    # Build a compact follow-up prompt with just the essentials
                    user_question = user_text if user_text else "your question"
                    
                    followup_prompt = (
                        f"User asked: {user_question}\n\n"
                        f"Tool Results:\n{tool_context}\n\n"
                        f"Based on these results, provide a brief, natural response to the user. "
                        f"Be specific about what you found."
                    )
                    
                    # Generate new response with tool results (disable tool calls to prevent recursion)
                    logger.info("Generating follow-up response with tool results")
                    logger.info(f"Follow-up prompt length: {len(followup_prompt)} chars")
                    return await self._generate(followup_prompt, max_tokens=requested_tokens or 150, min_tokens=min_tokens, allow_tool_calls=False, user_text=user_text)
                    
            # Remove any remaining tool calls from response for speaking
            if "TOOL_CALL:" in raw_response:
                response_lines = []
                for line in raw_response.split('\n'):
                    if 'TOOL_CALL:' not in line:
                        response_lines.append(line)
                raw_response = '\n'.join(response_lines).strip()

            # Sanitize for final output (this is what gets spoken)
            final_response = self._sanitize_response(raw_response)
            if not final_response:
                final_response = raw_response  # Fallback to raw if sanitization removes everything

            if len(final_response) < 3 and total_tokens < min_tokens:
                logger.warning(f"Response too short ({len(final_response)} chars, {total_tokens} tokens), regenerating...")
                # Ensure failsafe_tokens is always used if both requested_tokens and self.max_tokens are None
                if requested_tokens is None and self.max_tokens is None:
                    return await self._generate(
                        prompt,
                        max_tokens=self.failsafe_tokens,
                        min_tokens=0
                    )
                else:
                    return await self._generate(
                        prompt,
                        max_tokens=requested_tokens or self.max_tokens,
                        min_tokens=0
                    )

            logger.info(f"Generated ({total_tokens} tokens): {final_response[:100]}")
            
            # Collect and log GPU metrics
            gpu_metrics = gpu_profiler.end_inference(start_time)
            gpu_profiler.log_metrics(gpu_metrics)
            
            # Send GPU performance metrics to UI
            if gpu_metrics.memory_allocated_mb > 0:
                await self.bridge.post({
                    "type": "thought",
                    "text": f"GPU: {gpu_metrics.memory_allocated_mb:.0f}MB used, {gpu_metrics.inference_time_ms:.0f}ms",
                })
            
            # Send final thought showing what will be spoken
            await self.bridge.post({
                "type": "thought",
                "text": f"Final response ready: {final_response[:100]}{'...' if len(final_response) > 100 else ''}",
            })

            # Clean up GPU memory after generation
            gpu_profiler.optimize_memory()

            await self.bridge.post(msg_status("idle", "Ready"))
            return final_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            await self.bridge.post(msg_event(f"llm error: {str(e)}"))
            await self.bridge.post(msg_status("idle", "Error"))
            
            # Clean up GPU memory even on error
            gpu_profiler.optimize_memory()
            
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

            if self.analytics:
                self.analytics.observe_user_message(user_text)

            prompt = self._build_prompt(user_text, "text")
            response = await self._generate(prompt, user_text=user_text)

            if response:
                self._add_context("assistant", response)
                metrics_payload = None
                if self.analytics:
                    metrics_payload = self.analytics.observe_model_response(response)

                yield msg_speak(response)
                await self.tts.speak(response)
                if metrics_payload:
                    await self.bridge.post(msg_metrics(metrics_payload))
                if self.memory:
                    await self.memory.record_interaction("text", user_text, response)
                
                # Bind conversation to current speaker for person tracking (same as audio)
                if self.person_tracker:
                    try:
                        speaker = await self.person_tracker.get_current_speaker()
                        if speaker:
                            # Extract simple topics from the conversation
                            topics = self._extract_topics(user_text)
                            await self.person_tracker.bind_conversation(
                                speaker["person_id"],
                                user_text,
                                response,
                                topics
                            )
                            
                            # Check if user mentioned their name
                            name_match = self._extract_name_from_text(user_text)
                            if name_match:
                                await self.person_tracker.set_person_name(
                                    speaker["person_id"],
                                    name_match
                                )
                                logger.info(f"Learned speaker's name from chat: {name_match}")
                    except Exception as e:
                        logger.debug(f"Could not bind chat conversation: {e}")
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

            if self.analytics:
                self.analytics.observe_user_message(transcribed_text)

            prompt = self._build_prompt(transcribed_text, "audio")
            response = await self._generate(prompt)

            if response:
                self._add_context("assistant", response)
                await self.bridge.post(msg_speak(response))
                metrics_payload = None
                if self.analytics:
                    metrics_payload = self.analytics.observe_model_response(response)

                await self.tts.speak(response)
                if metrics_payload:
                    await self.bridge.post(msg_metrics(metrics_payload))
                if self.memory:
                    await self.memory.record_interaction("audio", transcribed_text, response)
                
                # Bind conversation to current speaker for person tracking
                if self.person_tracker:
                    try:
                        speaker = await self.person_tracker.get_current_speaker()
                        if speaker:
                            # Extract simple topics from the conversation
                            topics = self._extract_topics(transcribed_text)
                            await self.person_tracker.bind_conversation(
                                speaker["person_id"],
                                transcribed_text,
                                response,
                                topics
                            )
                            
                            # Check if user mentioned their name
                            name_match = self._extract_name_from_text(transcribed_text)
                            if name_match:
                                await self.person_tracker.set_person_name(
                                    speaker["person_id"],
                                    name_match
                                )
                                logger.info(f"Learned speaker's name: {name_match}")
                    except Exception as e:
                        logger.debug(f"Could not bind conversation: {e}")
                
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
            
            # Normalize for comparison
            normalized_description = description.strip().lower()
            
            # Check if this is essentially the same observation as before
            if self._last_spoken_vision:
                last_text, last_time = self._last_spoken_vision
                # More aggressive duplicate detection - check semantic similarity
                is_duplicate = self._is_similar_vision(normalized_description, last_text)
                if is_duplicate and now - last_time < max(self.vision_cooldown * 6, 90):
                    # Log silently but don't announce - suppress for longer (90 seconds min)
                    logger.debug("Vision (suppressed duplicate): %s", description[:80])
                    self._add_context("vision_quiet", description)
                    return
            
            # Check for meaningful changes worth commenting on
            # Only speak if something NEW and INTERESTING happens
            interesting_changes = [
                'gesture', 'waving', 'pointing', 'thumbs', 'peace',
                'left', 'entered', 'appeared', 'disappeared', 'returned',
                'different', 'new', 'changed', 'multiple'
            ]
            
            has_interesting_change = any(
                word in normalized_description 
                for word in interesting_changes
            )
            
            # Check if we've seen this person count/environment combo recently
            static_patterns = [
                'i see 1 person', 'i see one person',
                'the space looks', 'the scene looks',
                'moderately lit', 'dimly lit', 'well lit', 'dark'
            ]
            is_static_observation = all(
                pattern not in normalized_description or pattern in (self._last_spoken_vision[0] if self._last_spoken_vision else "")
                for pattern in static_patterns
                if pattern in normalized_description
            )
            
            # Decision: Should we comment on this?
            should_speak = has_interesting_change and not is_static_observation
            
            # Random quiet periods even for interesting things (30% quiet)
            import random
            if not should_speak or random.random() < 0.3:
                logger.info(f"Vision (quiet): {description[:80]}")
                self._add_context("vision_quiet", description)
                # Send as thought but don't speak
                await self.bridge.post({"type": "thought", "text": f"Observing: {description}"})
                return

            logger.info(f"Vision (speaking): {description[:100]}")
            self._add_context("vision", description)
            
            # Get person context if available
            person_context = ""
            if self.person_tracker:
                try:
                    present = await self.person_tracker.get_all_present()
                    if present:
                        person_parts = []
                        for p in present:
                            name = p.get("name") or p.get("display_name", "someone")
                            familiarity = p.get("familiarity", 0)
                            if familiarity > 0.5:
                                person_parts.append(f"{name} (familiar)")
                            elif familiarity > 0.2:
                                person_parts.append(f"{name} (seen before)")
                            else:
                                person_parts.append(name)
                        person_context = f"\nPeople present: {', '.join(person_parts)}"
                        
                        # Check if we should suggest asking for a name
                        for p in present:
                            if p.get("should_ask_name"):
                                person_context += f"\n(Consider asking {p.get('display_name', 'this person')} their name)"
                except Exception as e:
                    logger.debug(f"Could not get person context: {e}")
            
            # Build a prompt that encourages novel observations
            vision_prompt = (
                f"Visual observation: {description}{person_context}\n\n"
                "If this is something NEW or a CHANGE from before, comment on it briefly. "
                "If you recognize someone, address them by name. "
                "If it's the same as what you've been seeing, stay quiet or make a very brief acknowledgment. "
                "Don't just repeat 'I see X person in Y environment' - be more insightful or stay silent."
            )

            prompt = self._build_prompt(vision_prompt, "vision")
            response = await self._generate(prompt, max_tokens=60)

            if response:
                # Filter out generic responses
                generic_phrases = [
                    "i see one person", "i see 1 person", 
                    "in a dark environment", "in a moderately lit",
                    "the scene looks", "the space looks"
                ]
                response_lower = response.lower()
                is_generic = any(phrase in response_lower for phrase in generic_phrases)
                
                if is_generic and len(response) < 50:
                    # Don't speak generic responses
                    logger.info(f"Vision response suppressed (generic): {response}")
                    return
                
                self._add_context("assistant", response)
                await self.bridge.post(msg_speak(response))
                await self.tts.speak(response)
                self._last_spoken_vision = (normalized_description, now)
                if self.memory:
                    await self.memory.record_interaction("vision", description, response, tags=["vision"])
                
        finally:
            self._processing = False

    def _is_similar_vision(self, new_text: str, old_text: str) -> bool:
        """Check if two vision descriptions are semantically similar."""
        # Normalize texts
        new_text = new_text.lower().strip()
        old_text = old_text.lower().strip()
        
        # Exact match check first
        if new_text == old_text:
            return True
        
        # Static pattern detection - these patterns indicate nothing interesting changed
        static_patterns = [
            'i see person on the',
            'i see 1 person',
            'i see one person',
            'the space looks',
            'the scene looks',
            'moderately lit environment',
            'dimly lit environment',
            'well lit environment',
        ]
        
        # If both texts contain the same static pattern, consider them duplicates
        for pattern in static_patterns:
            if pattern in new_text and pattern in old_text:
                return True
        
        # Word overlap check
        new_words = set(new_text.split())
        old_words = set(old_text.split())
        
        if not new_words or not old_words:
            return False
            
        overlap = len(new_words & old_words)
        max_len = max(len(new_words), len(old_words))
        
        # If >60% word overlap, consider it similar (lowered from 70%)
        return (overlap / max_len) > 0.6 if max_len > 0 else False

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
                await self.bridge.post({"type": "thought", "text": response})
                # Only record substantial autonomous thoughts to memory (>15 chars)
                # Skip recording trivial outputs or silence decisions
                if self.memory and len(response) > 15:
                    await self.memory.record_interaction("autonomous", "internal_reflection", response)
            else:
                logger.info("Autonomous thought: no output")

        finally:
            self._processing = False

    def _extract_topics(self, text: str) -> List[str]:
        """Extract simple topics from conversation text."""
        topics = []
        text_lower = text.lower()
        
        # Common topic indicators
        topic_keywords = {
            "weather": ["weather", "rain", "sunny", "cold", "hot", "temperature"],
            "work": ["work", "job", "office", "meeting", "project", "deadline"],
            "technology": ["computer", "phone", "software", "code", "programming", "ai"],
            "food": ["food", "eat", "hungry", "lunch", "dinner", "breakfast"],
            "music": ["music", "song", "listen", "band", "artist"],
            "movies": ["movie", "film", "watch", "show", "series"],
            "health": ["health", "sick", "doctor", "exercise", "sleep"],
            "family": ["family", "kids", "children", "wife", "husband", "parents"],
            "hobbies": ["hobby", "game", "play", "read", "book"],
        }
        
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                topics.append(topic)
        
        return topics[:3]  # Max 3 topics

    def _extract_name_from_text(self, text: str) -> Optional[str]:
        """Try to extract a name if the user introduces themselves."""
        import re
        
        # Common patterns for self-introduction
        patterns = [
            r"(?:my name is|i'm|i am|call me|they call me|name's)\s+([A-Z][a-z]+)",
            r"(?:this is|it's)\s+([A-Z][a-z]+)(?:\s+speaking)?",
            r"^([A-Z][a-z]+)\s+here\b",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1)
                # Filter out common non-names
                non_names = {"the", "a", "an", "my", "your", "this", "that", "here", "there"}
                if name.lower() not in non_names and len(name) > 1:
                    return name.capitalize()
        
        return None
