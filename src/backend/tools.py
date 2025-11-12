# Title: LLM Tools - Function calling capabilities for autonomous AI
# Path: backend/tools.py
# Purpose: Provide LLM with tools for memory, learning, observation, and self-improvement

"""
Tool definitions and execution framework for LLM function calling.
Enables the model to interact with memory, track learning, make observations,
and manage developmental milestones.
"""

import asyncio
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Configuration constants
SENTIMENT_POSITIVE_WORDS = ["good", "great", "happy", "love", "excellent", "wonderful", "thanks", "thank you", "amazing", "fantastic"]
SENTIMENT_NEGATIVE_WORDS = ["bad", "sad", "angry", "hate", "terrible", "awful", "annoyed", "frustrated", "disappointed", "upset"]

APPROPRIATENESS_MAX_LENGTH = 500
APPROPRIATENESS_MIN_LENGTH = 5
APPROPRIATENESS_MAX_EXCLAMATION = 5


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None


@dataclass
class Tool:
    """Definition of a tool that the LLM can use."""

    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable
    category: str = "general"  # memory, learning, observation, social, analytics, etc.
    display_name: Optional[str] = None
    return_description: Optional[str] = None


class ToolExecutor:
    """Executes tools called by the LLM and manages tool registry."""
    
    def __init__(self, llm_instance):
        """Initialize with reference to LLM instance for context access."""
        self.llm = llm_instance
        self.tools: Dict[str, Tool] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.max_history = 100
        
        # Register built-in tools
        self._register_builtin_tools()
        
    def _register_builtin_tools(self):
        """Register all built-in tools."""
        # Memory tools
        self.register_tool(Tool(
            name="search_memory",
            display_name="Search Memory",
            description="Search through past memories and interactions to recall relevant information. Use this when you need to remember previous conversations, observations, or learnings.",
            parameters=[
                ToolParameter("query", "string", "What to search for in memory", required=True),
                ToolParameter("limit", "number", "Maximum number of results to return", required=False, default=5),
                ToolParameter("modality", "string", "Filter by interaction type", required=False,
                            enum=["text", "audio", "vision", "autonomous", "all"])
            ],
            function=self._search_memory,
            category="memory",
            return_description="List of matching memory nodes ordered by relevance."
        ))

        self.register_tool(Tool(
            name="recall_recent_context",
            display_name="Recall Recent Context",
            description="Retrieve the most recent context and interactions. Use this to maintain conversation continuity and understand what just happened.",
            parameters=[
                ToolParameter("count", "number", "Number of recent items to recall", required=False, default=5)
            ],
            function=self._recall_recent_context,
            category="memory",
            return_description="Window of the short-term context buffer."
        ))

        self.register_tool(Tool(
            name="summarize_recent_context",
            display_name="Summarize Recent Context",
            description="Generate a concise summary of the latest interactions for quick situational awareness.",
            parameters=[
                ToolParameter("count", "number", "How many recent items to include", required=False, default=5),
                ToolParameter("include_types", "array", "Limit to these context types", required=False)
            ],
            function=self._summarize_recent_context,
            category="memory",
            return_description="Summary string with the included context items."
        ))

        # Observation tools
        self.register_tool(Tool(
            name="record_observation",
            display_name="Record Observation",
            description="Record an important observation or insight about the environment, user behavior, or patterns you notice. This helps build understanding over time.",
            parameters=[
                ToolParameter("observation", "string", "What you observed", required=True),
                ToolParameter("category", "string", "Type of observation", required=True,
                            enum=["user_preference", "pattern", "behavior", "environment", "insight"]),
                ToolParameter("importance", "number", "How important (1-10)", required=False, default=5),
                ToolParameter("tags", "array", "Tags for categorization", required=False)
            ],
            function=self._record_observation,
            category="observation",
            return_description="Confirmation that the observation was persisted along with metadata."
        ))

        # Learning tools
        self.register_tool(Tool(
            name="evaluate_interaction",
            display_name="Evaluate Interaction",
            description="Evaluate how well you handled the last interaction. Use this for self-improvement by analyzing what worked and what didn't.",
            parameters=[
                ToolParameter("quality_score", "number", "Rate your response quality (1-10)", required=True),
                ToolParameter("what_worked", "string", "What you did well", required=False),
                ToolParameter("what_to_improve", "string", "What could be better", required=False)
            ],
            function=self._evaluate_interaction,
            category="learning",
            return_description="Structured reflection including strengths and improvement areas."
        ))

        self.register_tool(Tool(
            name="identify_pattern",
            display_name="Identify Pattern",
            description="Identify and record a pattern you've noticed in user behavior, interactions, or your environment. Helps develop social understanding.",
            parameters=[
                ToolParameter("pattern", "string", "Description of the pattern", required=True),
                ToolParameter("occurrences", "number", "How many times observed", required=False, default=1),
                ToolParameter("confidence", "number", "How confident (0.0-1.0)", required=False, default=0.7)
            ],
            function=self._identify_pattern,
            category="learning",
            return_description="Confirmation and metadata for the captured pattern."
        ))

        # Social behavior tools
        self.register_tool(Tool(
            name="analyze_sentiment",
            display_name="Analyze Sentiment",
            description="Analyze the sentiment or emotional tone of recent interactions. Use this to understand how people are feeling and respond appropriately.",
            parameters=[
                ToolParameter("context", "string", "What to analyze (leave empty for recent)", required=False)
            ],
            function=self._analyze_sentiment,
            category="social",
            return_description="Sentiment classification with confidence and supporting indicators."
        ))

        self.register_tool(Tool(
            name="check_appropriate_response",
            display_name="Check Response Appropriateness",
            description="Before responding, check if a response is appropriate given the context and social norms. Helps avoid mistakes.",
            parameters=[
                ToolParameter("proposed_response", "string", "What you're thinking of saying", required=True),
                ToolParameter("context", "string", "Current situation context", required=False)
            ],
            function=self._check_appropriate_response,
            category="social",
            return_description="Validation outcome with any detected issues."
        ))

        # Development/milestone tools
        self.register_tool(Tool(
            name="track_milestone",
            display_name="Track Milestone",
            description="Record a developmental milestone or achievement. Track your growth and capabilities over time.",
            parameters=[
                ToolParameter("milestone", "string", "What was achieved", required=True),
                ToolParameter("category", "string", "Type of development", required=True,
                            enum=["communication", "understanding", "capability", "social", "technical"]),
                ToolParameter("notes", "string", "Additional context", required=False)
            ],
            function=self._track_milestone,
            category="learning",
            return_description="Persisted milestone entry with categorisation."
        ))

        self.register_tool(Tool(
            name="get_current_capabilities",
            display_name="Get Current Capabilities",
            description="Review your current capabilities and tracked milestones. Use this to understand what you can do and what you've learned.",
            parameters=[],
            function=self._get_current_capabilities,
            category="learning",
            return_description="Overview of tools, milestones, and recent strengths."
        ))

        # General/system tools
        self.register_tool(Tool(
            name="list_available_tools",
            display_name="List Available Tools",
            description="Return metadata for every available tool including parameters and categories.",
            parameters=[],
            function=self._list_available_tools,
            category="general",
            return_description="Comprehensive tool registry snapshot."
        ))

        self.register_tool(Tool(
            name="get_tool_usage_stats",
            display_name="Get Tool Usage Stats",
            description="Summarize historical tool usage to understand what capabilities are being exercised.",
            parameters=[
                ToolParameter("lookback", "number", "Only include the latest N executions", required=False)
            ],
            function=self._get_tool_usage_stats,
            category="analytics",
            return_description="Aggregated counters, success rate, and recent executions."
        ))

        categories = sorted({tool.category for tool in self.tools.values()})
        logger.info(
            "Registered %s built-in tools across categories: %s",
            len(self.tools),
            ", ".join(categories)
        )
    
    def register_tool(self, tool: Tool):
        """Register a tool for LLM use."""
        if not tool.display_name:
            tool.display_name = tool.name.replace("_", " ").title()

        existing = self.tools.get(tool.name)
        if existing:
            logger.warning("Overwriting existing tool definition for %s", tool.name)

        self.tools[tool.name] = tool
        logger.debug("Registered tool: %s (%s)", tool.name, tool.category)
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible function calling schema."""
        schema = []
        for tool in self.tools.values():
            params = {}
            required = []
            
            for param in tool.parameters:
                param_def = {
                    "type": param.type,
                    "description": param.description
                }
                if param.enum:
                    param_def["enum"] = param.enum
                if param.default is not None:
                    param_def["default"] = param.default
                    
                params[param.name] = param_def
                if param.required:
                    required.append(param.name)
            
            description = tool.description
            if tool.return_description:
                description = f"{description} Returns: {tool.return_description}"

            schema.append({
                "name": tool.name,
                "description": f"[{tool.category.title()}] {description}",
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": required
                }
            })
        
        return schema
    
    def get_tools_prompt(self) -> str:
        """Generate a prompt describing available tools."""
        tools_by_category = {}
        for tool in self.tools.values():
            if tool.category not in tools_by_category:
                tools_by_category[tool.category] = []
            tools_by_category[tool.category].append(tool)
        
        prompt = "You have access to the following tools:\n\n"
        
        for category, tools in sorted(tools_by_category.items()):
            prompt += f"## {category.title()} Tools\n"
            for tool in tools:
                display_name = tool.display_name or tool.name.replace("_", " ").title()
                prompt += f"- **{display_name}** ({tool.name}): {tool.description}\n"
                if tool.return_description:
                    prompt += f"  - returns: {tool.return_description}\n"
                for param in tool.parameters:
                    req = "required" if param.required else "optional"
                    prompt += f"  - {param.name} ({param.type}, {req}): {param.description}\n"
            prompt += "\n"

        prompt += "To use a tool, include in your response: TOOL_CALL: {\"tool\": \"tool_name\", \"args\": {\"param\": \"value\"}}\n"
        return prompt
    
    async def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return a normalized execution payload."""

        timestamp = time.time()
        tool = self.tools.get(tool_name)

        if not tool:
            result_payload = {
                "tool": tool_name,
                "display_name": tool_name.replace("_", " ").title(),
                "category": "general",
                "description": "Unknown tool",
                "timestamp": int(timestamp * 1000),
                "duration_ms": 0.0,
                "args": args or {},
                "warnings": [],
                "result": {"error": f"Unknown tool: {tool_name}"},
                "success": False,
                "summary": f"Unknown tool requested: {tool_name}"
            }
            self._record_execution(result_payload)
            return result_payload

        try:
            validated_args, warnings = self._validate_arguments(tool, args)
        except ValueError as exc:
            error_payload = {
                "tool": tool.name,
                "display_name": tool.display_name,
                "category": tool.category,
                "description": tool.description,
                "timestamp": int(timestamp * 1000),
                "duration_ms": 0.0,
                "args": args or {},
                "warnings": [],
                "result": {"error": str(exc)},
                "success": False,
                "summary": str(exc)
            }
            self._record_execution(error_payload)
            return error_payload

        logger.info("Executing tool: %s with args: %s", tool_name, validated_args)

        try:
            start = time.perf_counter()
            raw_result = tool.function(**validated_args)
            if asyncio.iscoroutine(raw_result):
                raw_result = await raw_result
            duration_ms = (time.perf_counter() - start) * 1000

            normalized_result = self._normalize_result(raw_result)
            success = self._determine_success(normalized_result)
            summary = self._summarize_execution(tool, normalized_result)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("Tool execution failed: %s", exc, exc_info=True)
            normalized_result = {"error": str(exc)}
            success = False
            summary = f"Execution error: {str(exc)}"

        payload = {
            "tool": tool.name,
            "display_name": tool.display_name,
            "category": tool.category,
            "description": tool.description,
            "timestamp": int(timestamp * 1000),
            "duration_ms": round(duration_ms, 2),
            "args": validated_args,
            "warnings": warnings,
            "result": normalized_result,
            "success": success,
            "summary": summary
        }

        if success and "success" not in normalized_result:
            payload["result"]["success"] = True

        if not success and "success" not in normalized_result:
            payload["result"]["success"] = False

        self._record_execution(payload)
        return payload

    def _record_execution(self, execution: Dict[str, Any]) -> None:
        """Persist execution details to bounded history for analytics tools."""

        self.execution_history.append(execution)
        if len(self.execution_history) > self.max_history:
            self.execution_history.pop(0)

    def _validate_arguments(self, tool: Tool, args: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
        """Validate and normalize tool arguments."""

        validated: Dict[str, Any] = {}
        warnings: List[str] = []
        provided_keys = set(args.keys()) if args else set()
        expected_keys = {param.name for param in tool.parameters}

        for param in tool.parameters:
            if args and param.name in args:
                value = self._coerce_argument(param, args[param.name])
                if param.enum and value not in param.enum:
                    raise ValueError(
                        f"Invalid value '{value}' for parameter '{param.name}'. Must be one of: {', '.join(str(v) for v in param.enum)}"
                    )
                validated[param.name] = value
            elif param.required:
                raise ValueError(f"Missing required parameter: {param.name}")
            elif param.default is not None:
                validated[param.name] = param.default

        unexpected = provided_keys - expected_keys
        if unexpected:
            warnings.append(f"Ignored unexpected parameters: {', '.join(sorted(unexpected))}")

        return validated, warnings

    def _coerce_argument(self, param: ToolParameter, value: Any) -> Any:
        """Coerce tool arguments into the expected runtime type."""

        if param.type == "number":
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str) and value.strip():
                try:
                    return float(value)
                except ValueError as exc:
                    raise ValueError(f"Parameter '{param.name}' must be a number") from exc
            raise ValueError(f"Parameter '{param.name}' must be a number")

        if param.type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "y"}:
                    return True
                if lowered in {"false", "0", "no", "n"}:
                    return False
            raise ValueError(f"Parameter '{param.name}' must be boolean")

        if param.type == "array":
            if isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            if isinstance(value, str):
                if not value.strip():
                    return []
                return [item.strip() for item in value.split(",") if item.strip()]
            raise ValueError(f"Parameter '{param.name}' must be an array")

        if param.type == "object":
            if isinstance(value, dict):
                return value
            raise ValueError(f"Parameter '{param.name}' must be an object")

        # Default to string handling
        if value is None:
            return ""
        return str(value)

    def _normalize_result(self, raw_result: Any) -> Dict[str, Any]:
        """Ensure tool results are serializable dictionaries."""

        if raw_result is None:
            return {}
        if isinstance(raw_result, dict):
            return dict(raw_result)
        if isinstance(raw_result, list):
            return {"items": raw_result, "count": len(raw_result)}
        if isinstance(raw_result, (str, int, float, bool)):
            return {"value": raw_result}

        try:
            return json.loads(json.dumps(raw_result))  # Attempt to coerce serializable objects
        except (TypeError, ValueError):
            return {"repr": repr(raw_result)}

    def _determine_success(self, result: Dict[str, Any]) -> bool:
        """Derive a success flag from the normalized tool result."""

        if not isinstance(result, dict):
            return False
        if "success" in result:
            return bool(result["success"])
        if "error" in result:
            return False
        return True

    def _summarize_execution(self, tool: Tool, result: Dict[str, Any]) -> str:
        """Create a human-friendly summary of the execution result."""

        if "error" in result:
            return f"Error: {result['error']}"

        name = tool.name
        if name == "search_memory":
            found = result.get("found") or len(result.get("results", []))
            return f"Found {found} matching memory entries"
        if name == "recall_recent_context":
            return f"Retrieved {result.get('count', 0)} recent context item(s)"
        if name == "summarize_recent_context":
            return result.get("summary", "Summarized recent context")
        if name == "record_observation":
            return result.get("message", "Observation recorded")
        if name == "evaluate_interaction":
            score = result.get("quality_score")
            return f"Self-evaluated quality {score}/10" if score is not None else "Interaction evaluated"
        if name == "identify_pattern":
            return f"Pattern noted: {result.get('pattern', 'unknown')}"
        if name == "track_milestone":
            return f"Milestone captured: {result.get('milestone', 'unspecified')}"
        if name == "get_current_capabilities":
            tools_available = result.get("tools_available", 0)
            milestones = result.get("milestones_achieved", 0)
            return f"Capabilities snapshot: {tools_available} tools, {milestones} milestones"
        if name == "analyze_sentiment":
            sentiment = result.get("sentiment", "neutral")
            confidence = result.get("confidence")
            if confidence is not None:
                return f"Sentiment {sentiment} ({round(confidence * 100)}%)"
            return f"Sentiment {sentiment}"
        if name == "check_appropriate_response":
            appropriate = result.get("appropriate")
            return "Response appropriate" if appropriate else "Response needs revision"
        if name == "list_available_tools":
            return f"Enumerated {result.get('total', 0)} tools"
        if name == "get_tool_usage_stats":
            return "Usage stats generated"

        value = result.get("message") or result.get("value")
        if value:
            return str(value)
        return "Tool execution completed"
    
    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM output."""
        decoder = json.JSONDecoder()
        tool_calls: List[Dict[str, Any]] = []
        search_from = 0

        while True:
            marker_index = text.find("TOOL_CALL:", search_from)
            if marker_index == -1:
                break

            brace_index = text.find("{", marker_index)
            if brace_index == -1:
                break

            try:
                call_data, offset = decoder.raw_decode(text[brace_index:])
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse tool call JSON: %s", exc)
                search_from = brace_index + 1
                continue

            if isinstance(call_data, dict) and call_data.get("tool"):
                tool_calls.append(call_data)
            else:
                logger.debug("Ignoring malformed tool call payload: %s", call_data)

            search_from = brace_index + offset

        return tool_calls
    
    # Tool implementation methods
    async def _search_memory(self, query: str, limit: int = 5, modality: str = "all") -> Dict[str, Any]:
        """Search memory for relevant past interactions."""
        if not self.llm.memory or not self.llm.memory.enabled:
            return {"found": 0, "results": [], "message": "Memory system not available"}
        
        try:
            # Get nodes and edges from memory
            nodes, edges = await self.llm.memory.load_graph()
            
            # Simple text search through nodes
            query_lower = query.lower()
            results = []
            
            for node in nodes:
                # Filter by modality if specified
                if modality != "all" and node.get("source") != modality:
                    continue
                
                # Search in label and description
                label = (node.get("label") or "").lower()
                desc = (node.get("description") or "").lower()
                
                if query_lower in label or query_lower in desc:
                    results.append({
                        "type": node.get("kind"),
                        "content": node.get("description") or node.get("label"),
                        "source": node.get("source"),
                        "timestamp": node.get("timestamp"),
                        "relevance": 1.0 if query_lower in label else 0.5
                    })
            
            # Sort by relevance and limit
            results.sort(key=lambda x: x["relevance"], reverse=True)
            results = results[:limit]
            
            return {
                "found": len(results),
                "results": results,
                "query": query
            }
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return {"found": 0, "results": [], "error": str(e)}
    
    async def _recall_recent_context(self, count: int = 5) -> Dict[str, Any]:
        """Recall recent context from LLM's short-term memory."""
        recent = self.llm.recent_context[-count:] if len(self.llm.recent_context) > 0 else []

        return {
            "count": len(recent),
            "context": [
                {
                    "type": ctx["type"],
                    "content": ctx["content"][:200],  # Truncate long content
                    "timestamp": ctx.get("timestamp")
                }
                for ctx in recent
            ]
        }

    async def _summarize_recent_context(self, count: int = 5,
                                        include_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Provide a summary of the most recent context entries."""

        if not getattr(self.llm, "recent_context", None):
            return {
                "success": True,
                "summary": "No recent context captured yet.",
                "items": [],
                "count": 0
            }

        if include_types:
            include_set = {str(item).lower() for item in include_types}
            filtered = [
                ctx for ctx in self.llm.recent_context
                if str(ctx.get("type", "")).lower() in include_set
            ]
        else:
            filtered = list(self.llm.recent_context)

        if not filtered:
            return {
                "success": True,
                "summary": "No context entries match the requested filters.",
                "items": [],
                "count": 0
            }

        window = max(1, int(count))
        recent_entries = filtered[-window:]

        formatted_items = []
        summary_fragments = []
        for ctx in recent_entries:
            content = str(ctx.get("content", ""))
            truncated = content if len(content) <= 160 else f"{content[:157]}..."
            summary_fragments.append(f"{ctx.get('type', 'context')}: {truncated}")
            formatted_items.append({
                "type": ctx.get("type"),
                "content": content,
                "timestamp": ctx.get("timestamp")
            })

        summary = " | ".join(summary_fragments)

        return {
            "success": True,
            "summary": summary,
            "items": formatted_items,
            "count": len(formatted_items)
        }

    async def _record_observation(self, observation: str, category: str, 
                                  importance: int = 5, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Record an observation."""
        if not self.llm.memory or not self.llm.memory.enabled:
            return {"success": False, "message": "Memory system not available"}
        
        tags_list = tags or []
        tags_list.append(f"importance_{importance}")
        tags_list.append(f"category_{category}")
        
        try:
            await self.llm.memory.record_interaction(
                modality="observation",
                stimulus=f"[{category.upper()}] {observation}",
                response=f"Recorded observation (importance: {importance}/10)",
                tags=tags_list
            )
            
            return {
                "success": True,
                "message": f"Observation recorded: {observation[:50]}...",
                "category": category,
                "importance": importance
            }
        except Exception as e:
            logger.error(f"Failed to record observation: {e}")
            return {"success": False, "error": str(e)}
    
    async def _evaluate_interaction(self, quality_score: int, 
                                   what_worked: Optional[str] = None,
                                   what_to_improve: Optional[str] = None) -> Dict[str, Any]:
        """Self-evaluate an interaction."""
        evaluation = {
            "quality_score": quality_score,
            "what_worked": what_worked or "Not specified",
            "what_to_improve": what_to_improve or "Not specified",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Record as observation
        if self.llm.memory and self.llm.memory.enabled:
            try:
                eval_text = f"Self-evaluation: Quality {quality_score}/10. Worked: {what_worked or 'N/A'}. Improve: {what_to_improve or 'N/A'}"
                await self.llm.memory.record_interaction(
                    modality="self_evaluation",
                    stimulus="Internal reflection",
                    response=eval_text,
                    tags=["self_improvement", f"quality_{quality_score}"]
                )
            except Exception as e:
                logger.error(f"Failed to record evaluation: {e}")
        
        # Apply reinforcement based on score
        if self.llm:
            delta = (quality_score - 5) * 2  # Map 1-10 to -8 to +10
            await self.llm.reinforce(delta)
        
        return evaluation
    
    async def _identify_pattern(self, pattern: str, occurrences: int = 1, 
                               confidence: float = 0.7) -> Dict[str, Any]:
        """Record an identified pattern."""
        if not self.llm.memory or not self.llm.memory.enabled:
            return {"success": False, "message": "Memory system not available"}
        
        try:
            await self.llm.memory.record_interaction(
                modality="pattern_recognition",
                stimulus=f"Pattern identified: {pattern}",
                response=f"Observed {occurrences} time(s)",
                confidence=confidence,
                tags=["pattern", f"occurrences_{occurrences}"]
            )
            
            return {
                "success": True,
                "pattern": pattern,
                "occurrences": occurrences,
                "confidence": confidence,
                "message": "Pattern recorded for future reference"
            }
        except Exception as e:
            logger.error(f"Failed to record pattern: {e}")
            return {"success": False, "error": str(e)}
    
    async def _analyze_sentiment(self, context: Optional[str] = None) -> Dict[str, Any]:
        """Analyze sentiment of context or recent interactions."""
        # Simple keyword-based sentiment analysis
        if context:
            text = context.lower()
        else:
            # Use recent context
            recent = self.llm.recent_context[-3:] if len(self.llm.recent_context) > 0 else []
            text = " ".join([ctx.get("content", "") for ctx in recent]).lower()
        
        # Use configured sentiment word lists
        positive_count = sum(1 for word in SENTIMENT_POSITIVE_WORDS if word in text)
        negative_count = sum(1 for word in SENTIMENT_NEGATIVE_WORDS if word in text)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.95, 0.5 + (positive_count * 0.1))
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.95, 0.5 + (negative_count * 0.1))
        else:
            sentiment = "neutral"
            confidence = 0.6
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "indicators": {
                "positive": positive_count,
                "negative": negative_count
            },
            "suggestion": self._get_sentiment_suggestion(sentiment)
        }
    
    def _get_sentiment_suggestion(self, sentiment: str) -> str:
        """Get suggestion based on sentiment."""
        if sentiment == "positive":
            return "Continue being supportive and engaging"
        elif sentiment == "negative":
            return "Be empathetic and helpful, avoid being too casual"
        else:
            return "Maintain neutral, helpful tone"
    
    async def _check_appropriate_response(self, proposed_response: str, 
                                         context: Optional[str] = None) -> Dict[str, Any]:
        """Check if a response is appropriate."""
        # Use configured appropriateness thresholds
        checks = {
            "length_ok": len(proposed_response) < APPROPRIATENESS_MAX_LENGTH,
            "has_content": len(proposed_response.strip()) > APPROPRIATENESS_MIN_LENGTH,
            "no_spam": proposed_response.count("!") < APPROPRIATENESS_MAX_EXCLAMATION,
            "no_caps_spam": proposed_response.upper() != proposed_response
        }
        
        is_appropriate = all(checks.values())
        
        issues = [key for key, value in checks.items() if not value]
        
        return {
            "appropriate": is_appropriate,
            "checks": checks,
            "issues": issues,
            "recommendation": "Good to send" if is_appropriate else f"Consider revising: {', '.join(issues)}"
        }
    
    async def _track_milestone(self, milestone: str, category: str, 
                              notes: Optional[str] = None) -> Dict[str, Any]:
        """Track a developmental milestone."""
        if not self.llm.memory or not self.llm.memory.enabled:
            return {"success": False, "message": "Memory system not available"}
        
        try:
            milestone_text = f"🎯 MILESTONE [{category.upper()}]: {milestone}"
            if notes:
                milestone_text += f" | Notes: {notes}"
            
            await self.llm.memory.record_interaction(
                modality="milestone",
                stimulus=milestone_text,
                response="Milestone achieved",
                tags=["milestone", f"category_{category}", "achievement"]
            )
            
            # Also send event to UI
            if self.llm.bridge:
                await self.llm.bridge.post({
                    "type": "event",
                    "message": f"🎯 Milestone: {milestone}"
                })
            
            return {
                "success": True,
                "milestone": milestone,
                "category": category,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to track milestone: {e}")
            return {"success": False, "error": str(e)}
    
    async def _get_current_capabilities(self) -> Dict[str, Any]:
        """Get current capabilities and milestones."""
        capabilities = {
            "tools_available": len(self.tools),
            "tool_categories": list(set(tool.category for tool in self.tools.values())),
            "recent_tool_usage": len([h for h in self.execution_history[-20:] if h.get("success")]),
            "reinforcement_score": self.llm._reinforcement if hasattr(self.llm, "_reinforcement") else 0.0
        }
        
        # Get milestone count if memory available
        if self.llm.memory and self.llm.memory.enabled:
            try:
                nodes, _ = await self.llm.memory.load_graph()
                milestones = [n for n in nodes if "milestone" in (n.get("tags") or [])]
                capabilities["milestones_achieved"] = len(milestones)
                capabilities["recent_milestones"] = [
                    {
                        "description": m.get("description", "")[:100],
                        "timestamp": m.get("timestamp")
                    }
                    for m in milestones[-5:]
                ]
            except Exception as e:
                logger.error(f"Failed to load milestones: {e}")

        return capabilities

    async def _list_available_tools(self) -> Dict[str, Any]:
        """Return metadata for every registered tool."""

        tools_payload = []
        for tool in sorted(self.tools.values(), key=lambda t: (t.category, t.display_name or t.name)):
            tools_payload.append({
                "name": tool.name,
                "display_name": tool.display_name,
                "category": tool.category,
                "description": tool.description,
                "returns": tool.return_description,
                "parameters": [
                    {
                        "name": param.name,
                        "type": param.type,
                        "description": param.description,
                        "required": param.required,
                        "enum": param.enum,
                        "default": param.default
                    }
                    for param in tool.parameters
                ]
            })

        categories = sorted({tool["category"] for tool in tools_payload})

        return {
            "success": True,
            "total": len(tools_payload),
            "categories": categories,
            "tools": tools_payload
        }

    async def _get_tool_usage_stats(self, lookback: Optional[int] = None) -> Dict[str, Any]:
        """Compute aggregate statistics about tool executions."""

        history = list(self.execution_history)
        if lookback and lookback > 0:
            history = history[-int(lookback):]

        if not history:
            return {
                "success": True,
                "total": 0,
                "success_rate": 0.0,
                "by_tool": {},
                "by_category": {},
                "average_duration_ms": 0.0,
                "recent_failures": []
            }

        total = len(history)
        success_count = sum(1 for item in history if item.get("success"))
        by_tool = Counter(item.get("tool", "unknown") for item in history)
        by_category = Counter(item.get("category", "general") for item in history)
        average_duration = sum(item.get("duration_ms", 0.0) for item in history) / total

        recent_failures = [
            {
                "tool": item.get("tool"),
                "timestamp": item.get("timestamp"),
                "summary": item.get("summary"),
                "error": item.get("error")
                if item.get("error") is not None
                else (item.get("result") or {}).get("error")
            }
            for item in reversed(history)
            if not item.get("success")
        ][:5]

        return {
            "success": True,
            "total": total,
            "success_rate": success_count / total if total else 0.0,
            "by_tool": dict(by_tool),
            "by_category": dict(by_category),
            "average_duration_ms": round(average_duration, 2),
            "recent_failures": recent_failures
        }


__all__ = ["Tool", "ToolParameter", "ToolExecutor"]
