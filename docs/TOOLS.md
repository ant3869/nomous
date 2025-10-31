# LLM Tool System Documentation

## Overview

The Nomous LLM has access to a comprehensive toolset that enables it to:
- **Remember**: Search and recall past interactions and memories
- **Learn**: Self-evaluate, identify patterns, and track improvements
- **Observe**: Record important observations about users and environment
- **Grow**: Track developmental milestones and capabilities
- **Understand**: Analyze sentiment and social context

## How It Works

The LLM can invoke tools by including special markers in its output:

```
TOOL_CALL: {"tool": "tool_name", "args": {"param": "value"}}
```

When the LLM generates this pattern, the tool executor:
1. Parses the tool call
2. Validates parameters
3. Executes the tool function
4. Returns results to the LLM
5. Sends results to the UI

## Available Tools

### Memory Tools

#### search_memory
Search through past interactions and memories to recall relevant information.

**Parameters:**
- `query` (string, required): What to search for in memory
- `limit` (number, optional, default=5): Maximum results to return
- `modality` (string, optional): Filter by type: "text", "audio", "vision", "autonomous", "all"

**Example:**
```json
TOOL_CALL: {"tool": "search_memory", "args": {"query": "Python", "limit": 3}}
```

**Use Cases:**
- Remembering previous conversations
- Recalling user preferences
- Finding past observations
- Maintaining conversation continuity

#### recall_recent_context
Retrieve the most recent context and interactions.

**Parameters:**
- `count` (number, optional, default=5): Number of recent items to recall

**Example:**
```json
TOOL_CALL: {"tool": "recall_recent_context", "args": {"count": 3}}
```

**Use Cases:**
- Understanding what just happened
- Maintaining short-term context
- Following conversation flow

### Observation Tools

#### record_observation
Record an important observation or insight.

**Parameters:**
- `observation` (string, required): What you observed
- `category` (string, required): Type - "user_preference", "pattern", "behavior", "environment", "insight"
- `importance` (number, optional, default=5): How important (1-10)
- `tags` (array, optional): Tags for categorization

**Example:**
```json
TOOL_CALL: {"tool": "record_observation", "args": {
    "observation": "User prefers technical explanations",
    "category": "user_preference",
    "importance": 8,
    "tags": ["communication", "learning_style"]
}}
```

**Use Cases:**
- Learning user preferences
- Noting environmental changes
- Recording behavioral patterns
- Building long-term understanding

### Learning Tools

#### evaluate_interaction
Self-evaluate how well you handled an interaction.

**Parameters:**
- `quality_score` (number, required): Rate response quality (1-10)
- `what_worked` (string, optional): What you did well
- `what_to_improve` (string, optional): What could be better

**Example:**
```json
TOOL_CALL: {"tool": "evaluate_interaction", "args": {
    "quality_score": 8,
    "what_worked": "Gave clear technical explanation",
    "what_to_improve": "Could have been more concise"
}}
```

**Use Cases:**
- Self-improvement
- Learning from mistakes
- Tracking quality over time
- Reinforcement learning

**Note:** This tool automatically applies reinforcement based on the quality score.

#### identify_pattern
Identify and record a pattern you've noticed.

**Parameters:**
- `pattern` (string, required): Description of the pattern
- `occurrences` (number, optional, default=1): How many times observed
- `confidence` (number, optional, default=0.7): Confidence level (0.0-1.0)

**Example:**
```json
TOOL_CALL: {"tool": "identify_pattern", "args": {
    "pattern": "User asks technical questions in mornings",
    "occurrences": 5,
    "confidence": 0.85
}}
```

**Use Cases:**
- Social behavior learning
- Understanding user habits
- Predictive responses
- Personalization

#### track_milestone
Record a developmental milestone or achievement.

**Parameters:**
- `milestone` (string, required): What was achieved
- `category` (string, required): Type - "communication", "understanding", "capability", "social", "technical"
- `notes` (string, optional): Additional context

**Example:**
```json
TOOL_CALL: {"tool": "track_milestone", "args": {
    "milestone": "First successful tool use",
    "category": "technical",
    "notes": "Used search_memory effectively"
}}
```

**Use Cases:**
- Tracking growth
- Celebrating achievements
- Understanding capabilities
- Progress monitoring

#### get_current_capabilities
Review your current capabilities and tracked milestones.

**Parameters:** None

**Example:**
```json
TOOL_CALL: {"tool": "get_current_capabilities", "args": {}}
```

**Returns:**
- Number of available tools
- Tool categories
- Recent tool usage
- Reinforcement score
- Milestones achieved

**Use Cases:**
- Self-awareness
- Understanding limitations
- Planning responses
- Meta-cognition

### Social Behavior Tools

#### analyze_sentiment
Analyze the sentiment or emotional tone of interactions.

**Parameters:**
- `context` (string, optional): What to analyze (leave empty for recent context)

**Example:**
```json
TOOL_CALL: {"tool": "analyze_sentiment", "args": {"context": "user's last message"}}
```

**Returns:**
- Sentiment: "positive", "negative", or "neutral"
- Confidence level
- Indicators (positive/negative word counts)
- Suggestion for appropriate response tone

**Use Cases:**
- Understanding emotions
- Adjusting response tone
- Empathy
- Social awareness

#### check_appropriate_response
Check if a proposed response is appropriate before sending.

**Parameters:**
- `proposed_response` (string, required): What you're thinking of saying
- `context` (string, optional): Current situation context

**Example:**
```json
TOOL_CALL: {"tool": "check_appropriate_response", "args": {
    "proposed_response": "That's interesting! Tell me more.",
    "context": "user just shared something personal"
}}
```

**Returns:**
- `appropriate`: Boolean indicating if response is okay
- `checks`: Individual check results
- `issues`: List of problems found
- `recommendation`: What to do

**Checks performed:**
- Length appropriate (not too long)
- Has substance (not too short)
- Not over-excited (too many exclamation marks)
- Not all caps (shouting)

**Use Cases:**
- Quality control
- Social appropriateness
- Avoiding mistakes
- Professional communication

## Configuration

Enable/disable tools in `config.yaml`:

```yaml
llm:
  tools_enabled: true  # Set to false to disable all tools
```

## Tool Execution Flow

1. LLM generates response with embedded tool calls
2. Tool executor parses `TOOL_CALL:` patterns
3. Each tool is validated and executed
4. Results sent to UI via WebSocket
5. Results available for follow-up use
6. Tool execution tracked in history

## UI Integration

Tool results are sent to the frontend via WebSocket messages:

```json
{
  "type": "tool_result",
  "tool": "tool_name",
  "result": { ... }
}
```

The UI can display these results to show the AI's thinking process and tool usage.

## Best Practices

### When to Use Tools

**Use tools when:**
- You need to remember past information
- You're learning something important about the user
- You notice a pattern worth recording
- You want to improve your responses
- You achieve something new
- You need to understand emotional context

**Don't overuse tools:**
- Not every response needs a tool
- Simple conversations don't need memory searches
- Balance tool use with natural conversation

### Example Usage Patterns

**Pattern 1: Memory-Enhanced Response**
```
User: "Do you remember what we talked about last time?"
AI: Let me check... TOOL_CALL: {"tool": "search_memory", "args": {"query": "last conversation", "limit": 3}}
Based on my memory, we discussed Python programming and you were learning about async/await.
```

**Pattern 2: Self-Improvement**
```
User: "Thanks, that explanation was really clear!"
AI: Thank you! TOOL_CALL: {"tool": "evaluate_interaction", "args": {"quality_score": 9, "what_worked": "Clear explanation"}}
I'm glad it helped!
```

**Pattern 3: Pattern Recognition**
```
AI: I've noticed something interesting... 
TOOL_CALL: {"tool": "identify_pattern", "args": {"pattern": "User prefers code examples", "occurrences": 3, "confidence": 0.8}}
You seem to learn best with practical examples. I'll keep that in mind!
```

## Development

### Adding New Tools

To add a custom tool, register it in `ToolExecutor._register_builtin_tools()`:

```python
self.register_tool(Tool(
    name="my_custom_tool",
    description="What this tool does",
    parameters=[
        ToolParameter("param1", "string", "Description", required=True),
    ],
    function=self._my_custom_tool,
    category="general"
))
```

Then implement the tool function:

```python
async def _my_custom_tool(self, param1: str) -> Dict[str, Any]:
    """Implementation of my custom tool."""
    # Tool logic here
    return {"success": True, "result": "something"}
```

### Testing Tools

Run the test suite:

```bash
python tests/test_tools.py
```

Or with pytest:

```bash
pytest tests/test_tools.py -v
```

## Troubleshooting

### Tool Not Found
- Check tool name spelling
- Ensure tool is registered
- Verify tools_enabled is true

### Tool Execution Fails
- Check parameter requirements
- Verify parameter types
- Review tool execution logs
- Check memory system is enabled

### Tools Not Being Called
- Verify LLM is generating correct syntax
- Check tool instructions in prompt
- Review temperature settings (too low may be too conservative)
- Ensure tools_enabled is true

## Future Enhancements

Planned improvements:
- RAG (Retrieval Augmented Generation) integration
- Advanced pattern recognition with ML
- Multi-modal tool support
- Tool chaining and workflows
- External API integration
- Custom user-defined tools
- Tool usage analytics
- Automatic tool selection

## Performance Considerations

- Tool execution is asynchronous and non-blocking
- Tool results cached for performance
- Execution history limited to 100 recent calls
- Memory searches optimized for speed
- Tools designed for minimal latency impact

## Security

- All tool parameters are validated
- SQL injection protection in memory queries
- Tool execution sandboxed
- No external network access from tools
- User data privacy maintained
- Audit trail of all tool usage
