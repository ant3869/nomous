# LLM Tool System - Usage Examples

This document provides practical examples of how the LLM uses tools in real scenarios.

## Table of Contents
- [Memory Tools Examples](#memory-tools-examples)
- [Observation Tools Examples](#observation-tools-examples)
- [Learning Tools Examples](#learning-tools-examples)
- [Social Tools Examples](#social-tools-examples)
- [Multi-Tool Scenarios](#multi-tool-scenarios)

## Memory Tools Examples

### Example 1: Recalling Previous Conversations

**Scenario:** User asks about a previous discussion

**User:** "Do you remember what we talked about yesterday?"

**LLM Response:**
```
Let me check my memory...
TOOL_CALL: {"tool": "search_memory", "args": {"query": "yesterday conversation", "limit": 5}}

Based on my memory search, we discussed Python programming, specifically async/await patterns and how to handle concurrent operations. You were working on a WebSocket server implementation.
```

**Tool Result:**
```json
{
  "found": 3,
  "results": [
    {
      "type": "stimulus",
      "content": "User asked about async/await in Python",
      "source": "text",
      "timestamp": "2024-01-15T14:30:00",
      "relevance": 1.0
    },
    {
      "type": "concept",
      "content": "Explained asyncio event loop and coroutines",
      "source": "assistant",
      "timestamp": "2024-01-15T14:31:00",
      "relevance": 0.8
    }
  ]
}
```

### Example 2: Maintaining Conversation Context

**Scenario:** Following up on recent discussion

**User:** "Can you explain more about that?"

**LLM Response:**
```
Let me recall what we were just discussing...
TOOL_CALL: {"tool": "recall_recent_context", "args": {"count": 3}}

Sure! We were talking about WebSocket implementations. Specifically, you asked about handling multiple concurrent connections efficiently...
```

**Tool Result:**
```json
{
  "count": 3,
  "context": [
    {
      "type": "user_text",
      "content": "How do I handle multiple WebSocket connections?",
      "timestamp": 1705330200
    },
    {
      "type": "assistant",
      "content": "You can use asyncio to manage multiple connections...",
      "timestamp": 1705330210
    },
    {
      "type": "user_text",
      "content": "Can you explain more about that?",
      "timestamp": 1705330220
    }
  ]
}
```

## Observation Tools Examples

### Example 3: Recording User Preferences

**Scenario:** Noticing user prefers detailed technical explanations

**LLM Internal Process:**
```
User has asked for detailed code examples three times now. This seems to be a preference.
TOOL_CALL: {"tool": "record_observation", "args": {
  "observation": "User prefers detailed technical explanations with code examples",
  "category": "user_preference",
  "importance": 8,
  "tags": ["communication_style", "technical_depth"]
}}

I'll make sure to include more code examples in future responses.
```

**Tool Result:**
```json
{
  "success": true,
  "message": "Observation recorded: User prefers detailed technical explanations...",
  "category": "user_preference",
  "importance": 8
}
```

### Example 4: Environmental Observations

**Scenario:** Camera detects changes in environment

**LLM Internal Process:**
```
TOOL_CALL: {"tool": "record_observation", "args": {
  "observation": "Room lighting changed - much brighter now, possibly morning",
  "category": "environment",
  "importance": 5,
  "tags": ["vision", "lighting", "time_context"]
}}

Good morning! I notice it's gotten much brighter - looks like a beautiful day outside.
```

## Learning Tools Examples

### Example 5: Self-Evaluation After Interaction

**Scenario:** After explaining a complex topic

**LLM Internal Process:**
```
That explanation went well, but I could have been more concise.
TOOL_CALL: {"tool": "evaluate_interaction", "args": {
  "quality_score": 7,
  "what_worked": "Provided accurate technical information with code examples",
  "what_to_improve": "Response was too long, could have been more concise"
}}
```

**Tool Result:**
```json
{
  "quality_score": 7,
  "what_worked": "Provided accurate technical information with code examples",
  "what_to_improve": "Response was too long, could have been more concise",
  "timestamp": "2024-01-15T14:35:00"
}
```

**Note:** This automatically applies reinforcement (+4 points since 7-5=2, multiplied by 2)

### Example 6: Identifying Behavioral Patterns

**Scenario:** Noticing user behavior pattern over time

**LLM Internal Process:**
```
I've noticed that this user tends to ask technical questions in the morning and casual conversation in the evening.
TOOL_CALL: {"tool": "identify_pattern", "args": {
  "pattern": "User asks technical questions in morning (9-11 AM), casual topics in evening (6-9 PM)",
  "occurrences": 7,
  "confidence": 0.85
}}

This will help me adjust my tone appropriately based on time of day.
```

**Tool Result:**
```json
{
  "success": true,
  "pattern": "User asks technical questions in morning...",
  "occurrences": 7,
  "confidence": 0.85,
  "message": "Pattern recorded for future reference"
}
```

### Example 7: Tracking Developmental Milestones

**Scenario:** Successfully using a new capability

**LLM Internal Process:**
```
I just successfully used the memory search tool to help answer a question!
TOOL_CALL: {"tool": "track_milestone", "args": {
  "milestone": "First successful memory search to answer user question",
  "category": "technical",
  "notes": "Searched for 'Python async' and found 3 relevant past conversations"
}}
```

**Tool Result:**
```json
{
  "success": true,
  "milestone": "First successful memory search to answer user question",
  "category": "technical",
  "timestamp": "2024-01-15T14:40:00"
}
```

**UI Event:** 🎯 Milestone notification appears

### Example 8: Reviewing Capabilities

**Scenario:** User asks what the AI can do

**User:** "What can you do?"

**LLM Response:**
```
Let me check my current capabilities...
TOOL_CALL: {"tool": "get_current_capabilities", "args": {}}

I have access to 9 tools across 4 categories. I can search my memory (with 15 past interactions stored), make observations, learn from patterns, and analyze sentiment. I've achieved 3 milestones so far, including learning to use the memory system effectively.
```

**Tool Result:**
```json
{
  "tools_available": 9,
  "tool_categories": ["memory", "observation", "learning", "social"],
  "recent_tool_usage": 12,
  "reinforcement_score": 23.5,
  "milestones_achieved": 3,
  "recent_milestones": [
    {
      "description": "First successful memory search",
      "timestamp": "2024-01-15T14:40:00"
    }
  ]
}
```

## Social Tools Examples

### Example 9: Sentiment Analysis

**Scenario:** User expresses frustration

**User:** "This is so frustrating! Nothing is working!"

**LLM Internal Process:**
```
Let me check the emotional tone here...
TOOL_CALL: {"tool": "analyze_sentiment", "args": {}}

The sentiment is clearly negative. I should be empathetic and helpful, not casual.
```

**LLM Response:**
```
I can see this is frustrating for you. Let's work through this together - what specifically isn't working? I'll help you troubleshoot step by step.
```

**Tool Result:**
```json
{
  "sentiment": "negative",
  "confidence": 0.95,
  "indicators": {
    "positive": 0,
    "negative": 2
  },
  "suggestion": "Be empathetic and helpful, avoid being too casual"
}
```

### Example 10: Checking Response Appropriateness

**Scenario:** About to respond with a joke to a serious question

**LLM Internal Process:**
```
User is asking about a serious work problem. Let me check if a joke is appropriate...
TOOL_CALL: {"tool": "check_appropriate_response", "args": {
  "proposed_response": "Haha, sounds like you need more coffee! ☕😄",
  "context": "User asking about serious work issue"
}}

Hmm, that joke isn't appropriate here. Let me respond more professionally.
```

**Tool Result:**
```json
{
  "appropriate": false,
  "checks": {
    "length_ok": true,
    "has_content": true,
    "no_spam": false,
    "no_caps_spam": true
  },
  "issues": ["no_spam"],
  "recommendation": "Consider revising: no_spam"
}
```

**Actual Response:**
```
I understand you're dealing with a challenging work situation. Let me help you think through this systematically...
```

## Multi-Tool Scenarios

### Example 11: Comprehensive Response Pattern

**Scenario:** Complex user question requiring multiple tools

**User:** "I'm having trouble with the code you suggested yesterday. Can you help?"

**LLM Process:**
```
1. First, recall what code I suggested
TOOL_CALL: {"tool": "search_memory", "args": {"query": "code suggested yesterday", "limit": 3}}

2. Check sentiment to gauge urgency
TOOL_CALL: {"tool": "analyze_sentiment", "args": {}}

3. After helping, evaluate my response
TOOL_CALL: {"tool": "evaluate_interaction", "args": {
  "quality_score": 8,
  "what_worked": "Found relevant past code, understood the issue quickly",
  "what_to_improve": "Could have asked clarifying questions first"
}}
```

**LLM Response:**
```
I found the code from yesterday - it was the WebSocket server implementation. What specific issue are you encountering? Is it a connection problem, or something with the message handling?
```

### Example 12: Learning Loop

**Scenario:** Continuous improvement through tool use

**Day 1:**
```
TOOL_CALL: {"tool": "record_observation", "args": {
  "observation": "User struggles with async concepts",
  "category": "user_preference",
  "importance": 7
}}
```

**Day 3:**
```
TOOL_CALL: {"tool": "identify_pattern", "args": {
  "pattern": "User learns best with visual diagrams and step-by-step examples",
  "occurrences": 4,
  "confidence": 0.8
}}
```

**Day 5:**
```
User: "Can you explain asyncio again?"

TOOL_CALL: {"tool": "search_memory", "args": {"query": "user learning async", "limit": 5}}

Based on my observations, I'll use a step-by-step approach with examples...
```

**Day 7:**
```
TOOL_CALL: {"tool": "track_milestone", "args": {
  "milestone": "User successfully implemented async WebSocket server",
  "category": "communication",
  "notes": "Adapted teaching style based on observed learning patterns"
}}
```

### Example 13: Autonomous Thought with Tools

**Scenario:** LLM thinking autonomously without user input

**LLM Internal Process:**
```
It's been quiet for a while. Let me review what I've learned today...
TOOL_CALL: {"tool": "get_current_capabilities", "args": {}}

I've used tools 15 times today, primarily memory searches. I should reflect on this...
TOOL_CALL: {"tool": "evaluate_interaction", "args": {
  "quality_score": 7,
  "what_worked": "Effective use of memory to maintain context",
  "what_to_improve": "Could use observation tools more to learn user preferences"
}}

I've been relying heavily on memory searches. I should pay more attention to patterns and user preferences.
TOOL_CALL: {"tool": "record_observation", "args": {
  "observation": "Self-reflection: Over-relying on memory search, under-utilizing pattern recognition",
  "category": "insight",
  "importance": 6,
  "tags": ["meta_cognition", "self_improvement"]
}}
```

## Best Practices Demonstrated

### When to Use Each Tool

**Memory Tools (search_memory, recall_recent_context):**
- ✅ User asks "do you remember..."
- ✅ Need to maintain conversation context
- ✅ Following up on previous topics
- ❌ For brand new topics with no history

**Observation Tools (record_observation):**
- ✅ Notice consistent user preferences
- ✅ Environmental changes worth noting
- ✅ Behavioral patterns emerging
- ❌ For one-time random occurrences

**Learning Tools (evaluate_interaction, identify_pattern, track_milestone):**
- ✅ After meaningful interactions
- ✅ When noticing patterns (3+ occurrences)
- ✅ Achieving new capabilities
- ❌ After every single response (too frequent)

**Social Tools (analyze_sentiment, check_appropriate_response):**
- ✅ Emotional content detected
- ✅ Uncertain about response tone
- ✅ Sensitive topics
- ❌ For clearly neutral conversations

### Tool Usage Frequency Guidelines

| Tool | Recommended Frequency |
|------|---------------------|
| search_memory | As needed (user asks) |
| recall_recent_context | 1-2 times per conversation |
| record_observation | When pattern emerges |
| evaluate_interaction | Every 3-5 responses |
| identify_pattern | After 3+ similar observations |
| track_milestone | When achieving something new |
| get_current_capabilities | On request or daily review |
| analyze_sentiment | When emotion detected |
| check_appropriate_response | When uncertain |

## Tool Combinations That Work Well

1. **Memory + Evaluation**: After recalling past interactions, evaluate how well you used that memory
2. **Sentiment + Observation**: When detecting emotion, record the context as an observation
3. **Pattern + Milestone**: When a pattern is firmly established, track it as a milestone
4. **Capabilities + Evaluation**: Review capabilities, then evaluate performance

## Common Pitfalls to Avoid

❌ **Over-tooling**: Don't use tools for every response
```
Bad: Using evaluate_interaction after "Hello"
Good: Using it after helping solve a complex problem
```

❌ **Tool spam**: Don't call multiple tools unnecessarily
```
Bad: search_memory + recall_context + search_memory again
Good: Pick the most appropriate tool
```

❌ **Ignoring results**: Use tool results in your response
```
Bad: Call search_memory but don't mention findings
Good: "According to my memory search, we discussed..."
```

❌ **Wrong tool for job**: Use the right tool for the task
```
Bad: Using analyze_sentiment for factual questions
Good: Using search_memory for factual recall
```

## Success Metrics

Track these to measure effective tool use:
- Memory searches that successfully help answer questions
- Observations that lead to better future interactions
- Quality scores trending upward over time
- Patterns identified that improve responses
- Milestones achieved regularly
- Sentiment analyses that improve response appropriateness

## Next Steps

1. Experiment with tools in different scenarios
2. Track which tools help most
3. Refine tool usage based on feedback
4. Develop new tools as needs emerge
5. Share successful patterns with the community

For more information, see:
- [TOOLS.md](TOOLS.md) - Complete tool reference
- [README.md](../README.md) - Project overview
- [TESTING.md](TESTING.md) - Testing guidelines
