# 🔧 Bug Fixes & Improvements

## Issues Fixed

### 1. ❌ Role-Playing Responses
**Problem:** Model speaking like system prompts: "I'm here to help, so feel free to chat..."

**Fix:** Completely rewrote prompts to be natural and concise
- Removed verbose "system_prompt" instructions
- Simple, direct prompts: "Person: {input}\n\nYou (reply naturally):"
- Added stop sequences to prevent role-play artifacts: "Person:", "You:", "User:"

**Result:** Natural, conversational responses ✅

### 2. ❌ Console Clutter
**Problem:** Duplicate messages, spam from "unknown message: pong", "speak →" repeated

**Fix:** Added smart log filtering
- Skip pong/ping messages
- Filter duplicate consecutive messages
- Remove "speak →" (already in status)
- Reduced console history from 400 → 150 lines

**Result:** Clean, readable console ✅

### 3. ❌ Thoughts Tab Empty
**Problem:** Thoughts tab showed nothing useful

**Fix:** Added real thought streaming
- New message type: `{"type": "thought", "text": "..."}`
- Stream prompt, generation chunks, and final output
- Separate `thoughtLines` array in UI state
- Purple-colored, timestamped thought traces

**Result:** Live thinking process visible ✅

### 4. ❌ Microphone Not Working
**Problem:** Mic picking up sound but AI not responding

**Fix:** Enhanced audio trigger and logging
- Better logging: "STT FINAL: 'text'" → "Triggering LLM with audio: text"
- Error detection: Shows "ERROR: LLM not connected" if issue
- Reduced partial result spam (only log substantial partials >5 chars)
- Changed UI message: "🎤 Heard" → "🎤 You said" for clarity

**Result:** Mic now triggers AI responses + clear error messages ✅

### 5. ❌ Over-Talkative AI
**Problem:** AI spoke every single time it saw something, no personality

**Fix:** Added decision-making and silence
- Vision: Only speaks if sees person/gesture (80% of interesting events)
- Vision: 20% random silence even if interesting ("building character")
- Autonomous: 70% chance to stay quiet, think silently
- Quiet observations sent as thoughts: "Observing: {description}"

**Result:** AI with personality, doesn't talk constantly ✅

## File Changes

### llm.py
1. `_build_prompt()` - Simplified, natural prompts
2. `_generate()` - Added thought streaming, better stop sequences
3. `process_vision()` - Added decision-making (speak or stay quiet)
4. `autonomous_thought()` - 70% silence, more intentional

### audio.py
1. `feed_base64_pcm()` - Better logging, error messages
2. Reduced partial result spam

### App.tsx
1. Added `thoughtLines` to state
2. Added `thought` message handler
3. Updated Thoughts tab to show `thoughtLines`
4. Added log filtering (skip duplicates, pong, spam)

## New Features

### Thought Streaming
Watch the AI think in real-time:
```
[10:30:45] Prompt: Previous: audio: hello...
[10:30:46] Generating: Hi there! How...
[10:30:47] Final: Hi there! How can I help you?
```

### Decision-Making
AI decides when to speak:
- **Interesting event** (person, gesture): 80% speak, 20% silent
- **Boring event** (empty room): Silent, thinks only
- **Autonomous**: 30% speak, 70% silent

### Personality
AI can be thoughtful and quiet, not always chatty!

## How to Test

### 1. Microphone Response
```bash
# Speak clearly into mic: "Hello"
# Terminal should show:
STT FINAL: 'hello'
Triggering LLM with audio: hello

# UI should show:
🎤 You said: hello
# Then AI responds within 1-3 seconds
```

### 2. Thoughts Tab
```bash
# Open Thoughts tab
# Should see purple timestamped entries:
[10:30:45] Prompt: Previous: ...
[10:30:46] Generating: Hi there...
[10:30:47] Final: Hi there! How can I help?
```

### 3. Natural Responses
```bash
# AI should respond naturally:
✅ "Hi! What's up?"
✅ "Hey there!"
✅ "I see you waving!"

# NOT like this anymore:
❌ "I'm here to help, so feel free to chat..."
❌ "What's on your mind? (I'd be happy to respond accordingly...)"
```

### 4. Clean Console
```bash
# Console should NOT show:
❌ unknown message: {"type":"pong"}
❌ speak → [repeated messages]
❌ [duplicate] [duplicate] [duplicate]

# Console SHOULD show:
✅ [10:30:45] connected → ws://localhost:8765
✅ [10:30:46] 🎤 You said: hello
✅ [10:30:47] 👋 Gesture: waving
```

### 5. Quieter AI
```bash
# AI should NOT speak every time camera updates
# Sometimes it just thinks quietly:
# Thoughts tab: "Quietly observing..."

# Only speaks when:
- Sees a person
- Detects a gesture
- Someone talks to it (mic)
- Randomly decides to (30% autonomous)
```

## Configuration

### Make AI More Talkative
In `llm.py`, change:
```python
# Vision: Speak less
if not interesting or (random.random() < 0.2):  # Change 0.2 → 0.0 (always speak if interesting)

# Autonomous: Speak more
if random.random() < 0.7:  # Change 0.7 → 0.3 (speak 70% instead of 30%)
```

### Make AI Quieter (Current Settings)
```python
# Vision: 80% speak if interesting, 20% silent
if not interesting or (random.random() < 0.2):

# Autonomous: 30% speak, 70% silent
if random.random() < 0.7:
```

## Troubleshooting

### Mic Still Not Working?
Check terminal for:
```
STT FINAL: 'your text here'
Triggering LLM with audio: your text here
```

If you see "ERROR: LLM not connected":
- Restart the server
- Check that LLM initialized properly
- Look for "LLM reference set for AudioSTT"

### Thoughts Tab Empty?
- Restart the UI (reload browser)
- Trigger any action (speak to mic, wave)
- Should see thought streaming immediately

### Still Too Chatty?
Increase silence probability in llm.py:
```python
# Line ~180 in process_vision:
if random.random() < 0.5:  # 50% silence instead of 20%

# Line ~230 in autonomous_thought:
if random.random() < 0.9:  # 90% silence instead of 70%
```

### Console Still Cluttered?
Add more filters in App.tsx:
```typescript
const skipPatterns = [
  /^unknown message: \{"type":"pong"\}/,
  /^speak.*→.*$/,
  /YOUR_PATTERN_HERE/,  // Add custom patterns
];
```

## Summary

✅ **Natural responses** - No more role-play
✅ **Clean console** - No spam/duplicates
✅ **Thought streaming** - See AI think
✅ **Mic working** - Responds to speech
✅ **Personality** - Decides when to speak

The AI now feels more like a thoughtful companion rather than an over-eager chatbot!
