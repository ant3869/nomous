# ✅ Testing Guide - Verify All Fixes

## Quick Tests (5 minutes)

### 1. 🎤 Test Microphone
```
Action: Speak into mic: "Hello"

Expected:
- Terminal shows: "STT FINAL: 'hello'"
- Terminal shows: "Triggering LLM with audio: hello"
- Console shows: "🎤 You said: hello"
- AI responds within 1-3 seconds

✅ Working if: AI responds naturally
❌ Not working if: No response or error messages
```

### 2. 🧠 Test Thoughts Tab
```
Action: 
1. Open UI → Click "Thoughts" tab
2. Speak into mic or wave

Expected:
- Purple timestamped entries appear
- Shows: "Prompt: Previous: ..."
- Shows: "Generating: ..."
- Shows: "Final: [response]"

✅ Working if: See thinking process
❌ Not working if: Tab is empty or only shows old console logs
```

### 3. 💬 Test Natural Responses
```
Action: Speak: "What's up?"

Expected:
✅ "Hey! Not much, you?"
✅ "Hi there!"
✅ "Hello! How are you?"

NOT This:
❌ "I'm here to help, so feel free..."
❌ "What's on your mind? (I'd be happy to respond accordingly..."

✅ Working if: Natural, casual responses
❌ Not working if: Role-play or system-prompt style
```

### 4. 📺 Test Clean Console
```
Action: Open Console tab

Expected:
✅ Timestamped events
✅ Clear, readable messages
✅ No duplicate entries

NOT This:
❌ "unknown message: {"type":"pong"}"
❌ "speak → [repeated 10 times]"
❌ Duplicate identical messages

✅ Working if: Clean, unique messages
❌ Not working if: Spam or duplicates
```

### 5. 🤫 Test Personality (AI Stays Quiet Sometimes)
```
Action: Wait and watch video feed for 2-3 minutes

Expected:
- AI does NOT speak every single time
- Quiet observations in Thoughts: "Quietly observing..."
- Only speaks when sees person/gesture or feels like it

✅ Working if: AI quiet most of the time
❌ Not working if: AI talks constantly about every camera update
```

### 6. 👋 Test Gesture Recognition (Optional, needs MediaPipe)
```
Action: Wave at camera

Expected:
- Console: "👋 Gesture: open_hand"
- AI responds: "I see an open hand!"
- Response within 1 second

✅ Working if: Instant recognition
❌ Not working if: No gesture detection
(If not working: pip install mediapipe, then restart - Task models auto-download on first run)
```

### 7. ⚡ Test GPU Speed (Optional, if GPU enabled)
```
Action: 
1. Speak: "Tell me a short story"
2. Watch terminal

Expected:
- Response starts in 1-3 seconds
- nvidia-smi shows GPU usage 80-100%

✅ Working if: Fast response + GPU active
❌ Not working if: 8-15 second delay + GPU idle
(If not working: run the launcher - python scripts/start.py - it installs the GPU packages)
```

## Automated Tests

### Test Script
```bash
# Test GPU setup
python test_gpu.py

# Test bug fixes (requires server running)
python test_fixes.py
```

## Troubleshooting

### Microphone Not Working
1. Check terminal for "LLM reference set for AudioSTT"
2. Check for "STT FINAL: 'your text'"
3. If missing: Restart server

### Thoughts Tab Empty
1. Reload browser page
2. Trigger any action (speak/wave)
3. Should see purple entries immediately

### Role-Play Responses Still Happening
1. Check llm.py was updated
2. Restart server completely
3. Clear any cached responses

### Console Still Cluttered
1. Reload browser page
2. Check App.tsx was updated
3. Should filter duplicates automatically

### AI Too Talkative
1. Edit llm.py
2. Change `random.random() < 0.2` → `0.5` (line ~180)
3. Change `random.random() < 0.7` → `0.9` (line ~230)
4. Restart server

### AI Too Quiet
1. Edit llm.py
2. Change `random.random() < 0.2` → `0.0` (line ~180)
3. Change `random.random() < 0.7` → `0.3` (line ~230)
4. Restart server

## Expected Behavior Summary

| Test | Before | After |
|------|--------|-------|
| Mic Response | None | 1-3s ✅ |
| Thoughts Tab | Empty | Purple entries ✅ |
| Responses | Role-play | Natural ✅ |
| Console | Spam | Clean ✅ |
| Personality | Always talks | Quiet sometimes ✅ |
| Gestures | None | Instant ✅ |
| Speed (GPU) | 8-15s | 1-3s ✅ |

## Quick Fix Commands

```bash
# Restart server
Ctrl+C
python run_bridge.py

# Reload UI
F5 in browser

# Check GPU
nvidia-smi

# View logs
# (Look at terminal where server is running)
```

## All Tests Passing?

🎉 **Congratulations!** Your Nomous system is fully optimized!

You should have:
- ✅ Natural conversational AI
- ✅ Clean, readable console
- ✅ Visible thinking process
- ✅ Working microphone
- ✅ AI with personality
- ✅ Gesture recognition (if MediaPipe installed)
- ✅ GPU acceleration (if the launcher installed the GPU packages)

## Need Help?

- **BUG_FIXES.md** - Detailed fix documentation
- **QUICKSTART.md** - Setup guide
- **OPTIMIZATION_SUMMARY.md** - Performance details
- **CHANGELOG.md** - All changes

---

**Pro Tip:** Watch the Thoughts tab while the AI is processing - it's fascinating to see it think!
