# 📥 Installation Instructions - Bug Fixes

## Quick Install (Recommended)

### 1. Download & Extract
```bash
# Download nomous_fixes_v2.tar.gz from Claude
# Extract to a temporary folder

# Windows:
tar -xzf nomous_fixes_v2.tar.gz

# Or use 7-Zip / WinRAR
```

### 2. Backup Your Files
```bash
# In your project directory, backup these files:
copy backend\llm.py backend\llm_backup.py
copy backend\audio.py backend\audio_backup.py
copy config.yaml config_backup.yaml
copy App.tsx App_backup.tsx
```

### 3. Replace Files
```bash
# Copy the updated files to your project:
copy llm.py backend\
copy audio.py backend\
copy config.yaml .
copy App.tsx .
copy video_optimized.py .
```

### 4. Optional: GPU Setup
```bash
# For 5-10x performance boost:
setup_gpu.bat
# Then restart computer
```

### 5. Restart
```bash
# Restart the server
python run_bridge.py

# Reload UI (F5 in browser)
```

---

## Manual Install (If You Prefer)

Download these individual files from Claude and replace:

### Backend Files (in `backend/` folder):
1. **llm.py** - Natural prompts, thought streaming, GPU support
2. **audio.py** - Better logging, mic trigger fix

### Root Files:
3. **config.yaml** - Added GPU and optimization parameters
4. **App.tsx** - Thought streaming, clean console

### New Files:
5. **video_optimized.py** - GPU-friendly video + gestures
6. **setup_gpu.bat** - GPU installer (optional)
7. **test_gpu.py** - GPU verification (optional)
8. **test_fixes.py** - Fix verification (optional)

---

## What Changed in Each File

### llm.py (backend/)
- ✅ Natural prompts (removed role-play)
- ✅ Thought streaming to UI
- ✅ Decision-making (speak or stay quiet)
- ✅ GPU layer support

### audio.py (backend/)
- ✅ Better logging ("STT FINAL", "Triggering LLM")
- ✅ Error detection and messages
- ✅ Reduced partial result spam

### config.yaml (root)
```yaml
# Added these sections:
llm:
  n_gpu_layers: 35          # NEW

camera:
  process_width: 640        # NEW
  process_height: 360       # NEW
  frame_skip: 2             # NEW

ui:
  snapshot_debounce: 3      # CHANGED from 4
  motion_sensitivity: 25    # CHANGED from 30
  gesture_enabled: true     # NEW
  gesture_cooldown: 3       # NEW
  vision_cooldown: 12       # NEW
```

### App.tsx (root)
- ✅ Added `thoughtLines` state
- ✅ Added thought message handler
- ✅ Updated Thoughts tab
- ✅ Log filtering (no duplicates/spam)

---

## Quick Verification

After installing, test these:

### 1. Microphone
```bash
Speak: "Hello"
Expected: AI responds in 1-3 seconds
```

### 2. Thoughts Tab
```bash
Open Thoughts tab in UI
Expected: See purple timestamped entries
```

### 3. Console
```bash
Open Console tab in UI
Expected: Clean, no spam or duplicates
```

### 4. Natural Responses
```bash
Speak: "What's up?"
Expected: Casual response like "Hey! Not much, you?"
NOT: "I'm here to help, so feel free..."
```

---

## Troubleshooting

### "Files not working after install"
- Make sure you replaced files in correct locations
- Restart Python server completely
- Reload browser (F5)

### "Can't extract .tar.gz"
- Use 7-Zip or WinRAR on Windows
- Or download individual files instead

### "Still have old bugs"
- Check file timestamps (should be recent)
- Make sure you didn't miss any files
- Try manual install instead

---

## Files Checklist

Before starting server, verify you have:

- [ ] backend/llm.py (updated)
- [ ] backend/audio.py (updated)
- [ ] config.yaml (updated)
- [ ] App.tsx (updated)
- [ ] video_optimized.py (new, optional)
- [ ] setup_gpu.bat (new, optional)

Then:
- [ ] Restart server: `python run_bridge.py`
- [ ] Reload browser: Press F5
- [ ] Test microphone
- [ ] Check Thoughts tab
- [ ] Verify console is clean

---

## Need Help?

- **TESTING.md** - Simple 5-minute test guide
- **BUG_FIXES.md** - What was fixed and how
- **CHANGELOG.md** - Complete change list
- **QUICKSTART.md** - GPU setup guide

All documentation files are in the download package!

---

**Version:** 2.0  
**Files to Replace:** 4 (llm.py, audio.py, config.yaml, App.tsx)  
**New Files:** 8+ (optional documentation and tools)
