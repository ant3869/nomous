# Microphone and STT Fix Summary

## Problem
When speaking into the microphone, nothing was happening - no transcription appeared in the designated STT area, and the speech wasn't being processed by the model.

## Root Causes Identified
1. **Lack of visibility**: No detailed logging to diagnose where the audio pipeline was failing
2. **Silent failures**: Errors in microphone permission, audio context, or STT processing weren't being communicated to the user
3. **Missing feedback**: No clear indication when audio was actively streaming or when STT was disabled

## Fixes Implemented

### Frontend Improvements (`src/frontend/App.tsx`)

#### 1. Enhanced Microphone Logging
Added detailed console logging throughout the microphone lifecycle:
- 🎤 **Permission request**: Logs when requesting microphone access
- ✅ **Permission granted**: Confirms when user approves microphone
- 🔊 **Audio context creation**: Shows sample rate and initialization status
- 🎙️ **Pipeline setup**: Confirms audio processing chain is established
- 🎵 **Streaming started**: Notifies when first audio chunk is sent
- ❌ **Error messages**: Specific messages for different failure scenarios:
  - Permission denied (browser blocked or user denied)
  - No microphone found (device not connected)
  - Device busy (another app is using the mic)

#### 2. Auto-Start Microphone
Added `useEffect` hook that automatically starts the microphone when:
- WebSocket connection is established
- Microphone is enabled in settings
- Microphone is not already active

This ensures the microphone activates when you toggle it on in the UI.

#### 3. UI Enhancements
- **Warning badges** in the STT Monitor:
  - Shows warning if STT is disabled: "⚠️ STT is disabled. Enable it in the Devices tab"
  - Shows prompt if mic not started: "🎤 Click 'Enable Mic' above to start capturing audio"
- **Status indicators**: Shows mic status (Capturing/Armed/Muted) with color coding
- **VU meter**: Visual feedback showing audio levels in real-time

### Backend Improvements

#### 1. Enhanced STT Logging (`src/backend/audio.py`)
- Better warning messages when STT is disabled or model not loaded
- Clearer error messages to help diagnose configuration issues

#### 2. Audio Chunk Tracking (`scripts/run_bridge.py`)
- Added counter to track audio chunks received from frontend
- Logs when first audio chunk arrives: "🎤 First audio chunk received - STT processing started"
- Sends event to frontend: "🎤 Audio streaming active"
- Debug logging every 100 chunks to monitor continuous streaming

## How to Test

### Step 1: Start the Application
```powershell
python run_nomous.py
```

### Step 2: Open the Dashboard
Navigate to `http://localhost:5173` in your browser.

### Step 3: Check Initial Status
1. Look at the **Speech Transcription** card on the main dashboard
2. Check the console (bottom of the page) for any initialization messages

### Step 4: Enable Microphone
1. Click the **"Enable Mic"** button in the STT Monitor card, OR
2. Go to **Settings** → **Devices** tab
3. Toggle **"Microphone Capture"** to ON

### Step 5: Grant Permission
- Your browser will prompt for microphone permission
- Click **"Allow"** to grant access
- Watch the console for these messages:
  ```
  🎤 Microphone activation requested...
  🎤 Requesting microphone permission...
  ✅ Microphone permission granted
  🔊 Creating audio context...
  🔊 Audio context created (sample rate: 16000Hz)
  🎙️ Setting up audio processing pipeline...
  ✅ Microphone is now ACTIVE and listening
  ```

### Step 6: Verify STT is Enabled
1. In the STT Monitor card, check the status line shows: "STT • Enabled"
2. If it shows "Disabled", go to **Settings** → **Devices** → Toggle **"Speech-to-Text"** ON

### Step 7: Speak Into Microphone
1. Speak clearly into your microphone
2. Watch for:
   - **VU meter** should show activity (colored bar moves)
   - **Console log**: "🎵 Audio streaming started - speaking into microphone..."
   - **Backend log**: "🎤 First audio chunk received - STT processing started"
3. As you speak, you should see in the STT Monitor:
   - **Partial**: "Listening: [your words]..." (as you speak)
   - **Final**: "Captured: [complete sentence]" (when you pause)
   - **Forwarded**: "Dispatched: [text] → reasoning core" (sent to LLM)

### Step 8: Verify LLM Response
After speaking, the AI should:
1. Show status change to "Thinking"
2. Process your speech input
3. Generate a response
4. Speak the response (if TTS is enabled)

## Troubleshooting Guide

### No Console Logs Appearing
**Problem**: No "🎤 Microphone activation requested" message
**Solution**: 
- Check WebSocket connection status (should show "Connected")
- Refresh the page and try again

### Permission Denied Error
**Problem**: "❌ Microphone permission denied"
**Solution**:
1. Check browser URL bar for microphone icon
2. Click it and set permission to "Allow"
3. Reload the page
4. Try enabling mic again

### No Microphone Found
**Problem**: "❌ No microphone found"
**Solution**:
1. Check that a microphone is physically connected
2. Verify it's working in system settings
3. Try a different browser

### Microphone is Busy
**Problem**: "❌ Microphone is busy or being used by another application"
**Solution**:
1. Close other apps that might use the mic (Zoom, Discord, etc.)
2. Restart your browser
3. Try again

### VU Meter Not Moving
**Problem**: Microphone active but no audio detected
**Solution**:
1. Check system mic volume is not muted
2. Speak louder or closer to the microphone
3. Test mic in system sound settings
4. Try adjusting "Mic Sensitivity" in Settings → Audio

### No Transcription Appearing
**Problem**: Audio detected (VU moves) but no text appears
**Solution**:
1. **Check STT is enabled**: Settings → Devices → "Speech-to-Text" should be ON
2. **Check backend logs**: Look for "STT model not loaded" errors
3. **Verify Vosk model**: Check that `modules/models/vosk-model-small-en-us-0.15` exists
4. **Check sensitivity**: Lower sensitivity in Settings → Audio → "Mic Sensitivity"

### Backend Not Receiving Audio
**Problem**: Frontend logs show streaming but backend doesn't log "First audio chunk received"
**Solution**:
1. Check backend console for WebSocket connection errors
2. Verify backend is running and listening on correct port
3. Check firewall isn't blocking connections

## Configuration Options

### Mic Sensitivity
- **Location**: Settings → Audio → "Mic Sensitivity"
- **Range**: 0-100%
- **Effect**: 
  - Higher (85%+): Accepts very short utterances (1 char minimum)
  - Medium (60-84%): Accepts short phrases (2 chars minimum)
  - Lower (35-59%): Requires longer speech (3 chars minimum)
  - Very Low (<35%): Only accepts substantial input (4+ chars minimum)

### Sample Rate
- **Fixed**: 16000 Hz (required by Vosk model)
- **Auto-resampled**: Frontend automatically converts to correct rate

## Files Modified

1. **`src/frontend/App.tsx`**:
   - Enhanced `setMic()` function with detailed logging
   - Added auto-start effect for microphone
   - Improved error messages
   - Added UI warning badges

2. **`src/backend/audio.py`**:
   - Enhanced logging in `feed_base64_pcm()`
   - Better error messages for troubleshooting

3. **`scripts/run_bridge.py`**:
   - Added audio chunk counter
   - Added first-chunk notification
   - Enhanced WebSocket message handling

## Success Indicators

When everything is working correctly, you should see:

### Console Logs (in order):
```
🎤 Microphone activation requested...
🎤 Requesting microphone permission...
✅ Microphone permission granted
🔊 Creating audio context...
🔊 Audio context created (sample rate: 16000Hz)
🎙️ Setting up audio processing pipeline...
✅ Microphone is now ACTIVE and listening
🎵 Audio streaming started - speaking into microphone...
🎤 Audio streaming active
```

### STT Monitor Display:
```
[timestamp] Listening: hello
[timestamp] Listening: hello world
[timestamp] Captured: hello world
[timestamp] Dispatched: hello world → reasoning core
```

### LLM Response:
- Status changes to "Thinking"
- Response appears in chat
- Audio response plays (if TTS enabled)

## Additional Notes

- **Browser Compatibility**: Works best in Chrome/Edge. Safari may have restrictions.
- **HTTPS Requirement**: Some browsers require HTTPS for microphone access. Use `localhost` for development.
- **Model Loading**: Vosk model must be in `modules/models/vosk-model-small-en-us-0.15/`
- **Performance**: First recognition may be slower as model initializes

## Next Steps

If you're still experiencing issues after following this guide:

1. **Check backend logs**: Look in `logs/nomous.log` for detailed error messages
2. **Verify model files**: Ensure all required models are downloaded and in correct paths
3. **Test WebSocket**: Use browser DevTools → Network → WS to see WebSocket messages
4. **Browser console**: Check for JavaScript errors in browser DevTools → Console

The system is now much more verbose and will tell you exactly where things are failing!
