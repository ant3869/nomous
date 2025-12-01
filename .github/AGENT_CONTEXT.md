# Nomous Agent Context

This document provides comprehensive context for AI coding agents working on the Nomous project. It describes the architecture, recent changes, and key systems.

## Project Overview

**Nomous** is an offline-first autonomous AI assistant with:
- Local LLM via llama.cpp (GGUF models)
- Speech-to-text via Vosk
- Text-to-speech via Piper
- Computer vision via OpenCV + MediaPipe
- Persistent memory with SQLite + vector embeddings
- **Person identity tracking** for relationship building

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     React Frontend                          │
│  (App.tsx, components/, Tailwind CSS, Framer Motion)       │
└─────────────────────────┬───────────────────────────────────┘
                          │ WebSocket
┌─────────────────────────┴───────────────────────────────────┐
│                   Python Backend                            │
│  scripts/run_bridge.py - Main WebSocket server              │
├─────────────────────────────────────────────────────────────┤
│  Core Modules:                                              │
│  ├── llm.py        - LocalLLM with tools & autonomous mode │
│  ├── video.py      - CameraLoop with face detection        │
│  ├── audio.py      - STT engine (Vosk)                     │
│  ├── tts.py        - TTS engine (Piper)                    │
│  ├── memory.py     - MemoryStore with RAG                  │
│  ├── tools.py      - ToolExecutor with 25+ tools           │
│  ├── person_tracker.py - Identity persistence system       │
│  └── handlers.py   - Message routing                        │
└─────────────────────────────────────────────────────────────┘
```

## Key Files

### Backend (src/backend/)

| File | Purpose | Lines |
|------|---------|-------|
| `llm.py` | Local LLM with autonomous thinking, vision processing, tool calling | ~1260 |
| `video.py` | Camera capture, face detection, gesture recognition, person tracking | ~660 |
| `tools.py` | 25+ tools for memory, learning, social interaction, person tracking | ~1700 |
| `person_tracker.py` | Person identity persistence across sessions | ~620 |
| `memory.py` | SQLite + vector embeddings for RAG | ~800 |
| `audio.py` | Vosk STT integration | ~300 |
| `tts.py` | Piper TTS integration | ~400 |

### Frontend (src/frontend/)

| File | Purpose |
|------|---------|
| `App.tsx` | Main dashboard (~4700 lines) |
| `components/GenerationProgress.tsx` | Token generation progress bar |
| `components/ConversationMessage.tsx` | Chat message bubbles |

### Entry Points

| File | Purpose |
|------|---------|
| `run_nomous.py` | One-command launcher |
| `scripts/start.py` | Bootstrap venv, install deps, start server |
| `scripts/run_bridge.py` | WebSocket server main loop |

## Person Identity Tracking System

### Overview
The system allows Nomous to:
1. **Distinguish different people** using face position, size, and visual descriptions
2. **Learn names** automatically when people introduce themselves
3. **Associate conversations** with specific individuals
4. **Build relationships** through familiarity scores and interaction history
5. **Persist identities** across sessions via memory store

### Key Classes

#### TrackedPerson (person_tracker.py)
```python
@dataclass
class TrackedPerson:
    person_id: str
    name: Optional[str]              # Given name ("Joe")
    visual_signature: VisualSignature  # Face position, size, features
    conversations: List[ConversationMemory]  # Chat history with this person
    behaviors_observed: List[str]    # "waved", "smiled", etc.
    interests: List[str]             # Topics from conversations
    familiarity_score: float         # 0-1, increases with interaction
    asked_for_name: bool             # Whether we've asked their name
```

#### VisualSignature
```python
@dataclass
class VisualSignature:
    typical_position: str       # "left", "center", "right"
    face_size_avg: float        # Running average of face bounding box
    hair_description: str       # "long dark hair"
    distinguishing_features: List[str]  # Visual notes
```

### Data Flow

```
Camera Frame
    │
    ▼
Face Detection (OpenCV Haar Cascade)
    │
    ▼
PersonTracker.process_frame(faces, width, height)
    │
    ├── Match faces to existing TrackedPerson by visual signature
    ├── Create new TrackedPerson for unmatched faces
    ├── Mark absent persons who aren't detected
    │
    ▼
LLM.process_vision(description)
    │
    ├── Gets person context from tracker
    ├── Includes names if known
    ├── Suggests asking names if familiarity > threshold
    │
    ▼
LLM Response (may use person tools)
```

### Person Tracking Tools

| Tool | Purpose |
|------|---------|
| `remember_person_name` | Store name when someone introduces themselves |
| `describe_person_appearance` | Record visual features for recognition |
| `recall_person` | Look up history with a person |
| `get_people_present` | See who is currently visible |
| `note_person_behavior` | Record actions/reactions |
| `add_person_note` | Add observations about someone |

### Automatic Name Detection
In `llm.py`, the `_extract_name_from_text()` method detects patterns like:
- "My name is Joe"
- "I'm Anthony"
- "Call me Mike"

When detected, the name is automatically associated with the current speaker.

## Tool System

### Tool Categories
1. **Memory** - search_memory, recall_recent_context, summarize_recent_context
2. **Learning** - evaluate_interaction, identify_pattern, track_milestone
3. **Observation** - record_observation, remember_fact
4. **Social** - analyze_sentiment, check_appropriate_response, learn_preference
5. **Person Tracking** - remember_person_name, recall_person, get_people_present, etc.
6. **System** - list_available_tools, get_tool_usage_stats, get_current_capabilities

### Tool Execution Flow
```
LLM Output: "TOOL_CALL: {\"tool\": \"remember_person_name\", \"args\": {\"name\": \"Joe\"}}"
    │
    ▼
ToolExecutor.execute_tool("remember_person_name", {"name": "Joe"})
    │
    ├── Validate parameters
    ├── Call _remember_person_name()
    ├── Access self.llm.person_tracker
    │
    ▼
PersonTracker.set_person_name(person_id, "Joe")
    │
    ▼
Result sent to frontend via WebSocket
```

## Recent Changes (December 2025)

### Person Identity System
- Created comprehensive `PersonTracker` class with visual signatures
- Added 6 person tracking tools to `tools.py`
- Integrated tracker with LLM for conversation binding
- Auto-detection of name introductions in speech

### Vision Improvements
- Enhanced `process_vision()` with person context
- Improved duplicate description filtering
- Added familiarity-based announcements

### Bug Fixes
- Fixed venv creation with `system_site_packages=False` (was causing numpy/CUDA conflicts)
- Fixed duplicate `finally` block in `llm.py`
- Filtered token processing messages from Think tab display

## Configuration

### config.yaml Key Settings
```yaml
llm:
  tools_enabled: true
  temperature: 0.7
  max_tokens: 256

camera:
  index: 0
  width: 1280
  height: 720
  process_width: 640
  process_height: 360

ui:
  vision_enabled: true
  vision_cooldown: 12
  gesture_enabled: true
```

## Testing

```bash
# Syntax check all backend files
python -m py_compile src/backend/llm.py
python -m py_compile src/backend/person_tracker.py
python -m py_compile src/backend/tools.py

# Run tests
pytest tests/ -v

# Start application
python run_nomous.py
```

## Common Issues

### GPU/CUDA Not Loading
- Delete `.venv` folder and restart to recreate clean environment
- Ensure `system_site_packages=False` in `scripts/start.py`

### Repetitive Vision Announcements
- Check `_is_similar_vision()` in `llm.py`
- Adjust `vision_cooldown` in config

### Person Tracking Not Working
- Verify `person_tracker` is connected in `run_bridge.py`
- Check face detection cascade is loading
- Ensure memory store is enabled

## WebSocket Message Types

| Type | Direction | Purpose |
|------|-----------|---------|
| `speak` | Server→Client | Text for TTS and chat |
| `thought` | Server→Client | Internal reasoning display |
| `tool_result` | Server→Client | Tool execution results |
| `vision` | Server→Client | Camera frame updates |
| `status` | Server→Client | System status (idle, thinking) |
| `audio` | Client→Server | PCM audio for STT |
| `toggle` | Client→Server | Enable/disable features |
| `param` | Client→Server | Update configuration |

## Code Style

- **Python**: Type hints, async/await, dataclasses
- **TypeScript**: Strict mode, interfaces for messages
- **Naming**: snake_case (Python), camelCase (TypeScript)
- **Logging**: Use module-level `logger = logging.getLogger(__name__)`
