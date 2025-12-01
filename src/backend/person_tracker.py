# Title: PersonTracker - Persistent person recognition and identity tracking
# Path: backend/person_tracker.py
# Purpose: Track distinct people, associate identities with conversations/behaviors, persist across sessions

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VisualSignature:
    """Visual characteristics used to identify a person."""
    # Position-based tracking (quadrant in frame)
    typical_position: str = ""  # "left", "center", "right"
    
    # Size-based estimation (rough height/build indicator)
    face_size_avg: float = 0.0  # Average face bounding box size
    face_size_samples: List[float] = field(default_factory=list)
    
    # Appearance notes (LLM-generated descriptions)
    hair_description: str = ""
    clothing_notes: List[str] = field(default_factory=list)
    distinguishing_features: List[str] = field(default_factory=list)
    
    # Future: actual face embeddings when available
    face_embedding: Optional[Any] = None
    
    def update_face_size(self, size: float):
        """Update running average of face size."""
        self.face_size_samples.append(size)
        if len(self.face_size_samples) > 20:
            self.face_size_samples = self.face_size_samples[-20:]
        self.face_size_avg = sum(self.face_size_samples) / len(self.face_size_samples)
    
    def similarity_score(self, other: "VisualSignature") -> float:
        """Calculate similarity between two visual signatures."""
        score = 0.0
        factors = 0
        
        # Face size comparison (if both have samples)
        if self.face_size_avg > 0 and other.face_size_avg > 0:
            size_diff = abs(self.face_size_avg - other.face_size_avg)
            max_size = max(self.face_size_avg, other.face_size_avg)
            size_similarity = 1.0 - min(size_diff / max_size, 1.0)
            score += size_similarity * 0.3
            factors += 0.3
        
        # Position similarity
        if self.typical_position and other.typical_position:
            if self.typical_position == other.typical_position:
                score += 0.2
            factors += 0.2
        
        # Description overlap
        if self.hair_description and other.hair_description:
            if self.hair_description.lower() == other.hair_description.lower():
                score += 0.3
            factors += 0.3
        
        # Distinguishing features overlap
        if self.distinguishing_features and other.distinguishing_features:
            my_features = set(f.lower() for f in self.distinguishing_features)
            other_features = set(f.lower() for f in other.distinguishing_features)
            if my_features & other_features:
                overlap = len(my_features & other_features) / max(len(my_features), len(other_features))
                score += overlap * 0.2
            factors += 0.2
        
        return score / factors if factors > 0 else 0.0


@dataclass 
class ConversationMemory:
    """A conversation or interaction associated with a person."""
    timestamp: float
    summary: str
    topics: List[str] = field(default_factory=list)
    sentiment: str = "neutral"  # positive, neutral, negative
    user_message: str = ""
    model_response: str = ""


@dataclass
class TrackedPerson:
    """A person being tracked with full identity persistence."""
    person_id: str
    
    # Identity
    name: Optional[str] = None  # Given name (e.g., "Joe")
    nickname: Optional[str] = None  # Alternative name or description
    
    # Visual signature for recognition
    visual_signature: VisualSignature = field(default_factory=VisualSignature)
    
    # Timing
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    total_time_observed: float = 0.0  # Cumulative seconds observed
    session_count: int = 1  # Number of separate observation sessions
    
    # Presence
    is_present: bool = True
    presence_confidence: float = 1.0  # 0-1 confidence they're still here
    
    # Behavior & Interactions
    conversations: List[ConversationMemory] = field(default_factory=list)
    behaviors_observed: List[str] = field(default_factory=list)  # "waved", "smiled", "looked away"
    interests: List[str] = field(default_factory=list)  # Topics they've discussed
    
    # Relationship
    familiarity_score: float = 0.0  # 0-1, increases with interactions
    last_interaction_topic: str = ""
    asked_for_name: bool = False  # Whether we've asked their name
    
    # Notes from the model
    model_notes: List[str] = field(default_factory=list)
    
    def update_sighting(self, face_rect: Optional[Tuple[int, int, int, int]] = None, 
                       position: str = "", frame_width: int = 0):
        """Update when person is seen in frame."""
        now = time.time()
        
        # Update timing
        if self.is_present:
            self.total_time_observed += now - self.last_seen
        else:
            # Was absent, now back
            self.session_count += 1
        
        self.last_seen = now
        self.is_present = True
        self.presence_confidence = 1.0
        
        # Update visual signature
        if face_rect:
            x, y, w, h = face_rect
            face_area = w * h
            self.visual_signature.update_face_size(face_area)
            
            # Determine position in frame
            if frame_width > 0:
                center_x = x + w // 2
                if center_x < frame_width * 0.33:
                    self.visual_signature.typical_position = "left"
                elif center_x > frame_width * 0.67:
                    self.visual_signature.typical_position = "right"
                else:
                    self.visual_signature.typical_position = "center"
        
        # Update familiarity
        self.familiarity_score = min(1.0, self.familiarity_score + 0.01)
    
    def decay_presence(self, seconds_elapsed: float):
        """Reduce presence confidence over time when not seen."""
        decay_rate = 0.1  # 10% per second
        self.presence_confidence = max(0.0, self.presence_confidence - (decay_rate * seconds_elapsed))
        if self.presence_confidence < 0.3:
            self.is_present = False
    
    def add_conversation(self, user_msg: str, model_response: str, topics: List[str] = None):
        """Record a conversation with this person."""
        self.conversations.append(ConversationMemory(
            timestamp=time.time(),
            summary=f"User: {user_msg[:100]}... Model: {model_response[:100]}...",
            topics=topics or [],
            user_message=user_msg,
            model_response=model_response
        ))
        
        # Update interests based on topics
        if topics:
            for topic in topics:
                if topic not in self.interests:
                    self.interests.append(topic)
        
        # Increase familiarity significantly with conversation
        self.familiarity_score = min(1.0, self.familiarity_score + 0.1)
    
    def get_display_name(self) -> str:
        """Get the best available name for this person."""
        if self.name:
            return self.name
        if self.nickname:
            return self.nickname
        
        # Generate description-based identifier
        sig = self.visual_signature
        if sig.hair_description:
            return f"person with {sig.hair_description}"
        if sig.typical_position:
            return f"person on the {sig.typical_position}"
        
        return f"person #{self.person_id.split('_')[-1]}"
    
    def get_conversation_history_summary(self) -> str:
        """Get a summary of conversation history with this person."""
        if not self.conversations:
            return "No previous conversations"
        
        recent = self.conversations[-5:]  # Last 5 conversations
        summaries = []
        for conv in recent:
            time_ago = time.time() - conv.timestamp
            if time_ago < 60:
                time_str = "just now"
            elif time_ago < 3600:
                time_str = f"{int(time_ago / 60)} min ago"
            else:
                time_str = f"{int(time_ago / 3600)} hours ago"
            
            topic_str = f" about {', '.join(conv.topics)}" if conv.topics else ""
            summaries.append(f"- {time_str}: talked{topic_str}")
        
        return "\n".join(summaries)
    
    def should_ask_name(self) -> bool:
        """Determine if we should ask this person their name."""
        if self.name:
            return False  # Already have name
        if self.asked_for_name:
            return False  # Already asked
        if self.familiarity_score < 0.3:
            return False  # Not familiar enough
        if len(self.conversations) < 2:
            return False  # Not enough interaction
        return True
    
    def to_memory_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for memory storage."""
        return {
            "person_id": self.person_id,
            "name": self.name,
            "nickname": self.nickname,
            "physical_description": self.visual_signature.hair_description,
            "distinguishing_features": self.visual_signature.distinguishing_features,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "total_time_observed": self.total_time_observed,
            "session_count": self.session_count,
            "familiarity_score": self.familiarity_score,
            "interests": self.interests,
            "conversation_count": len(self.conversations),
            "behaviors": self.behaviors_observed,
            "model_notes": self.model_notes,
            "asked_for_name": self.asked_for_name,
        }
    
    @classmethod
    def from_memory_dict(cls, data: Dict[str, Any]) -> "TrackedPerson":
        """Reconstruct from memory storage."""
        person = cls(
            person_id=data.get("person_id", f"person_{time.time()}"),
            name=data.get("name"),
            nickname=data.get("nickname"),
            first_seen=data.get("first_seen", time.time()),
            last_seen=data.get("last_seen", time.time()),
            total_time_observed=data.get("total_time_observed", 0),
            session_count=data.get("session_count", 1),
            is_present=False,  # Assume not present until confirmed
            familiarity_score=data.get("familiarity_score", 0),
            interests=data.get("interests", []),
            behaviors_observed=data.get("behaviors", []),
            model_notes=data.get("model_notes", []),
            asked_for_name=data.get("asked_for_name", False),
        )
        
        # Restore visual signature
        person.visual_signature.hair_description = data.get("physical_description", "")
        person.visual_signature.distinguishing_features = data.get("distinguishing_features", [])
        
        return person


class PersonTracker:
    """
    Tracks people across time and sessions with identity persistence.
    
    Key capabilities:
    - Distinguish different people using visual signatures
    - Associate conversations and behaviors with specific individuals
    - Persist identity knowledge across sessions via memory store
    - Enable the model to build relationships over time
    """
    
    def __init__(self, memory_store=None):
        self.memory = memory_store
        self.active_persons: Dict[str, TrackedPerson] = {}
        self.person_counter = 0
        self._lock = asyncio.Lock()
        
        # Current frame tracking
        self._current_faces: List[Tuple[int, int, int, int]] = []  # (x, y, w, h)
        self._frame_width = 0
        self._frame_height = 0
        
        # Conversation binding
        self._current_speaker: Optional[str] = None  # person_id of who we think is talking
        
        # Thresholds
        self.absence_threshold = 15.0  # Seconds before presence decays
        self.match_threshold = 0.5  # Minimum similarity to match existing person
        
        # State for announcements
        self._last_status = ""
        self._last_announcement_time = 0
        self._announcement_cooldown = 10.0
        
        logger.info("PersonTracker initialized with identity persistence")
    
    async def load_from_memory(self):
        """Load known persons from memory store."""
        if not self.memory:
            logger.debug("No memory store - skipping person load")
            return
        
        try:
            entities = await self.memory.get_entities(entity_type="person", limit=50)
            loaded = 0
            
            for entity in entities:
                props = entity.get("properties", {})
                person_data = {
                    "person_id": entity.get("id", f"person_{self.person_counter}"),
                    "name": entity.get("name"),
                    **props
                }
                
                person = TrackedPerson.from_memory_dict(person_data)
                self.active_persons[person.person_id] = person
                self.person_counter = max(self.person_counter, 
                                          int(person.person_id.split("_")[-1]) + 1 
                                          if "_" in person.person_id else self.person_counter)
                loaded += 1
            
            logger.info(f"Loaded {loaded} known persons from memory")
            
        except Exception as e:
            logger.error(f"Failed to load persons from memory: {e}")
    
    async def process_frame(self, faces: List[Tuple[int, int, int, int]], 
                           frame_width: int, frame_height: int) -> Dict[str, Any]:
        """
        Process detected faces from a video frame.
        
        Args:
            faces: List of (x, y, w, h) face bounding boxes
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            Dict with tracking results and any announcements
        """
        now = time.time()
        
        async with self._lock:
            self._current_faces = faces
            self._frame_width = frame_width
            self._frame_height = frame_height
            
            result = {
                "person_count": len(faces),
                "new_arrivals": [],
                "departures": [],
                "present": [],
                "status": "",
                "should_announce": False,
                "suggested_action": None,  # e.g., "ask_name"
            }
            
            # Match faces to existing tracked persons
            matched_persons: List[str] = []
            unmatched_faces: List[Tuple[int, int, int, int]] = []
            
            for face in faces:
                best_match = self._match_face_to_person(face, frame_width)
                if best_match:
                    best_match.update_sighting(face, frame_width=frame_width)
                    matched_persons.append(best_match.person_id)
                else:
                    unmatched_faces.append(face)
            
            # Create new persons for unmatched faces
            for face in unmatched_faces:
                self.person_counter += 1
                person_id = f"person_{self.person_counter}"
                
                new_person = TrackedPerson(person_id=person_id)
                new_person.update_sighting(face, frame_width=frame_width)
                
                self.active_persons[person_id] = new_person
                matched_persons.append(person_id)
                result["new_arrivals"].append(new_person.get_display_name())
                result["should_announce"] = True
            
            # Update presence for persons not seen
            for person_id, person in self.active_persons.items():
                if person_id not in matched_persons and person.is_present:
                    time_since_seen = now - person.last_seen
                    if time_since_seen > self.absence_threshold:
                        person.decay_presence(time_since_seen)
                        if not person.is_present:
                            result["departures"].append(person.get_display_name())
                            result["should_announce"] = True
            
            # Build present list
            for person_id in matched_persons:
                person = self.active_persons.get(person_id)
                if person:
                    result["present"].append({
                        "id": person_id,
                        "name": person.get_display_name(),
                        "familiarity": person.familiarity_score,
                        "has_name": person.name is not None,
                    })
            
            # Generate status
            result["status"] = self._generate_status()
            
            # Check if we should suggest asking for a name
            for person in self.active_persons.values():
                if person.is_present and person.should_ask_name():
                    result["suggested_action"] = {
                        "type": "ask_name",
                        "person_id": person.person_id,
                        "reason": f"Been interacting with {person.get_display_name()} but don't know their name"
                    }
                    break
            
            return result
    
    def _match_face_to_person(self, face: Tuple[int, int, int, int], 
                              frame_width: int) -> Optional[TrackedPerson]:
        """Try to match a face detection to an existing tracked person."""
        x, y, w, h = face
        face_area = w * h
        
        # Determine position
        center_x = x + w // 2
        if center_x < frame_width * 0.33:
            position = "left"
        elif center_x > frame_width * 0.67:
            position = "right"
        else:
            position = "center"
        
        # Create temporary signature for comparison
        temp_sig = VisualSignature(
            typical_position=position,
            face_size_avg=face_area
        )
        
        best_match = None
        best_score = 0.0
        
        # Only consider present or recently-present persons
        for person in self.active_persons.values():
            if not person.is_present and person.presence_confidence < 0.5:
                continue
            
            score = person.visual_signature.similarity_score(temp_sig)
            
            # Boost score if position matches and was recently seen
            if person.visual_signature.typical_position == position:
                time_since = time.time() - person.last_seen
                if time_since < 5:  # Within 5 seconds
                    score += 0.3
            
            if score > best_score and score >= self.match_threshold:
                best_score = score
                best_match = person
        
        return best_match
    
    def _generate_status(self) -> str:
        """Generate natural language status of tracked persons."""
        present = [p for p in self.active_persons.values() if p.is_present]
        
        if not present:
            return "No one is currently visible."
        
        parts = []
        
        # Named people first
        named = [p for p in present if p.name]
        unnamed = [p for p in present if not p.name]
        
        if named:
            names = [p.name for p in named]
            if len(names) == 1:
                person = named[0]
                familiarity = "familiar" if person.familiarity_score > 0.5 else ""
                parts.append(f"I see {names[0]}" + (f" ({familiarity})" if familiarity else ""))
            else:
                parts.append(f"I see {', '.join(names[:-1])} and {names[-1]}")
        
        if unnamed:
            if len(unnamed) == 1:
                person = unnamed[0]
                desc = person.get_display_name()
                if person.familiarity_score > 0.3:
                    parts.append(f"I also see {desc} who I've seen before")
                else:
                    parts.append(f"I see {desc}")
            else:
                parts.append(f"plus {len(unnamed)} other people")
        
        return ". ".join(parts) + "." if parts else ""
    
    async def bind_conversation(self, person_id: str, user_message: str, 
                                model_response: str, topics: List[str] = None):
        """Associate a conversation with a specific person."""
        async with self._lock:
            if person_id in self.active_persons:
                person = self.active_persons[person_id]
                person.add_conversation(user_message, model_response, topics)
                self._current_speaker = person_id
                
                # Persist to memory
                await self._save_person(person)
                
                logger.debug(f"Bound conversation to {person.get_display_name()}")
    
    async def set_person_name(self, person_id: str, name: str, 
                              description: str = "") -> Dict[str, Any]:
        """
        Set the name for a tracked person.
        
        This is called when we learn someone's name through conversation.
        """
        async with self._lock:
            if person_id not in self.active_persons:
                # Maybe they provided a description instead
                person = self._find_person_by_description(description)
                if person:
                    person_id = person.person_id
                else:
                    return {"success": False, "error": "Person not found"}
            
            person = self.active_persons[person_id]
            old_name = person.get_display_name()
            person.name = name
            
            if description:
                person.visual_signature.hair_description = description
            
            # Persist
            await self._save_person(person)
            
            logger.info(f"Named person: {old_name} -> {name}")
            
            return {
                "success": True,
                "person_id": person_id,
                "name": name,
                "message": f"I'll remember that {old_name} is {name}"
            }
    
    async def add_person_note(self, person_id: str, note: str) -> Dict[str, Any]:
        """Add a model-generated note about a person."""
        async with self._lock:
            if person_id not in self.active_persons:
                return {"success": False, "error": "Person not found"}
            
            person = self.active_persons[person_id]
            person.model_notes.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {note}")
            
            await self._save_person(person)
            
            return {"success": True, "person_id": person_id}
    
    async def describe_person(self, person_id: str, description: str) -> Dict[str, Any]:
        """
        Update the visual description for a person.
        
        Called when the model observes distinguishing features.
        """
        async with self._lock:
            if person_id not in self.active_persons:
                return {"success": False, "error": "Person not found"}
            
            person = self.active_persons[person_id]
            
            # Parse description for key features
            desc_lower = description.lower()
            
            # Hair
            hair_words = ["hair", "bald", "beard", "mustache"]
            for word in hair_words:
                if word in desc_lower:
                    person.visual_signature.hair_description = description
                    break
            
            # Add to distinguishing features
            if description not in person.visual_signature.distinguishing_features:
                person.visual_signature.distinguishing_features.append(description)
                # Keep only last 5
                person.visual_signature.distinguishing_features = \
                    person.visual_signature.distinguishing_features[-5:]
            
            await self._save_person(person)
            
            return {
                "success": True, 
                "person_id": person_id,
                "message": f"Noted: {person.get_display_name()} has {description}"
            }
    
    async def record_behavior(self, person_id: str, behavior: str) -> Dict[str, Any]:
        """Record an observed behavior for a person."""
        async with self._lock:
            if person_id not in self.active_persons:
                return {"success": False, "error": "Person not found"}
            
            person = self.active_persons[person_id]
            timestamp_behavior = f"{behavior} ({datetime.now().strftime('%H:%M')})"
            person.behaviors_observed.append(timestamp_behavior)
            
            # Keep last 20 behaviors
            person.behaviors_observed = person.behaviors_observed[-20:]
            
            await self._save_person(person)
            
            return {"success": True, "person_id": person_id}
    
    def _find_person_by_description(self, description: str) -> Optional[TrackedPerson]:
        """Find a person by description matching."""
        if not description:
            return None
        
        desc_lower = description.lower()
        
        for person in self.active_persons.values():
            # Check name
            if person.name and person.name.lower() in desc_lower:
                return person
            
            # Check physical description
            if person.visual_signature.hair_description:
                if person.visual_signature.hair_description.lower() in desc_lower:
                    return person
            
            # Check features
            for feature in person.visual_signature.distinguishing_features:
                if feature.lower() in desc_lower:
                    return person
        
        return None
    
    async def get_person_history(self, person_id: str = None, 
                                  name: str = None, 
                                  description: str = None) -> Dict[str, Any]:
        """
        Get the history and information about a specific person.
        
        Can look up by person_id, name, or description.
        """
        async with self._lock:
            person = None
            
            if person_id and person_id in self.active_persons:
                person = self.active_persons[person_id]
            elif name:
                for p in self.active_persons.values():
                    if p.name and p.name.lower() == name.lower():
                        person = p
                        break
            elif description:
                person = self._find_person_by_description(description)
            
            if not person:
                return {
                    "success": False,
                    "error": "Person not found",
                    "suggestion": "Try using their name or describing their appearance"
                }
            
            return {
                "success": True,
                "person_id": person.person_id,
                "name": person.name or "Unknown",
                "display_name": person.get_display_name(),
                "is_present": person.is_present,
                "familiarity": round(person.familiarity_score, 2),
                "first_seen": datetime.fromtimestamp(person.first_seen).isoformat(),
                "total_time_together": f"{int(person.total_time_observed / 60)} minutes",
                "session_count": person.session_count,
                "physical_description": person.visual_signature.hair_description,
                "distinguishing_features": person.visual_signature.distinguishing_features,
                "interests": person.interests,
                "conversation_count": len(person.conversations),
                "recent_conversations": person.get_conversation_history_summary(),
                "recent_behaviors": person.behaviors_observed[-5:],
                "notes": person.model_notes[-5:],
            }
    
    async def get_all_present(self) -> List[Dict[str, Any]]:
        """Get summary of all currently present people."""
        async with self._lock:
            present = []
            for person in self.active_persons.values():
                if person.is_present:
                    present.append({
                        "person_id": person.person_id,
                        "name": person.name,
                        "display_name": person.get_display_name(),
                        "familiarity": round(person.familiarity_score, 2),
                        "should_ask_name": person.should_ask_name(),
                    })
            return present
    
    async def get_current_speaker(self) -> Optional[Dict[str, Any]]:
        """Get information about who we think is currently speaking."""
        async with self._lock:
            if not self._current_speaker:
                # Default to the most familiar present person
                present = [p for p in self.active_persons.values() if p.is_present]
                if present:
                    present.sort(key=lambda p: p.familiarity_score, reverse=True)
                    self._current_speaker = present[0].person_id
            
            if self._current_speaker and self._current_speaker in self.active_persons:
                person = self.active_persons[self._current_speaker]
                return {
                    "person_id": person.person_id,
                    "name": person.name,
                    "display_name": person.get_display_name(),
                    "familiarity": person.familiarity_score,
                }
            return None
    
    async def _save_person(self, person: TrackedPerson):
        """Save person to memory store."""
        if not self.memory:
            return
        
        try:
            await self.memory.store_entity(
                entity_type="person",
                name=person.name or person.person_id,
                description=f"Tracked person: {person.get_display_name()}",
                properties=person.to_memory_dict()
            )
        except Exception as e:
            logger.error(f"Failed to save person {person.person_id}: {e}")
    
    def mark_asked_for_name(self, person_id: str):
        """Mark that we've asked a person for their name."""
        if person_id in self.active_persons:
            self.active_persons[person_id].asked_for_name = True
