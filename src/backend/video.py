# Title: CameraLoop - GPU-Optimized with Gesture Recognition
# Path: backend/video.py
# Purpose: Lightweight frame processing + MediaPipe hand gestures for RTX 2080 Ti

import cv2
import threading
import time
import asyncio
import logging
import numpy as np
from pathlib import Path
from queue import SimpleQueue, Empty
from .utils import to_data_url, msg_event, msg_image

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe available - gesture detection enabled")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available - install with: pip install mediapipe")


class CameraLoop:
    def __init__(self, cfg, bridge, loop):
        """Initialize GPU-optimized camera loop."""
        self.bridge = bridge
        self.loop = loop
        self.enabled = bool(cfg["ui"]["vision_enabled"])
        
        cam = cfg["camera"]
        self.index = int(cam["index"])
        self.backend = cam.get("backend", "dshow")
        
        # Capture at full resolution
        self.capture_width = int(cam["width"])
        self.capture_height = int(cam["height"])
        
        # Process at lower resolution for speed
        self.process_width = int(cam.get("process_width", 640))
        self.process_height = int(cam.get("process_height", 360))
        self.frame_skip = int(cam.get("frame_skip", 2))
        
        self.debounce_sec = float(cfg["ui"]["snapshot_debounce"])
        self.motion_threshold = int(cfg["ui"]["motion_sensitivity"])

        self._thr = None
        self._stop = threading.Event()
        self._running = False
        self._command_queue = SimpleQueue()
        self._brightness_scale = 1.0

        # Vision analysis
        self.llm = None
        self.vision_description_enabled = True
        self.vision_model_path = None
        self.last_analysis_time = 0
        self.analysis_cooldown = float(cfg["ui"].get("vision_cooldown", 12))
        
        # Gesture detection
        self.gesture_enabled = bool(cfg["ui"].get("gesture_enabled", True))
        self.last_gesture_time = 0
        self.gesture_cooldown = float(cfg["ui"].get("gesture_cooldown", 3))
        
        # Initialize MediaPipe
        if MEDIAPIPE_AVAILABLE and self.gesture_enabled:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe hand detection initialized")
        else:
            self.hands = None
        
        # Face detection
        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("Face detection initialized")
        except Exception as e:
            logger.warning(f"Could not load face detection: {e}")
            self.face_cascade = None
        
        logger.info(f"Video optimization: {self.capture_width}x{self.capture_height} capture, "
                   f"{self.process_width}x{self.process_height} processing, "
                   f"frame_skip={self.frame_skip}")

    def _enqueue(self, name: str, *args):
        try:
            self._command_queue.put_nowait((name, args))
        except Exception as e:
            logger.error(f"Failed to queue camera command {name}: {e}")

    def set_llm(self, llm):
        """Set LLM reference for vision analysis."""
        self.llm = llm
        logger.info("LLM reference set for vision analysis")

    def set_enabled(self, value: bool):
        self.enabled = bool(value)
        logger.info(f"Camera {'enabled' if self.enabled else 'disabled'}")

    def set_resolution(self, width: int, height: int):
        self.capture_width = int(width)
        self.capture_height = int(height)
        self.process_width = max(160, self.capture_width // 2)
        self.process_height = max(120, self.capture_height // 2)
        self._enqueue("resolution", self.capture_width, self.capture_height)
        self._post_event(f"Camera resolution → {self.capture_width}x{self.capture_height}")

    def set_exposure(self, percent: int):
        percent = max(0, min(100, int(percent)))
        self._enqueue("exposure", percent)
        self._post_event(f"Camera exposure → {percent}%")

    def set_brightness(self, percent: int):
        percent = max(0, min(100, int(percent)))
        self._enqueue("brightness", percent)
        self._post_event(f"Camera brightness → {percent}%")

    def set_vision_model_path(self, path: str):
        self.vision_model_path = path
        self._post_event(f"Vision model → {Path(path).name if Path(path).exists() else path}")

    def start(self):
        """Start the camera capture thread."""
        if self._running:
            logger.warning("CameraLoop already running")
            return
        
        self._stop.clear()
        self._running = True
        self._thr = threading.Thread(target=self._run, daemon=True, name="CameraLoop")
        self._thr.start()
        logger.info("CameraLoop started")

    def stop(self):
        """Stop the camera capture thread gracefully."""
        if not self._running:
            return
        
        logger.info("Stopping CameraLoop...")
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=2)
        self._running = False
        
        if self.hands:
            self.hands.close()
        
        logger.info("CameraLoop stopped")

    def _detect_gesture(self, frame, frame_rgb) -> str:
        """Detect hand gestures using MediaPipe."""
        if not self.hands:
            return ""
        
        try:
            results = self.hands.process(frame_rgb)
            
            if not results.multi_hand_landmarks:
                return ""
            
            # Analyze first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get key landmarks
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
            
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            
            # Calculate finger heights relative to wrist
            thumb_up = thumb_tip.y < wrist.y
            index_up = index_tip.y < wrist.y - 0.1
            middle_up = middle_tip.y < wrist.y - 0.1
            ring_up = ring_tip.y < wrist.y - 0.1
            pinky_up = pinky_tip.y < wrist.y - 0.1
            
            # Count extended fingers
            fingers_up = sum([index_up, middle_up, ring_up, pinky_up])
            
            # Detect gestures
            if fingers_up >= 4:
                return "waving" if thumb_up else "open_hand"
            elif index_up and middle_up and not ring_up and not pinky_up:
                return "peace_sign"
            elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
                return "thumbs_up"
            elif index_up and not middle_up and not ring_up and not pinky_up:
                return "pointing"
            
            return ""
            
        except Exception as e:
            logger.debug(f"Gesture detection error: {e}")
            return ""

    def _analyze_complexity(self, gray):
        """Analyze visual complexity using edge detection."""
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.3:
            return "very detailed"
        elif edge_density > 0.2:
            return "detailed"
        elif edge_density > 0.1:
            return "moderately detailed"
        else:
            return "simple"

    def _describe_frame(self, frame, motion_level: int, gesture: str = "") -> str:
        """Generate detailed description of the frame."""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Brightness
        brightness = gray.mean()
        if brightness < 70:
            light = "dark"
        elif brightness < 140:
            light = "moderately lit"
        elif brightness < 220:
            light = "bright"
        else:
            light = "very bright"
        
        # Motion
        if motion_level > 40:
            activity = "with noticeable movement"
        elif motion_level > 20:
            activity = "with some movement"
        elif motion_level > 5:
            activity = "with slight movement"
        else:
            activity = "very still"
        
        # Face detection
        face_info = ""
        if self.face_cascade is not None:
            try:
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                if len(faces) > 0:
                    face_info = f"I see {len(faces)} {'person' if len(faces) == 1 else 'people'}"
            except Exception as e:
                logger.debug(f"Face detection error: {e}")
        
        # Construct description
        description_bits = []

        if face_info:
            description_bits.append(face_info)
            description_bits.append(f"in a {light} environment")
        else:
            description_bits.append(f"The space looks {light}")

        if motion_level > 5:
            description_bits.append(activity)

        description = ", ".join(description_bits)
        if not description:
            description = f"The scene looks {light}."
        elif not description.endswith("."):
            description += "."

        if gesture:
            gesture_text = {
                "waving": "a quick wave",
                "peace_sign": "a peace sign",
                "thumbs_up": "a thumbs up",
                "open_hand": "an open hand",
                "pointing": "a pointing motion"
            }.get(gesture, f"a {gesture} gesture")
            description += f" I also notice {gesture_text}."

        return description

    def _run(self):
        """Main camera capture loop (runs in separate thread)."""
        cap = None
        try:
            # Initialize camera
            backend_id = cv2.CAP_DSHOW if self.backend == 'dshow' else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.index, backend_id)

            if not cap.isOpened():
                logger.error(f"Failed to open camera at index {self.index}")
                self._post_event(f"Camera {self.index} failed to open")
                return
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info(f"Camera opened: {self.capture_width}x{self.capture_height}, backend={self.backend}")

            # Determine native brightness scaling so slider percentages map correctly
            try:
                native_brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
                if self.backend == 'dshow' or (native_brightness is not None and native_brightness > 1.0):
                    # DirectShow and similar backends typically use a 0-255 range
                    self._brightness_scale = 255.0
                else:
                    self._brightness_scale = 1.0
                logger.info(
                    f"Camera brightness scale set to {self._brightness_scale:.2f} "
                    f"(backend={self.backend}, native={native_brightness})"
                )
            except Exception as e:
                logger.debug(f"Unable to determine brightness scale: {e}")
                self._brightness_scale = 1.0

            # Read first frame
            ok, prev = cap.read()
            if not ok:
                logger.error("Failed to read initial frame from camera")
                self._post_event("Failed to read from camera")
                return
            
            # Resize for processing
            prev_small = cv2.resize(prev, (self.process_width, self.process_height))
            
            last_frame_send = 0.0
            frame_count = 0
            
            # Main capture loop
            while not self._stop.is_set():
                # Apply pending control commands
                while True:
                    try:
                        cmd, args = self._command_queue.get_nowait()
                    except Empty:
                        break
                    try:
                        if cmd == "resolution":
                            w, h = args
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
                            logger.info(f"Camera resolution updated to {w}x{h}")
                        elif cmd == "exposure":
                            (percent,) = args
                            exposure = -13 + (percent / 100.0) * 12  # map 0-100 -> [-13,-1]
                            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                            logger.info(f"Camera exposure set to {exposure:.2f}")
                        elif cmd == "brightness":
                            (percent,) = args
                            scale = self._brightness_scale if self._brightness_scale > 0 else 1.0
                            brightness = (percent / 100.0) * scale
                            cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
                            logger.info(
                                f"Camera brightness set to {brightness:.2f} "
                                f"(percent={percent}, scale={scale:.2f})"
                            )
                    except Exception as e:
                        logger.error(f"Failed to apply camera command {cmd}: {e}")

                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.03)
                    continue
                
                frame_count += 1
                
                # Frame skipping for performance
                if frame_count % (self.frame_skip + 1) != 0:
                    continue
                
                # Skip processing if vision is disabled
                if not self.enabled:
                    time.sleep(0.1)
                    continue
                
                # Resize for processing (much faster)
                frame_small = cv2.resize(frame, (self.process_width, self.process_height))
                
                # Motion detection on small frame
                try:
                    gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                    pgray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(gray, pgray)
                    motion = int(diff.mean() * 100 / 255)
                except Exception as e:
                    logger.error(f"Error in motion detection: {e}")
                    time.sleep(0.1)
                    continue
                
                now = time.time()
                
                # Gesture detection (lightweight, runs every frame)
                gesture = ""
                if self.gesture_enabled and (now - self.last_gesture_time) >= self.gesture_cooldown:
                    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
                    gesture = self._detect_gesture(frame_small, frame_rgb)
                    
                    if gesture:
                        self.last_gesture_time = now
                        logger.info(f"GESTURE: {gesture}")
                        self._post_event(f"👋 Gesture: {gesture}")
                
                # Send frame if motion detected or debounce time elapsed
                if motion >= self.motion_threshold or (now - last_frame_send) >= self.debounce_sec or gesture:
                    last_frame_send = now
                    
                    try:
                        # Send full resolution frame to UI
                        data_url = to_data_url(frame)
                        asyncio.run_coroutine_threadsafe(
                            self.bridge.post(msg_image(data_url)), 
                            self.loop
                        )
                        
                        # Trigger vision analysis if gesture detected or cooldown passed
                        should_analyze = (
                            self.vision_description_enabled and 
                            self.llm and 
                            ((now - self.last_analysis_time) >= self.analysis_cooldown or gesture)
                        )
                        
                        if should_analyze:
                            self.last_analysis_time = now
                            description = self._describe_frame(frame_small, motion, gesture)
                            logger.info(f"Vision: {description}")
                            
                            # Schedule vision processing in LLM
                            asyncio.run_coroutine_threadsafe(
                                self.llm.process_vision(description),
                                self.loop
                            )
                            
                    except Exception as e:
                        logger.error(f"Error posting frame: {e}")
                
                prev_small = frame_small
            
            logger.info(f"CameraLoop processed {frame_count} frames before stopping")
            
        except Exception as e:
            logger.error(f"Unexpected error in camera loop: {e}", exc_info=True)
            self._post_event(f"Camera error: {str(e)}")
        finally:
            if cap is not None:
                cap.release()
                logger.info("Camera released")
            self._running = False

    def _post_event(self, message):
        """Helper to post event messages from the thread."""
        try:
            asyncio.run_coroutine_threadsafe(
                self.bridge.post(msg_event(message)), 
                self.loop
            )
        except Exception as e:
            logger.error(f"Failed to post event: {e}")
