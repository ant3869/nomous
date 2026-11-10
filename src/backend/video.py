# Title: CameraLoop - GPU-Optimized with Gesture Recognition
# Path: backend/video.py
# Purpose: Lightweight frame processing + MediaPipe Tasks gestures/faces for RTX 5070 Ti

import cv2
import threading
import time
import asyncio
import logging
from pathlib import Path
from queue import SimpleQueue, Empty
from typing import Optional
from .utils import to_data_url, msg_event, msg_image
from .person_tracker import PersonTracker
from .mediapipe_assets import MEDIAPIPE_MODELS, resolve_model  # noqa: F401  (MEDIAPIPE_MODELS re-exported)

logger = logging.getLogger(__name__)

try:
    import numpy as np
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - exercised via tests with monkeypatch
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False
    NUMPY_IMPORT_ERROR = exc
    logger.warning(
        "NumPy import failed (%s). Vision complexity metrics will run in a degraded mode.",
        exc,
    )
else:
    NUMPY_AVAILABLE = True
    NUMPY_IMPORT_ERROR = None

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python import BaseOptions as MpBaseOptions
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe Tasks available - gesture/face detection enabled")
except (ImportError, AttributeError):
    mp = None  # type: ignore[assignment]
    mp_vision = None  # type: ignore[assignment]
    MpBaseOptions = None  # type: ignore[assignment]
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available - install with: pip install mediapipe")

# Task model files live vendored in assets/mediapipe/ (manifest + checksums
# in src/backend/mediapipe_assets.py, with a pinned-URL download fallback
# into modules/models/mediapipe/).

# GestureRecognizer category -> Nomous gesture label.
GESTURE_LABELS = {
    "Thumb_Up": "thumbs_up",
    "Victory": "peace_sign",
    "Open_Palm": "open_hand",
    "Pointing_Up": "pointing",
    "Closed_Fist": "closed_fist",
    "ILoveYou": "rock_on",
}


class CameraLoop:
    def __init__(self, cfg, bridge, loop):
        """Initialize GPU-optimized camera loop."""
        self.bridge = bridge
        self.loop = loop
        self.enabled = bool(cfg["ui"]["vision_enabled"])

        self._numpy_available = NUMPY_AVAILABLE
        self._numpy_error = NUMPY_IMPORT_ERROR
        self._numpy_warning_sent = False
        if not self._numpy_available:
            logger.warning(
                "CameraLoop will continue without NumPy. Gesture and motion detection remain active, "
                "but vision descriptions may be less detailed."
            )

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
        
        self.preview_interval_sec = float(cfg["ui"].get("preview_interval_sec", 0.5))
        self.motion_threshold = int(cfg["ui"]["motion_sensitivity"])

        self._thr = None
        self._stop = threading.Event()
        self._running = False
        self._command_queue = SimpleQueue()
        self._brightness_scale = 1.0
        self._brightness_ema: Optional[float] = None

        # Vision analysis
        self.llm = None
        self.vision_description_enabled = True
        self.vision_model_path = None
        self.vision_frame_width = int(cfg.get("vision", {}).get("frame_width", 640) or 640)
        self.last_analysis_time = 0
        # Single source of truth (finding 5): vision.vision_cooldown.
        self.analysis_cooldown = float(cfg.get("vision", {}).get("vision_cooldown", 12))

        # Gesture detection
        self.gesture_enabled = bool(cfg["ui"].get("gesture_enabled", True))
        self.last_gesture_time = 0
        self.gesture_cooldown = float(cfg["ui"].get("gesture_cooldown", 3))

        # MediaPipe Tasks detectors (gestures + faces); None when unavailable.
        # Face detection is independent of the gesture toggle: person tracking
        # needs faces even when hand gestures are off.
        self._gesture_recognizer = None
        self._face_detector = None
        self._face_embedder = None
        if MEDIAPIPE_AVAILABLE:
            self._init_mediapipe_tasks()
            self._init_face_embedder()

        # Person tracking for persistent recognition
        self.person_tracker = PersonTracker(
            embedding_match_threshold=float(
                cfg.get("vision", {}).get("embedding_match_threshold", 0.7)
            )
        )
        self._last_person_count = 0
        self._last_person_update = 0
        self._person_tracking_cooldown = 5.0  # Seconds between person tracking updates
        
        logger.info(f"Video optimization: {self.capture_width}x{self.capture_height} capture, "
                   f"{self.process_width}x{self.process_height} processing, "
                   f"frame_skip={self.frame_skip}")

    def _enqueue(self, name: str, *args):
        try:
            self._command_queue.put_nowait((name, args))
        except Exception as e:
            logger.error(f"Failed to queue camera command {name}: {e}")

    def _init_mediapipe_tasks(self):
        """Build the gesture/face detectors from verified model files.

        Models resolve from the vendored assets/mediapipe/ copies first
        (checksum-checked against the manifest), then the legacy download
        dir, then a pinned-URL download. A missing model disables only its
        own detector instead of taking the other one down with it.
        """
        gesture_path = None
        if self.gesture_enabled:
            gesture_path = resolve_model("gesture_recognizer.task")
            if gesture_path is None:
                logger.warning("Gesture recognizer model unavailable; gesture detection off")
        face_path = resolve_model("blaze_face_short_range.tflite")
        if face_path is None:
            logger.warning("Face detector model unavailable; face detection off")
        try:
            if gesture_path is not None:
                self._gesture_recognizer = mp_vision.GestureRecognizer.create_from_options(
                    mp_vision.GestureRecognizerOptions(
                        base_options=MpBaseOptions(model_asset_path=str(gesture_path)),
                        running_mode=mp_vision.RunningMode.IMAGE,
                        num_hands=2,
                    )
                )
            if face_path is not None:
                self._face_detector = mp_vision.FaceDetector.create_from_options(
                    mp_vision.FaceDetectorOptions(
                        base_options=MpBaseOptions(model_asset_path=str(face_path)),
                        running_mode=mp_vision.RunningMode.IMAGE,
                    )
                )
            logger.info("MediaPipe Tasks detectors initialized")
        except Exception as exc:
            logger.warning("MediaPipe Tasks init failed: %s", exc)
            self._gesture_recognizer = None
            self._face_detector = None

    def _init_face_embedder(self):
        """Build the ImageEmbedder used for real person recognition.

        MobileNetV3 (vendored) turns each detected face crop into a
        1024-dim L2-normalized embedding; the person tracker matches on
        cosine similarity instead of face size + screen position. Without
        it, recognition falls back to the old size/position heuristic.
        """
        model_path = resolve_model("mobilenet_v3_small.tflite")
        if model_path is None:
            logger.warning("Face embedder model unavailable; person recognition falls back to size/position")
            return
        try:
            self._face_embedder = mp_vision.ImageEmbedder.create_from_options(
                mp_vision.ImageEmbedderOptions(
                    base_options=MpBaseOptions(model_asset_path=str(model_path)),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    l2_normalize=True,
                    quantize=False,
                )
            )
            logger.info("Face embedder initialized")
        except Exception as exc:
            logger.warning("Face embedder init failed: %s", exc)
            self._face_embedder = None

    def _embed_face_crop(self, frame_rgb, face_rect):
        """Embed one detected face crop; None when the embedder is unavailable."""
        if self._face_embedder is None or not MEDIAPIPE_AVAILABLE:
            return None
        try:
            x, y, w, h = (int(v) for v in face_rect)
            height, width = frame_rgb.shape[:2]
            # Pad ~20% so the embedding sees the whole face, not a tight box.
            pad_x, pad_y = int(w * 0.2), int(h * 0.2)
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
            if x1 <= x0 or y1 <= y0:
                return None
            crop = frame_rgb[y0:y1, x0:x1]
            resized = cv2.resize(crop, (224, 224))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=resized)
            result = self._face_embedder.embed(mp_image)
            if not result.embeddings:
                return None
            embedding = result.embeddings[0]
            vector = getattr(embedding, "embedding", None)
            if vector is None:
                vector = getattr(embedding, "float_embedding", None)
            if vector is None:
                return None
            return [float(v) for v in vector]
        except Exception as exc:
            logger.debug("Face embedding failed: %s", exc)
            return None

    def set_llm(self, llm):
        """Set LLM reference for vision analysis."""
        self.llm = llm
        # Also set memory reference for person tracking
        if llm and hasattr(llm, 'memory') and llm.memory:
            self.person_tracker.memory = llm.memory
            asyncio.run_coroutine_threadsafe(
                self.person_tracker.load_from_memory(),
                self.loop
            )
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

        if not self._numpy_available and not self._numpy_warning_sent:
            self._numpy_warning_sent = True
            warning = (
                "⚠️ Vision detail analysis is reduced because NumPy could not be loaded. "
                "If you are running inside the NumPy source tree, leave that directory and restart Nomous."
            )
            if self._numpy_error is not None:
                logger.debug("NumPy import error: %s", self._numpy_error)
            logger.warning(warning)
            self._post_event(warning)

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

        for detector in (self._gesture_recognizer, self._face_detector, self._face_embedder):
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass
        self._gesture_recognizer = None
        self._face_detector = None
        self._face_embedder = None

        logger.info("CameraLoop stopped")

    def _detect_gesture(self, frame, frame_rgb) -> str:
        """Detect hand gestures using the MediaPipe Tasks GestureRecognizer."""
        if self._gesture_recognizer is None or not MEDIAPIPE_AVAILABLE:
            return ""

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._gesture_recognizer.recognize(mp_image)
        except Exception as exc:
            logger.debug("Gesture detection error: %s", exc)
            return ""

        for hand_gestures in result.gestures or []:
            if not hand_gestures:
                continue
            best = max(hand_gestures, key=lambda g: g.score)
            label = GESTURE_LABELS.get(best.category_name)
            if label:
                return label
        return ""

    def _detect_faces(self, frame_rgb) -> list:
        """Detect faces with the MediaPipe Tasks FaceDetector; [] when unavailable."""
        if self._face_detector is None or not MEDIAPIPE_AVAILABLE:
            return []

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._face_detector.detect(mp_image)
        except Exception as exc:
            logger.debug("Face detection error: %s", exc)
            return []

        faces = []
        for detection in result.detections or []:
            box = detection.bounding_box
            if box is None:
                continue
            faces.append((int(box.origin_x), int(box.origin_y), int(box.width), int(box.height)))
        return faces

    def _analyze_complexity(self, gray):
        """Analyze visual complexity using edge detection."""
        edges = cv2.Canny(gray, 100, 200)
        edge_density = self._compute_edge_density(edges)

        if edge_density > 0.3:
            return "very detailed"
        elif edge_density > 0.2:
            return "detailed"
        elif edge_density > 0.1:
            return "moderately detailed"
        else:
            return "simple"

    def _compute_edge_density(self, edges) -> float:
        """Compute edge density with graceful degradation when NumPy is unavailable."""
        def _get_total_pixels(edges) -> int:
            total_pixels = int(getattr(edges, "size", 0))
            if total_pixels <= 0:
                shape = getattr(edges, "shape", None)
                if shape and len(shape) >= 2:
                    total_pixels = int(shape[0] * shape[1])
            return max(total_pixels, 1)

        if self._numpy_available and np is not None:
            try:
                total_pixels = _get_total_pixels(edges)
                non_zero = int(np.count_nonzero(edges))
                return non_zero / float(total_pixels)
            except Exception as exc:
                logger.debug("Falling back to OpenCV edge density calculation: %s", exc)

        try:
            total_pixels = _get_total_pixels(edges)
            non_zero = int(cv2.countNonZero(edges))
            return non_zero / float(total_pixels)
        except Exception as exc:
            logger.debug("Unable to compute edge density without NumPy: %s", exc)

        try:
            buffer = memoryview(edges)
            non_zero = sum(1 for value in buffer if value)
            total_pixels = len(buffer) or 1
            return non_zero / float(total_pixels)
        except Exception:
            logger.debug("Edge density fallbacks failed", exc_info=True)

        return 0.0

    def _describe_frame(self, frame, motion_level: int, gesture: str = "") -> str:
        """Generate detailed description of the frame."""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Estimate brightness using HSV value channel for more robust lighting detection.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        value_channel = hsv[:, :, 2]

        if self._numpy_available and np is not None:
            brightness_sample = float(value_channel.mean())
        else:
            brightness_sample = float(cv2.mean(value_channel)[0])

        # Smooth readings to avoid sudden jumps when exposure changes frame-to-frame.
        ema = self._brightness_ema
        if ema is None:
            ema = brightness_sample
        else:
            alpha = 0.18  # slow updates to better represent ambient light
            ema = (1 - alpha) * ema + alpha * brightness_sample
        self._brightness_ema = ema

        brightness = ema
        logger.debug(
            "Frame brightness raw=%.1f ema=%.1f used=%.1f",
            brightness_sample,
            ema,
            brightness,
        )

        if brightness < 45:
            light = "very dark"
        elif brightness < 75:
            light = "dark"
        elif brightness < 115:
            light = "dimly lit"
        elif brightness < 155:
            light = "moderately lit"
        elif brightness < 195:
            light = "well lit"
        elif brightness < 230:
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
        
        # Face detection with person tracking
        face_count = 0
        face_info = ""
        person_tracking_update = None
        faces_list = []

        if self._face_detector is not None:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self._detect_faces(frame_rgb)
                face_count = len(faces)

                # Convert faces to list of tuples for person tracker
                faces_list = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
                # One embedding per face: the tracker's real identity signal.
                face_embeddings = [self._embed_face_crop(frame_rgb, rect) for rect in faces_list]

                # Update person tracker if count changed or enough time passed
                now = time.time()
                should_update_tracking = (
                    face_count != self._last_person_count or
                    now - self._last_person_update > self._person_tracking_cooldown
                )

                if should_update_tracking:
                    self._last_person_count = face_count
                    self._last_person_update = now

                    # Get tracking status using new process_frame API
                    try:
                        future = asyncio.run_coroutine_threadsafe(
                            self.person_tracker.process_frame(
                                faces_list,
                                self.process_width,
                                self.process_height,
                                face_embeddings=face_embeddings
                            ),
                            self.loop
                        )
                        person_tracking_update = future.result(timeout=0.5)
                    except Exception as e:
                        logger.debug(f"Person tracking update failed: {e}")

                # Generate face info based on tracking status
                if person_tracking_update and person_tracking_update.get("status"):
                    face_info = person_tracking_update["status"]
                elif face_count > 0:
                    face_info = f"I see {face_count} {'person' if face_count == 1 else 'people'}"

            except Exception as e:
                logger.debug(f"Face detection error: {e}")
        
        complexity = self._analyze_complexity(gray)

        # Construct description - prioritize meaningful changes
        description_bits = []
        
        # Check if there's an interesting change to report (new API uses "new_arrivals" and "departures")
        has_arrival = person_tracking_update and person_tracking_update.get("new_arrivals")
        has_departure = person_tracking_update and person_tracking_update.get("departures")
        should_announce = person_tracking_update and person_tracking_update.get("should_announce")

        if has_arrival and should_announce:
            arrivals = person_tracking_update["new_arrivals"]  # Now a list of display names
            if len(arrivals) == 1:
                name = arrivals[0]
                if name and not name.startswith("person #"):
                    description_bits.append(f"{name} has arrived")
                else:
                    description_bits.append("Someone new has entered the scene")
            else:
                description_bits.append(f"{len(arrivals)} new people have arrived")
        elif has_departure and should_announce:
            departures = person_tracking_update["departures"]  # Now a list of display names
            if len(departures) == 1:
                name = departures[0]
                if name and not name.startswith("person #"):
                    description_bits.append(f"{name} has left")
                else:
                    description_bits.append("Someone has left the scene")
            else:
                description_bits.append(f"{len(departures)} people have left")
        elif face_info:
            description_bits.append(face_info)
            description_bits.append(f"in a {light} environment")
        else:
            description_bits.append(f"The space looks {light}")
            if complexity in {"very detailed", "detailed"}:
                description_bits.append(f"with {complexity} visuals")

        if motion_level > 5:
            description_bits.append(activity)

        description = ", ".join(description_bits)
        if not description:
            description = f"The scene looks {light}."
        elif not description.endswith("."):
            description += "."

        if gesture:
            gesture_text = {
                "peace_sign": "a peace sign",
                "thumbs_up": "a thumbs up",
                "open_hand": "an open hand",
                "pointing": "a pointing motion",
                "closed_fist": "a closed fist",
                "rock_on": "a rock-on gesture"
            }.get(gesture, f"a {gesture} gesture")
            description += f" I also notice {gesture_text}."

        return description

    def _should_send_preview(self, now: float, last_preview_send: float, gesture: str) -> bool:
        """Preview cadence: every preview_interval_sec, or immediately on a gesture.

        Motion plays no part — in a still room the dashboard would
        otherwise freeze on one frame for seconds at a time.
        """
        return bool(gesture) or (now - last_preview_send) >= self.preview_interval_sec

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
            
            # Enable auto-exposure for better lighting adaptation
            try:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Enable auto-exposure
            except Exception as e:
                logger.debug(f"Could not enable auto-exposure: {e}")
            
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
                
                # Set a reasonable default brightness (60% of max) if too low
                if native_brightness is not None and native_brightness < (0.5 * self._brightness_scale):
                    default_brightness = 0.6 * self._brightness_scale
                    cap.set(cv2.CAP_PROP_BRIGHTNESS, default_brightness)
                    logger.info(f"Camera brightness adjusted to {default_brightness:.1f} (60% of scale)")
                    
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
            
            last_preview_send = 0.0
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
                
                # Dashboard preview: steady cadence (~2 fps) regardless of
                # motion, so the UI shows a live feed instead of a still
                # frame in a quiet room. A gesture forces an immediate
                # preview. Analysis is NOT gated on the preview send.
                if self._should_send_preview(now, last_preview_send, gesture):
                    last_preview_send = now
                    try:
                        data_url = to_data_url(frame)
                        asyncio.run_coroutine_threadsafe(
                            self.bridge.post(msg_image(data_url)),
                            self.loop
                        )
                    except Exception as e:
                        logger.error(f"Error posting preview frame: {e}")

                # Vision analysis stays on its own cooldown (or a gesture),
                # independent of the preview cadence.
                if (
                    self.vision_description_enabled
                    and self.llm
                    and ((now - self.last_analysis_time) >= self.analysis_cooldown or gesture)
                ):
                    self.last_analysis_time = now
                    try:
                        description = self._describe_frame(frame_small, motion, gesture)
                        logger.info(f"Vision: {description}")

                        # Send the actual frame to the vision model when it can take one.
                        image_data_url = None
                        if getattr(self.llm, "_vision_available", False):
                            vision_frame = frame_small
                            if vision_frame.shape[1] > self.vision_frame_width:
                                scale = self.vision_frame_width / vision_frame.shape[1]
                                vision_frame = cv2.resize(
                                    vision_frame,
                                    (self.vision_frame_width, int(vision_frame.shape[0] * scale)),
                                )
                            try:
                                image_data_url = to_data_url(vision_frame)
                            except Exception as exc:
                                logger.debug("Could not encode vision frame: %s", exc)

                        # Schedule vision processing in LLM
                        asyncio.run_coroutine_threadsafe(
                            self.llm.process_vision(description, image_data_url),
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
