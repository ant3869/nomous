# Title: Nomous WS Bridge - Autonomous Edition
# Path: run_bridge.py
# Purpose: Start WS server with autonomous AI that thinks, sees, and speaks

import os
import sys
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional
import websockets
from websockets.server import WebSocketServerProtocol

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.backend.analytics import ConversationAnalytics
from src.backend.config import load_config
from src.backend.utils import Bridge, msg_status, msg_event
from src.backend.protocol import msg_entities, msg_timeline, msg_search_results
from src.backend.video import CameraLoop
from src.backend.audio import AudioSTT
from src.backend.dreaming import Dreamer, seconds_until_next_run
from src.backend.llm import LocalLLM
from src.backend.memory import MemoryStore
from src.backend.tts import PiperTTS
from src.backend.system import SystemMonitor

# Set up logging
log_dir = Path(project_root) / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "nomous.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

CFG = load_config()


class Server:
    def __init__(self):
        self.bridge = Bridge()
        self.clients: set[WebSocketServerProtocol] = set()
        self.llm: Optional[LocalLLM] = None
        self.stt: Optional[AudioSTT] = None
        self.tts: Optional[PiperTTS] = None
        self.cam: Optional[CameraLoop] = None
        self._autonomous_task: Optional[asyncio.Task] = None
        self._dreaming_task: Optional[asyncio.Task] = None
        self.dreamer = None
        self.memory: Optional[MemoryStore] = None
        self.system_monitor: Optional[SystemMonitor] = None
        self.analytics: Optional[ConversationAnalytics] = None
        self._audio_chunk_count = 0  # Track audio chunks received

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        value = float(size_bytes)
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if value < 1024.0 or unit == "TB":
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{value:.1f} TB"

    @staticmethod
    def _infer_model_type(name: str) -> str:
        lowered = name.lower()
        if any(keyword in lowered for keyword in ("vision", "vl", "llava", "clip", "see")):
            return "vision"
        if any(keyword in lowered for keyword in ("coder", "code", "codellama", "program")):
            return "coder"
        if any(keyword in lowered for keyword in ("reason", "think", "deepseek", "logic")):
            return "reasoning"
        if any(keyword in lowered for keyword in ("audio", "voice", "speech")):
            return "audio"
        if any(keyword in lowered for keyword in ("chat", "talk", "assistant", "conversation", "dialog")):
            return "conversation"
        return "other"

    async def handle_toggle(self, what: str, value: bool):
        if not what:
            logger.warning("Toggle without target")
            return
        handled = False
        if what == "tts" and self.tts:
            self.tts.set_enabled(value)
            handled = True
        elif what == "speaker" and self.tts:
            self.tts.set_auto_play(value)
            handled = True
        elif what == "vision" and self.cam:
            self.cam.set_enabled(value)
            handled = True
        elif what == "stt" and self.stt:
            self.stt.set_enabled(value)
            handled = True
        elif what == "autonomous" and self.llm:
            self.llm.autonomous_mode = value
            handled = True
        elif what == "dreaming" and self.dreamer:
            self.dreamer.enabled = value
            handled = True

        if handled:
            state = "enabled" if value else "disabled"
            await self.bridge.post(msg_event(f"{what} {state}"))
        else:
            logger.warning(f"Unknown toggle target: {what}")

    async def handle_param(self, key: str, value):
        if not key:
            logger.warning("Param without key")
            return
        if key == "tts_voice" and self.tts:
            await self.tts.set_voice(str(value))
            return
        if key == "audio_model_path" and self.tts:
            await self.tts.set_voice_path(str(value))
            return
        if key == "master_volume" and self.tts:
            self.tts.set_volume(int(value))
            await self.bridge.post(msg_event(f"master volume → {int(value)}%"))
            return
        if key == "mic_sensitivity" and self.stt:
            self.stt.set_sensitivity(int(value))
            await self.bridge.post(msg_event(f"mic sensitivity → {int(value)}%"))
            return
        if key == "stt_model_path" and self.stt:
            await self.stt.reload_model(str(value))
            await self.bridge.post(msg_event(f"STT model → {Path(str(value)).name}"))
            return
        if key == "camera_resolution" and self.cam:
            try:
                width, height = str(value).lower().split("x")
                self.cam.set_resolution(int(width), int(height))
            except Exception as e:
                logger.error(f"Invalid camera resolution '{value}': {e}")
            return
        if key == "camera_exposure" and self.cam:
            self.cam.set_exposure(int(value))
            return
        if key == "camera_brightness" and self.cam:
            self.cam.set_brightness(int(value))
            return
        if key == "vision_model_path" and self.llm:
            await self.llm.reload_vision_model(str(value))
            if self.cam:
                self.cam.set_vision_model_path(str(value))
            return
        if key == "llm_temperature" and self.llm:
            self.llm.update_sampling(temperature=float(value))
            await self.bridge.post(msg_event(f"temperature → {float(value):.2f}"))
            return
        if key == "llm_max_tokens" and self.llm:
            self.llm.update_sampling(max_tokens=int(value))
            await self.bridge.post(msg_event(f"max tokens → {int(value)}"))
            return
        if key == "llm_model_path" and self.llm:
            await self.llm.reload_model(str(value))
            return
        if key == "system_prompt" and self.llm:
            await self.llm.set_system_prompt(str(value))
            return
        if key == "thinking_prompt" and self.llm:
            await self.llm.set_thinking_prompt(str(value))
            return
        if key == "preview_interval_sec" and self.cam:
            self.cam.preview_interval_sec = float(value)
            return
        if key == "snapshot_debounce" and self.cam:
            # Alias for old dashboards: snapshot_debounce was renamed to
            # preview_interval_sec (finding 9).
            self.cam.preview_interval_sec = float(value)
            return
        if key == "motion_sensitivity" and self.cam:
            self.cam.motion_threshold = int(value)
            return
        if key == "vision_cooldown":
            # Single source of truth (finding 5): vision.vision_cooldown.
            # The live param updates both readers — the camera loop's
            # analysis cooldown and the LLM's process_vision cooldown.
            if self.cam:
                self.cam.analysis_cooldown = float(value)
            if self.llm:
                self.llm.vision_cooldown = int(float(value))
            return
        if key == "thought_cooldown" and self.llm:
            self.llm.thought_cooldown = float(value)
            return

        logger.warning(f"Unknown parameter: {key}")

    async def start_workers(self):
        """Initialize and start all worker components."""
        try:
            # Get the current event loop to pass to threaded components
            loop = asyncio.get_running_loop()
            logger.info("Starting workers...")
            
            # Initialize components in correct order
            self.memory = MemoryStore(CFG, self.bridge)
            if self.memory.enabled:
                logger.info("Memory store initialized")
            else:
                logger.info("Memory store unavailable (disabled)")

            self.analytics = ConversationAnalytics()
            logger.info("Conversation analytics initialized")

            self.tts = PiperTTS(CFG, self.bridge)
            logger.info("TTS initialized")

            self.llm = await LocalLLM.create(
                CFG,
                self.bridge,
                self.tts,
                self.memory,
                loop,
                analytics=self.analytics,
            )
            logger.info("LLM initialized")

            prompts = self.llm.get_prompts()
            await self.bridge.post(
                {
                    "type": "prompt_state",
                    "system_prompt": prompts.get("system_prompt", ""),
                    "thinking_prompt": prompts.get("thinking_prompt", ""),
                }
            )

            self.stt = AudioSTT(CFG, self.bridge)
            logger.info("STT initialized")

            # Wire up cross-references for autonomous behavior
            self.stt.set_llm(self.llm)  # STT can trigger LLM
            logger.info("STT -> LLM connection established")
            
            # Pass event loop to camera for thread-safe async calls
            self.cam = CameraLoop(CFG, self.bridge, loop)
            self.cam.set_llm(self.llm)  # Camera can trigger vision analysis
            self.cam.start()
            logger.info("Camera -> LLM connection established")
            
            # Connect person tracker from camera to LLM for identity tools
            if hasattr(self.cam, 'person_tracker') and self.cam.person_tracker:
                self.llm.set_person_tracker(self.cam.person_tracker)
                logger.info("Person tracker connected to LLM")

            self.system_monitor = SystemMonitor(self.bridge)
            await self.system_monitor.start()
            info = self.system_monitor.device_info
            await self.bridge.post(
                msg_event(
                    f"compute backend → {info.backend} ({info.name})"
                )
            )
            logger.info(
                "System monitor initialized (%s - %s)", info.backend, info.reason
            )

            await self.bridge.post(msg_status("idle", "Ready"))
            logger.info("All workers started successfully")

            if self.memory and self.memory.enabled:
                await self.memory.publish_graph()

            # Start autonomous thinking loop
            self._autonomous_task = asyncio.create_task(self._autonomous_loop())
            logger.info("Autonomous thinking loop started")

            # Start nightly dreaming (memory consolidation) whenever a
            # memory store exists — _start_dreaming gates execution on
            # the enabled flag instead of skipping setup.
            self._start_dreaming()
            
        except Exception as e:
            logger.error(f"Error starting workers: {e}", exc_info=True)
            raise

    async def _autonomous_loop(self):
        """Background task for autonomous thoughts.

        Free-running idle tick: wakes the model on a schedule (no user or
        camera event needed) so it can think, speak, or act on its own.
        Cadence comes from autonomous.tick_seconds in config.yaml.
        """
        tick = max(5, int((CFG.get("autonomous", {}) or {}).get("tick_seconds", 30) or 30))
        logger.info("Autonomous idle loop started/running (tick %ss)", tick)
        await asyncio.sleep(tick)  # Initial delay

        while True:
            try:
                if self.llm and self.llm.autonomous_mode:
                    await self.llm.autonomous_thought()
                await asyncio.sleep(tick)
            except asyncio.CancelledError:
                logger.info("Autonomous loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
                await asyncio.sleep(10)

    def _start_dreaming(self) -> None:
        """Create the dreamer + scheduler whenever a memory store exists.

        The enabled flag gates execution (the loop and run_once both
        check it), so a later WebSocket toggle can enable dreaming
        after a disabled startup — the dreamer must exist for the
        toggle to work."""
        if self.memory is None:
            logger.info("Dreaming disabled: no memory store")
            return
        self.dreamer = Dreamer(self.llm, self.memory, self.bridge, CFG)
        self._dreaming_task = asyncio.create_task(self._dreaming_loop())
        logger.info(
            "Dreaming nightly pass scheduled for %s (enabled=%s)",
            self.dreamer.time,
            self.dreamer.enabled,
        )

    async def _dreaming_loop(self):
        """Nightly memory-consolidation pass (dreaming).

        Sleeps until the next configured dreaming.time (default 03:00),
        runs one light -> REM -> deep pass, then sleeps until the next
        day. Silent by definition: the pass never speaks.
        """
        if self.dreamer is None:
            return
        logger.info("Dreaming loop running (next run at %s)", self.dreamer.time)
        while True:
            try:
                delay = seconds_until_next_run(self.dreamer.time)
                await asyncio.sleep(delay)
                if self.dreamer.enabled:
                    await self.dreamer.run_once()
            except asyncio.CancelledError:
                logger.info("Dreaming loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in dreaming loop: {e}")
                await asyncio.sleep(10)

    async def stop_workers(self):
        """Stop all worker components gracefully."""
        logger.info("Stopping workers...")
        
        # Stop autonomous loop
        if self._autonomous_task:
            self._autonomous_task.cancel()
            try:
                await self._autonomous_task
            except asyncio.CancelledError:
                pass

        # Stop dreaming loop
        if self._dreaming_task:
            self._dreaming_task.cancel()
            try:
                await self._dreaming_task
            except asyncio.CancelledError:
                pass

        try:
            if self.system_monitor:
                await self.system_monitor.stop()
                logger.info("System monitor stopped")
        except Exception as e:
            logger.error(f"Error stopping system monitor: {e}")
        finally:
            self.system_monitor = None

        try:
            if self.cam:
                self.cam.stop()
                logger.info("Camera stopped")
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
        
        try:
            if self.stt:
                await self.stt.stop()
                logger.info("STT stopped")
        except Exception as e:
            logger.error(f"Error stopping STT: {e}")
        
        try:
            if self.llm:
                await self.llm.stop()
                logger.info("LLM stopped")
        except Exception as e:
            logger.error(f"Error stopping LLM: {e}")
        
        try:
            if self.tts:
                await self.tts.stop()
                logger.info("TTS stopped")
        except Exception as e:
            logger.error(f"Error stopping TTS: {e}")

        try:
            if self.memory:
                await self.memory.stop()
                logger.info("Memory store stopped")
        except Exception as e:
            logger.error(f"Error stopping memory store: {e}")
        finally:
            self.memory = None

        self.analytics = None

        logger.info("All workers stopped")

    async def handle_scan_models(self, directory: str):
        directory = str(directory or "").strip()
        if not directory:
            await self.bridge.post(
                {"type": "model_catalog", "directory": "", "models": [], "error": "missing_directory"}
            )
            await self.bridge.post(msg_event("model scan failed: directory not provided"))
            return

        path = Path(directory).expanduser()
        if not path.exists() or not path.is_dir():
            await self.bridge.post(
                {
                    "type": "model_catalog",
                    "directory": str(path),
                    "models": [],
                    "error": "not_found",
                }
            )
            await self.bridge.post(msg_event(f"model directory not found: {path}"))
            return

        models = []
        for candidate in sorted(path.glob("*.gguf")):
            try:
                stat = candidate.stat()
            except OSError as exc:
                logger.warning(f"Unable to stat model {candidate}: {exc}")
                continue

            size_bytes = int(stat.st_size)
            models.append(
                {
                    "name": candidate.name,
                    "path": str(candidate),
                    "size_bytes": size_bytes,
                    "size_label": self._format_size(size_bytes),
                    "type": self._infer_model_type(candidate.name),
                }
            )

        await self.bridge.post(
            {
                "type": "model_catalog",
                "directory": str(path),
                "models": models,
                "error": None,
            }
        )
        await self.bridge.post(msg_event(f"model directory scanned: {len(models)} .gguf found"))

    async def handle(self, ws: WebSocketServerProtocol):
        """Handle WebSocket client connection and messages."""
        client_id = f"{ws.remote_address[0]}:{ws.remote_address[1]}"
        logger.info(f"Client connected: {client_id}")
        
        self.clients.add(ws)
        await self.bridge.register_ws(ws)
        
        try:
            await self.bridge.post(msg_event("connected"))
            if self.memory and self.memory.enabled:
                await self.memory.publish_graph()

            async for data in ws:
                try:
                    msg = json.loads(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON from {client_id}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error parsing message from {client_id}: {e}")
                    continue
                
                msg_type = msg.get("type")
                
                try:
                    if msg_type == "ping":
                        await self.bridge.post({"type": "pong"})
                    
                    elif msg_type == "audio":
                        if self.stt:
                            pcm = msg.get("pcm16", "")
                            rate = int(msg.get("rate", 16000))
                            if pcm:
                                self._audio_chunk_count += 1
                                if self._audio_chunk_count == 1:
                                    logger.info("🎤 First audio chunk received from frontend - STT processing started")
                                    await self.bridge.post(msg_event("🎤 Audio streaming active"))
                                elif self._audio_chunk_count % 100 == 0:
                                    logger.debug(f"Audio chunks processed: {self._audio_chunk_count}")
                                await self.stt.feed_base64_pcm(pcm, rate)
                        else:
                            logger.warning("STT not initialized, ignoring audio")
                    
                    elif msg_type == "text":
                        text = msg.get("value", "").strip()
                        if text and self.llm:
                            logger.info(f"Processing text from {client_id}: {text[:50]}...")
                            async for chunk in self.llm.chat(text):
                                await self.bridge.post(chunk)
                        elif not self.llm:
                            logger.warning("LLM not initialized, ignoring text")
                    
                    elif msg_type == "toggle":
                        what = msg.get("what")
                        value = bool(msg.get("value"))
                        logger.info(f"Toggle {what} = {value}")
                        await self.handle_toggle(what, value)

                    elif msg_type == "dream_now":
                        # Manual dreaming trigger (verify without waiting for 3 AM).
                        if self.dreamer:
                            logger.info("Manual dreaming run requested")
                            asyncio.create_task(self.dreamer.run_once())
                        else:
                            logger.warning("dream_now ignored: no dreamer")

                    elif msg_type == "param":
                        key = msg.get("key")
                        value = msg.get("value")
                        logger.info(f"Parameter {key} = {value}")
                        await self.handle_param(key, value)
                    
                    elif msg_type == "reinforce":
                        if self.llm:
                            delta = float(msg.get("delta", 0))
                            await self.llm.reinforce(delta)
                        else:
                            logger.warning("LLM not initialized, ignoring reinforce")

                    elif msg_type == "tools/list":
                        # MCP-style registry listing, local only: the
                        # dashboard (or any local client) can enumerate the
                        # same tool definitions the model calls through.
                        if self.llm:
                            await self.bridge.post({
                                "type": "tools_list",
                                "tools": self.llm.tools.get_mcp_tools(),
                            })
                        else:
                            logger.warning("LLM not initialized, ignoring tools/list")

                    elif msg_type == "tools/call":
                        # Direct tool invocation through the same validated
                        # registry/dispatch the model uses. Local only.
                        if self.llm:
                            tool_name = str(msg.get("tool", ""))
                            args = msg.get("args") or {}
                            if not isinstance(args, dict):
                                args = {}
                            execution = await self.llm.tools.execute_tool(tool_name, args)
                            await self.bridge.post({
                                "type": "tool_result",
                                **execution,
                            })
                        else:
                            logger.warning("LLM not initialized, ignoring tools/call")

                    elif msg_type == "memory_update":
                        if not self.memory:
                            logger.warning("Memory not initialized, ignoring memory_update")
                            continue
                        node_id = msg.get("id") or msg.get("nodeId")
                        patch = msg.get("patch") or {}
                        if not node_id or not isinstance(patch, dict):
                            logger.warning("Invalid memory_update payload: %s", msg)
                            continue
                        success = await self.memory.update_node(str(node_id), patch)
                        if success:
                            await self.bridge.post(msg_event(f"memory node updated → {node_id}"))
                        else:
                            await self.bridge.post(msg_event(f"memory update failed → {node_id}"))

                    elif msg_type == "memory_delete":
                        if not self.memory:
                            logger.warning("Memory not initialized, ignoring memory_delete")
                            continue
                        node_id = msg.get("id") or msg.get("nodeId")
                        if not node_id:
                            continue
                        success = await self.memory.delete_node(str(node_id))
                        if success:
                            await self.bridge.post(msg_event(f"memory node removed → {node_id}"))

                    elif msg_type == "memory_create":
                        if not self.memory:
                            logger.warning("Memory not initialized, ignoring memory_create")
                            continue
                        node_payload = msg.get("node") or msg.get("payload") or {}
                        if not isinstance(node_payload, dict):
                            logger.warning("Invalid memory_create payload: %s", msg)
                            continue
                        node_id = await self.memory.create_node(node_payload)
                        if node_id:
                            await self.bridge.post(msg_event(f"memory node added → {node_id}"))

                    elif msg_type == "memory_link":
                        if not self.memory:
                            logger.warning("Memory not initialized, ignoring memory_link")
                            continue
                        from_id = msg.get("from") or msg.get("fromId")
                        to_id = msg.get("to") or msg.get("toId")
                        if not from_id or not to_id:
                            continue
                        weight = float(msg.get("weight", 1.0))
                        relationship = msg.get("relationship")
                        context = msg.get("context") if isinstance(msg.get("context"), dict) else None
                        success = await self.memory.create_edge(
                            str(from_id),
                            str(to_id),
                            weight=weight,
                            relationship=relationship,
                            context=context,
                        )
                        if success:
                            await self.bridge.post(msg_event(f"memory link {from_id} → {to_id}"))

                    elif msg_type == "memory_unlink":
                        if not self.memory:
                            logger.warning("Memory not initialized, ignoring memory_unlink")
                            continue
                        edge_id = msg.get("id") or msg.get("edgeId")
                        if not edge_id:
                            continue
                        success = await self.memory.delete_edge(str(edge_id))
                        if success:
                            await self.bridge.post(msg_event(f"memory link removed → {edge_id}"))
                    elif msg_type == "scan_models":
                        directory = msg.get("directory", "")
                        logger.info(f"Scanning model directory: {directory}")
                        await self.handle_scan_models(directory)
                    
                    elif msg_type == "get_entities":
                        if self.memory and self.memory.enabled:
                            entity_type = msg.get("entity_type")
                            limit = msg.get("limit", 100)
                            entities = await self.memory.get_entities(entity_type=entity_type, limit=limit)
                            await self.bridge.post(msg_entities(entities))
                        else:
                            logger.warning("Memory not initialized, ignoring get_entities")
                    
                    elif msg_type == "get_timeline":
                        if self.memory and self.memory.enabled:
                            entity_id = msg.get("entity_id")
                            limit = msg.get("limit", 50)
                            events = await self.memory.get_learning_timeline(entity_id=entity_id, limit=limit)
                            await self.bridge.post(msg_timeline(events))
                        else:
                            logger.warning("Memory not initialized, ignoring get_timeline")
                    
                    elif msg_type == "semantic_search":
                        if self.memory and self.memory.enabled:
                            query = msg.get("query", "")
                            limit = msg.get("limit", 10)
                            threshold = msg.get("threshold", 0.3)
                            entity_type = msg.get("entity_type")
                            results = await self.memory.semantic_search(query, limit=limit, threshold=threshold)
                            await self.bridge.post(msg_search_results(results))
                        else:
                            logger.warning("Memory not initialized, ignoring semantic_search")

                    else:
                        logger.warning(f"Unknown message type from {client_id}: {msg_type}")
                        await self.bridge.post(msg_event(f"unknown message: {msg_type}"))

                except Exception as e:
                    logger.error(f"Error handling message type '{msg_type}' from {client_id}: {e}", exc_info=True)
                    await self.bridge.post(msg_event(f"error processing {msg_type}: {str(e)}"))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected normally: {client_id}")
        except Exception as e:
            logger.error(f"Error in client handler for {client_id}: {e}", exc_info=True)
        finally:
            self.clients.discard(ws)
            await self.bridge.unregister_ws(ws)
            try:
                await self.bridge.post(msg_event("disconnected"))
            except Exception as e:
                logger.error(f"Error posting disconnect event: {e}")
            logger.info(f"Client cleanup complete: {client_id}")


async def main():
    """Main entry point for the WebSocket bridge server."""
    s = Server()
    
    try:
        host = CFG["ws"]["host"]
        port = int(CFG["ws"]["port"])

        logger.info(f"Starting WebSocket server on {host}:{port}")

        async with websockets.serve(s.handle, host, port, max_size=2**22):
            await s.bridge.post(msg_event(f"listening on ws://{host}:{port}"))
            logger.info(f"🚀 Autonomous AI ready on ws://{host}:{port}")

            # Start background workers after the socket is bound so external
            # readiness checks detect the service immediately (see start.py).
            try:
                await s.start_workers()
            except Exception as e:
                # Ensure partial startup is unwound before re-raising so the
                # caller can log the original failure.
                logger.error("Error during start_workers", exc_info=True)
                try:
                    await s.stop_workers()
                except Exception as stop_exc:
                    logger.error("Error during stop_workers while unwinding startup failure", exc_info=True)
                    raise stop_exc from e
                raise

            try:
                # Keep server running indefinitely
                await asyncio.Future()
            except asyncio.CancelledError:
                logger.info("Server shutdown requested")
            finally:
                await s.stop_workers()
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        await s.stop_workers()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        await s.stop_workers()
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server crashed: {e}", exc_info=True)
        exit(1)
