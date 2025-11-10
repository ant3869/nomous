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

from src.backend.config import load_config
from src.backend.utils import Bridge, msg_status, msg_event
from src.backend.video import CameraLoop
from src.backend.audio import AudioSTT
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
        self.memory: Optional[MemoryStore] = None
        self.system_monitor: Optional[SystemMonitor] = None

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
        if key == "vision_model_path" and self.cam:
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
        if key == "snapshot_debounce" and self.cam:
            self.cam.debounce_sec = float(value)
            return
        if key == "motion_sensitivity" and self.cam:
            self.cam.motion_threshold = int(value)
            return
        if key == "vision_cooldown" and self.cam:
            self.cam.analysis_cooldown = float(value)
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

            self.tts = PiperTTS(CFG, self.bridge)
            logger.info("TTS initialized")

            self.llm = await LocalLLM.create(CFG, self.bridge, self.tts, self.memory, loop)
            logger.info("LLM initialized")

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
            
        except Exception as e:
            logger.error(f"Error starting workers: {e}", exc_info=True)
            raise

    async def _autonomous_loop(self):
        """Background task for autonomous thoughts."""
        logger.info("Autonomous loop running")
        await asyncio.sleep(30)  # Initial delay
        
        while True:
            try:
                if self.llm and self.llm.autonomous_mode:
                    await self.llm.autonomous_thought()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                logger.info("Autonomous loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in autonomous loop: {e}")
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

        logger.info("All workers stopped")

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
