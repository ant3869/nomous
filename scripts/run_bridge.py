# Title: Nomous WS Bridge - Autonomous Edition
# Path: run_bridge.py
# Purpose: Start WS server with autonomous AI that thinks, sees, and speaks

import os
import sys
import asyncio
import json
import logging
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
from src.backend.tts import PiperTTS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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

    async def start_workers(self):
        """Initialize and start all worker components."""
        try:
            # Get the current event loop to pass to threaded components
            loop = asyncio.get_running_loop()
            logger.info("Starting workers...")
            
            # Initialize components in correct order
            self.tts = PiperTTS(CFG, self.bridge)
            logger.info("TTS initialized")
            
            self.llm = LocalLLM(CFG, self.bridge, self.tts)
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
            
            await self.bridge.post(msg_status("idle", "Ready"))
            logger.info("All workers started successfully")
            
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
        
        logger.info("All workers stopped")

    async def handle(self, ws: WebSocketServerProtocol):
        """Handle WebSocket client connection and messages."""
        client_id = f"{ws.remote_address[0]}:{ws.remote_address[1]}"
        logger.info(f"Client connected: {client_id}")
        
        self.clients.add(ws)
        self.bridge.register_ws(ws)
        
        try:
            await self.bridge.post(msg_event("connected"))
            
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
                        
                        if what == "tts" and self.tts:
                            self.tts.enabled = value
                        elif what == "vision" and self.cam:
                            self.cam.enabled = value
                        elif what == "autonomous" and self.llm:
                            self.llm.autonomous_mode = value
                            logger.info(f"Autonomous mode: {value}")
                        else:
                            logger.warning(f"Unknown toggle target: {what}")
                    
                    elif msg_type == "param":
                        key = msg.get("key")
                        value = msg.get("value")
                        logger.info(f"Parameter {key} = {value}")
                        
                        if key == "snapshot_debounce" and self.cam:
                            self.cam.debounce_sec = float(value)
                        elif key == "motion_sensitivity" and self.cam:
                            self.cam.motion_threshold = int(value)
                        elif key == "vision_cooldown" and self.cam:
                            self.cam.analysis_cooldown = float(value)
                        elif key == "thought_cooldown" and self.llm:
                            self.llm.thought_cooldown = float(value)
                        else:
                            logger.warning(f"Unknown parameter: {key}")
                    
                    elif msg_type == "reinforce":
                        if self.llm:
                            delta = float(msg.get("delta", 0))
                            await self.llm.reinforce(delta)
                        else:
                            logger.warning("LLM not initialized, ignoring reinforce")
                    
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
            self.bridge.unregister_ws(ws)
            try:
                await self.bridge.post(msg_event("disconnected"))
            except Exception as e:
                logger.error(f"Error posting disconnect event: {e}")
            logger.info(f"Client cleanup complete: {client_id}")


async def main():
    """Main entry point for the WebSocket bridge server."""
    s = Server()
    
    try:
        await s.start_workers()
        
        host = CFG["ws"]["host"]
        port = int(CFG["ws"]["port"])
        
        logger.info(f"Starting WebSocket server on {host}:{port}")
        
        async with websockets.serve(s.handle, host, port, max_size=2**22):
            await s.bridge.post(msg_event(f"listening on ws://{host}:{port}"))
            logger.info(f"🚀 Autonomous AI ready on ws://{host}:{port}")
            
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
