# Title: PiperTTS - With Auto-Playback
# Path: backend/tts.py
# Purpose: Call piper.exe to synthesize speech AND play it automatically

import os
import time
import asyncio
import logging
from pathlib import Path
from .utils import msg_speak, msg_event

logger = logging.getLogger(__name__)


class PiperTTS:
    def __init__(self, cfg, bridge):
        self.bridge = bridge
        p = cfg["paths"]
        
        self.exe = p["piper_exe"]
        self.voice = p["piper_voice"]
        self.out_dir = p["piper_out_dir"]
        self.rate = 22050
        self.enabled = bool(cfg["ui"]["tts_enabled"])
        self.auto_play = True  # NEW: Auto-play generated speech
        
        # Validate paths
        if not os.path.exists(self.exe):
            logger.error(f"Piper executable not found: {self.exe}")
            self.enabled = False
        else:
            logger.info(f"Piper found: {self.exe}")
        
        if not os.path.exists(self.voice):
            logger.error(f"Piper voice not found: {self.voice}")
            self.enabled = False
        else:
            logger.info(f"Voice model found: {self.voice}")
        
        # Create output directory
        os.makedirs(self.out_dir, exist_ok=True)
        logger.info(f"TTS output directory: {self.out_dir}")
        
        if self.enabled:
            logger.info("PiperTTS initialized successfully")
        else:
            logger.warning("PiperTTS disabled due to missing files")

    async def stop(self):
        logger.info("PiperTTS stopping...")

    async def _play_audio(self, wav_path: str):
        """Play audio file using Windows sound API."""
        try:
            # Use PowerShell to play audio (works on Windows without extra dependencies)
            play_cmd = [
                "powershell", "-c",
                f"(New-Object Media.SoundPlayer '{wav_path}').PlaySync()"
            ]
            
            proc = await asyncio.create_subprocess_exec(
                *play_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            
            await asyncio.wait_for(proc.wait(), timeout=30.0)
            logger.info(f"Audio playback completed: {os.path.basename(wav_path)}")
            
        except asyncio.TimeoutError:
            logger.warning("Audio playback timed out")
            try:
                proc.kill()
            except:
                pass
        except Exception as e:
            logger.error(f"Error playing audio: {e}")

    async def speak(self, text: str):
        """Synthesize speech from text and play it."""
        if not self.enabled:
            logger.warning("TTS is disabled, skipping speech")
            return
        
        text = text.strip()
        if not text:
            return
        
        # Remove quotes and problematic characters
        text = text.replace('"', "'").replace('\n', ' ')
        
        # Generate unique filename
        ts = int(time.time() * 1000)
        wav = os.path.join(self.out_dir, f"speech_{ts}.wav")
        
        # Build command - Piper reads from stdin
        cmd = [
            self.exe,
            "-m", self.voice,
            "-f", wav,
            "--sample-rate", str(self.rate)
        ]
        
        try:
            logger.info(f"Speaking: {text[:100]}...")
            await self.bridge.post(msg_speak(text))
            
            # Run Piper process
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Send text to Piper via stdin
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=text.encode("utf-8")),
                timeout=10.0
            )
            
            # Check result
            if proc.returncode == 0:
                if os.path.exists(wav):
                    file_size = os.path.getsize(wav)
                    logger.info(f"Speech generated: {wav} ({file_size} bytes)")
                    await self.bridge.post(msg_event(f"🔊 Speech saved: {os.path.basename(wav)}"))
                    
                    # NEW: Auto-play the audio
                    if self.auto_play:
                        await self._play_audio(wav)
                else:
                    logger.error(f"Piper succeeded but wav file not found: {wav}")
            else:
                error_msg = stderr.decode("utf-8", errors="ignore").strip()
                logger.error(f"Piper failed with code {proc.returncode}: {error_msg}")
                await self.bridge.post(msg_event(f"TTS error: {error_msg[:100]}"))
                
        except asyncio.TimeoutError:
            logger.error("Piper process timed out")
            try:
                proc.kill()
            except:
                pass
            await self.bridge.post(msg_event("TTS error: Process timed out"))
            
        except FileNotFoundError:
            logger.error(f"Piper executable not found: {self.exe}")
            await self.bridge.post(msg_event("TTS error: Piper not found"))
            self.enabled = False
            
        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            await self.bridge.post(msg_event(f"TTS error: {str(e)}"))
