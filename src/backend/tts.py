# Title: PiperTTS - With Auto-Playback
# Path: backend/tts.py
# Purpose: Call piper.exe to synthesize speech AND play it automatically

import os
import time
import asyncio
import logging
import platform
import shutil
import wave
import audioop
from pathlib import Path
from typing import List, Optional
from .utils import msg_event

logger = logging.getLogger(__name__)


class PiperTTS:
    def __init__(self, cfg, bridge):
        self.bridge = bridge
        p = cfg["paths"]
        
        self.exe = self._resolve_executable(p.get("piper_exe"))
        self.voice = p["piper_voice"]
        self.out_dir = p["piper_out_dir"]
        self.rate = 22050
        self.enabled = bool(cfg["ui"]["tts_enabled"])
        self.auto_play = True  # NEW: Auto-play generated speech
        self.master_volume = 0.7

        self._voices_root = Path(self.voice).parent

        # Validate executable path
        if not self.exe or not os.path.exists(self.exe):
            logger.error(
                "Piper executable not found. Checked configured path and common fallbacks."
            )
            self.enabled = False
        else:
            logger.info(f"Piper found: {self.exe}")

        # Validate or discover voice model
        try:
            resolved_voice = self._resolve_voice_path(self.voice)
            self.voice = str(resolved_voice)
            logger.info(f"Voice model found: {self.voice}")
        except FileNotFoundError:
            logger.warning("Configured voice model not found, attempting auto-discovery")
            fallback_voice = self._auto_discover_voice()
            if fallback_voice:
                self.voice = str(fallback_voice)
                logger.info(f"Using fallback voice model: {self.voice}")
            else:
                logger.error("No Piper voice models available. Disabling TTS.")
                self.enabled = False

        # Create output directory
        os.makedirs(self.out_dir, exist_ok=True)
        logger.info(f"TTS output directory: {self.out_dir}")

        if self.enabled:
            logger.info("PiperTTS initialized successfully")
        else:
            logger.warning("PiperTTS disabled due to missing files")

    def _resolve_executable(self, configured: Optional[str]) -> Optional[str]:
        if configured:
            configured_path = Path(configured).expanduser()
        else:
            configured_path = None

        candidates: List[Path] = []

        if configured_path:
            candidates.append(configured_path)
            if configured_path.suffix.lower() == ".exe":
                candidates.append(configured_path.with_suffix(""))
            if configured_path.name.lower().endswith("piper.exe"):
                candidates.append(configured_path.with_name("piper"))

        env_path = shutil.which("piper")
        if env_path:
            candidates.append(Path(env_path))

        for candidate in candidates:
            if candidate and candidate.exists():
                return str(candidate)

        return str(configured_path) if configured_path and configured_path.exists() else None

    def _auto_discover_voice(self) -> Optional[Path]:
        root = self._voices_root
        if not root.exists():
            return None

        try:
            for candidate in root.rglob("*.onnx"):
                return candidate
        except Exception as exc:
            logger.debug("Voice discovery failed: %s", exc)
        return None

    async def stop(self):
        logger.info("PiperTTS stopping...")

    def set_enabled(self, value: bool):
        self.enabled = bool(value)
        logger.info(f"PiperTTS {'enabled' if self.enabled else 'disabled'}")

    def set_auto_play(self, value: bool):
        self.auto_play = bool(value)
        logger.info(f"Piper auto-play {'enabled' if self.auto_play else 'disabled'}")

    def set_volume(self, percent: int):
        percent = max(0, min(100, int(percent)))
        self.master_volume = percent / 100.0
        logger.info(f"Piper master volume set to {percent}%")

    def _resolve_voice_path(self, voice: str) -> Path:
        path = Path(voice)

        if path.exists():
            return path

        candidates = []

        # Directly inside voices root keeping any nested folders
        if not path.is_absolute():
            candidates.append(self._voices_root / path)

        # Attempt with only the filename component
        name_only = path.name
        if name_only:
            candidates.append(self._voices_root / name_only)

        # Ensure the ONNX extension is present when omitted
        if path.suffix.lower() != ".onnx":
            with_suffix = path.with_suffix(".onnx")
            candidates.append(self._voices_root / with_suffix)
            if name_only:
                candidates.append(self._voices_root / f"{Path(name_only).stem}.onnx")

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f"Voice model not found: {voice}")

    async def set_voice(self, voice: str):
        voice_path = await asyncio.to_thread(self._resolve_voice_path, voice)
        self.voice = str(voice_path)
        logger.info(f"Piper voice set to {self.voice}")
        await self.bridge.post(msg_event(f"TTS voice → {voice_path.name}"))

    async def set_voice_path(self, path: str):
        await self.set_voice(path)

    def _get_play_command(self, wav_path: str) -> Optional[List[str]]:
        """Determine the best available playback command for the current OS."""

        system = platform.system()

        if system == "Windows":
            powershell = shutil.which("powershell") or shutil.which("powershell.exe")
            if powershell:
                return [
                    powershell,
                    "-c",
                    f"(New-Object Media.SoundPlayer '{wav_path}').PlaySync()"
                ]
            logger.debug("PowerShell not available for audio playback")
            return None

        if system == "Darwin":  # macOS
            afplay = shutil.which("afplay")
            if afplay:
                return [afplay, wav_path]
            logger.debug("afplay command not available on macOS")
            return None

        # Linux or other Unix-like systems
        for candidate in ("paplay", "aplay", "ffplay"):
            path = shutil.which(candidate)
            if path:
                if candidate == "ffplay":
                    return [path, "-nodisp", "-autoexit", wav_path]
                return [path, wav_path]

        logger.debug("No suitable audio playback command found on this system")
        return None

    async def _play_audio(self, wav_path: str):
        """Play audio file using a platform-appropriate command."""
        cmd = self._get_play_command(wav_path)

        if not cmd:
            logger.warning("Audio playback skipped: no supported player found")
            return

        logger.debug("Using playback command: %s", cmd)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )

            await asyncio.wait_for(proc.wait(), timeout=30.0)
            logger.info(f"Audio playback completed: {os.path.basename(wav_path)}")

        except asyncio.TimeoutError:
            logger.warning("Audio playback timed out")
            try:
                proc.kill()
            except Exception as e:
                logger.error(f"Failed to kill audio playback process: {e}")
        except FileNotFoundError:
            logger.warning("Audio playback command disappeared during execution")
        except Exception as e:
            logger.error(f"Error playing audio: {e}")

    def _apply_volume(self, wav_path: str):
        if not os.path.exists(wav_path):
            return

        if abs(self.master_volume - 1.0) < 0.01:
            return

        with wave.open(wav_path, "rb") as src:
            params = src.getparams()
            frames = src.readframes(src.getnframes())

        scaled = audioop.mul(frames, params.sampwidth, self.master_volume)

        with wave.open(wav_path, "wb") as dst:
            dst.setparams(params)
            dst.writeframes(scaled)

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
                    self._apply_volume(wav)
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
