# Title: PiperTTS - With Auto-Playback
# Path: backend/tts.py
# Purpose: Call piper.exe to synthesize speech AND play it automatically

import os
import time
import asyncio
import logging
import shutil
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

        # Speak queue (finding 7): speak() enqueues and returns immediately;
        # a single worker synthesizes + plays in order, so overlapping chat,
        # vision, and autonomous replies no longer race Piper. Interrupt
        # bumps the generation: the worker drops the in-flight item and
        # everything queued behind it. Wav files are pruned to the newest
        # _keep_wavs so piper_out stops growing forever.
        self._speak_queue: asyncio.Queue = asyncio.Queue()
        self._speak_worker_task: Optional[asyncio.Task] = None
        self._current_proc = None
        self._generation = 0
        self._keep_wavs = 20
        self.muted = False

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
        await self.interrupt()
        if self._speak_worker_task is not None:
            self._speak_worker_task.cancel()
            try:
                await self._speak_worker_task
            except asyncio.CancelledError:
                pass
            self._speak_worker_task = None

    def set_muted(self, value: bool):
        """Mute/unmute speech (gesture fist toggles this). Muted speech is
        dropped, not queued for later — unmuting does not replay it."""
        self.muted = bool(value)
        logger.info(f"PiperTTS {'muted' if self.muted else 'unmuted'}")

    def toggle_muted(self) -> bool:
        """Flip the mute flag; returns the new muted state."""
        self.set_muted(not self.muted)
        return self.muted

    def _ensure_worker(self):
        if self._speak_worker_task is None or self._speak_worker_task.done():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            self._speak_worker_task = loop.create_task(self._speak_worker())

    async def interrupt(self):
        """Stop current playback and drop everything queued behind it.

        Called when new chat/audio input arrives so the model never talks
        over the user. Bumps the generation so the worker abandons the
        in-flight item even if it is mid-synthesis.
        """
        self._generation += 1
        drained = 0
        while True:
            try:
                self._speak_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._speak_queue.task_done()
            drained += 1
        if self._current_proc is not None:
            try:
                self._current_proc.kill()
            except Exception:
                pass
            self._current_proc = None
        try:
            import sounddevice as sd
            await asyncio.to_thread(sd.stop)
        except Exception:
            pass
        if drained:
            logger.info("TTS interrupted, dropped %s queued item(s)", drained)

    async def _speak_worker(self):
        """Serial worker: synthesize + play queued speech in order."""
        while True:
            text = await self._speak_queue.get()
            gen = self._generation
            try:
                if self.muted or not self.enabled:
                    continue
                wav = await self._synthesize(text)
                if wav is None or gen != self._generation:
                    continue
                if self.auto_play:
                    await self._play_audio(wav)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("TTS worker error: %s", exc, exc_info=True)
            finally:
                self._speak_queue.task_done()

    def _prune_wav_files(self):
        """Keep only the newest _keep_wavs files in piper_out."""
        try:
            wavs = sorted(
                Path(self.out_dir).glob("speech_*.wav"),
                key=lambda p: p.stat().st_mtime,
            )
            if len(wavs) > self._keep_wavs:
                for stale in wavs[:-self._keep_wavs]:
                    try:
                        stale.unlink()
                    except OSError as exc:
                        logger.debug("Could not delete old wav %s: %s", stale, exc)
        except Exception as exc:
            logger.debug("Wav pruning failed: %s", exc)

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

    async def _play_audio(self, wav_path: str):
        """Play a WAV file through sounddevice (cross-platform, no shelling out)."""
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            logger.warning("sounddevice/soundfile missing; audio playback skipped")
            return
        try:
            data, samplerate = await asyncio.to_thread(sf.read, wav_path, dtype="float32")
        except Exception as exc:
            logger.error("Could not read audio for playback: %s", exc)
            return
        try:
            await asyncio.to_thread(sd.play, data * self.master_volume, samplerate)
            await asyncio.to_thread(sd.wait)
            logger.info("Audio playback completed: %s", os.path.basename(wav_path))
        except Exception as exc:
            logger.error("Error playing audio: %s", exc)

    async def speak(self, text: str):
        """Queue text for speech and return once queued.

        Playback is serial through the worker, so callers never block on
        synthesis and overlapping replies queue in order instead of
        racing. Use interrupt() to cut speech when new input arrives.
        """
        if not self.enabled:
            logger.warning("TTS is disabled, skipping speech")
            return

        text = text.strip()
        if not text:
            return

        if self.muted:
            logger.debug("TTS muted, skipping speech")
            return

        self._ensure_worker()
        self._speak_queue.put_nowait(text)

    async def _synthesize(self, text: str) -> Optional[str]:
        """Synthesize one utterance with Piper; wav path or None on failure."""
        # Remove quotes and problematic characters
        text = text.replace('"', "'").replace('\\n', ' ')

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
            self._current_proc = proc

            # Send text to Piper via stdin
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=text.encode("utf-8")),
                timeout=10.0
            )
            self._current_proc = None

            # Check result
            if proc.returncode == 0:
                if os.path.exists(wav):
                    file_size = os.path.getsize(wav)
                    logger.info(f"Speech generated: {wav} ({file_size} bytes)")
                    await self.bridge.post(msg_event(f"🔊 Speech saved: {os.path.basename(wav)}"))
                    self._prune_wav_files()
                    return wav
                logger.error(f"Piper succeeded but wav file not found: {wav}")
            else:
                error_msg = stderr.decode("utf-8", errors="ignore").strip()
                logger.error(f"Piper failed with code {proc.returncode}: {error_msg}")
                await self.bridge.post(msg_event(f"TTS error: {error_msg[:100]}"))

        except asyncio.TimeoutError:
            logger.error("Piper process timed out")
            try:
                proc.kill()
            except Exception:
                pass
            self._current_proc = None
            await self.bridge.post(msg_event("TTS error: Process timed out"))

        except FileNotFoundError:
            logger.error(f"Piper executable not found: {self.exe}")
            await self.bridge.post(msg_event("TTS error: Piper not found"))
            self.enabled = False

        except Exception as e:
            logger.error(f"TTS error: {e}", exc_info=True)
            await self.bridge.post(msg_event(f"TTS error: {str(e)}"))

        return None
