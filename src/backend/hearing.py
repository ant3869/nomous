# Title: AmbientHearing - hearing as a sense
# Path: backend/hearing.py
# Purpose: Continuous host-mic capture with VAD + Vosk transcription,
# feeding the idle loop as audio evidence. This is NOT chat STT — that is
# AudioSTT on browser-mic PCM (user-initiated). Hearing is the mic as a
# sense on its own: it hears stuff in the room and decides what to do
# with it, under the same think/speak/act rules as vision.
#
# Privacy: raw audio is never written to disk — PCM is transcribed in
# memory and discarded. Transcripts live only as transient idle-loop
# evidence (TTL) plus whatever the model's own thoughts about them record
# to memory, gated by hearing.retain_transcripts. The sense is opt-in
# (hearing.enabled defaults to false) and the dashboard shows a live
# mic indicator whenever capture is running.

import asyncio
import json
import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# webrtcvad only accepts 10/20/30 ms frames at 8/16/32/48 kHz.
FRAME_MS = 30
_NOISE_RUN_FRAMES = 3  # consecutive loud non-speech frames before a noise event


def _rms_normalized(frame: bytes) -> float:
    """RMS amplitude of an int16 PCM frame, 0.0-1.0."""
    if not frame:
        return 0.0
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples * samples)))


class HearingSegmenter:
    """Pure VAD segmentation: frames in, speech/noise events out.

    No mic, no model, no I/O — the VAD function is injected so tests can
    drive it with a fake and production passes webrtcvad. A speech
    segment ends after `silence_ms` of non-speech or at `max_segment_s`;
    a loud non-speech burst (door slam, dropped object) is a noise event.
    """

    def __init__(
        self,
        vad_fn: Callable[[bytes], bool],
        *,
        sample_rate: int = 16000,
        silence_ms: int = 500,
        max_segment_s: int = 10,
        noise_rms_threshold: float = 0.35,
        noise_debounce_s: float = 2.0,
    ):
        self._vad_fn = vad_fn
        self.sample_rate = sample_rate
        self.silence_ms = max(0, int(silence_ms))
        self.max_segment_s = max(1, int(max_segment_s))
        self.noise_rms_threshold = float(noise_rms_threshold)
        self.noise_debounce_s = float(noise_debounce_s)
        self._frame_ms = FRAME_MS
        self._speech_buf: list[bytes] = []
        self._silence_frames = 0
        self._noise_run = 0
        self._last_noise_at = 0.0

    @property
    def _silence_frames_needed(self) -> int:
        return max(1, self.silence_ms // self._frame_ms)

    @property
    def _max_segment_frames(self) -> int:
        return max(1, (self.max_segment_s * 1000) // self._frame_ms)

    def _emit_speech(self) -> list:
        events = []
        if self._speech_buf:
            events.append(("speech", b"".join(self._speech_buf)))
            self._speech_buf = []
        self._silence_frames = 0
        return events

    def feed(self, frame: bytes) -> list:
        """Consume one 30 ms frame; returns zero or more events."""
        if not frame:
            return []
        events: list = []
        is_speech = bool(self._vad_fn(frame))
        if is_speech:
            self._speech_buf.append(frame)
            self._silence_frames = 0
            self._noise_run = 0
            if len(self._speech_buf) >= self._max_segment_frames:
                events.extend(self._emit_speech())
            return events
        # Non-speech frame.
        if self._speech_buf:
            self._silence_frames += 1
            if self._silence_frames >= self._silence_frames_needed:
                events.extend(self._emit_speech())
            return events
        # No open segment: watch for loud non-speech bursts.
        if _rms_normalized(frame) >= self.noise_rms_threshold:
            self._noise_run += 1
            now = time.monotonic()
            if self._noise_run >= _NOISE_RUN_FRAMES and now - self._last_noise_at >= self.noise_debounce_s:
                self._last_noise_at = now
                self._noise_run = 0
                events.append(("noise", _rms_normalized(frame)))
        else:
            self._noise_run = 0
        return events

    def flush(self) -> list:
        """End any open segment (e.g. on shutdown)."""
        return self._emit_speech()


class AmbientHearing:
    """Host-mic capture -> VAD segmentation -> Vosk -> idle-loop evidence.

    Capture runs on a sounddevice callback thread (never the event loop);
    frames queue into a bounded queue and an async drain task segments and
    transcribes them. The Vosk recognizer here is separate from AudioSTT's
    — chat input and ambient hearing must never share recognizer state —
    but it reuses AudioSTT's loaded Model when one is passed in, so the
    acoustic model is only loaded once.
    """

    def __init__(self, cfg, bridge, vosk_model=None):
        hearing = cfg.get("hearing", {}) or {}
        self._cfg = hearing
        self.bridge = bridge
        self.enabled = bool(hearing.get("enabled", False))
        self.device = hearing.get("device", "default")
        self.sample_rate = int(hearing.get("sample_rate", 16000) or 16000)
        self.vad_aggressiveness = max(0, min(3, int(hearing.get("vad_aggressiveness", 2) or 0)))
        self.segment_silence_ms = max(0, int(hearing.get("segment_silence_ms", 500) or 0))
        self.max_segment_s = max(1, int(hearing.get("max_segment_s", 10) or 10))
        self.min_transcript_chars = max(1, int(hearing.get("min_transcript_chars", 3) or 3))
        self.noise_rms_threshold = float(hearing.get("noise_rms_threshold", 0.35))
        self.listen_boost_seconds = max(0, int(hearing.get("listen_boost_seconds", 30) or 0))
        self.model_path = (cfg.get("paths", {}) or {}).get("vosk_model_dir", "")

        self._llm = None
        self._vosk_model = vosk_model  # shared with AudioSTT when available
        self._own_model = False
        self._recognizer = None
        self._vad = None
        self._vad_mode: Optional[int] = None
        self._segmenter: Optional[HearingSegmenter] = None
        self._stream = None
        self._queue: queue.Queue = queue.Queue(maxsize=200)
        self._drain_task: Optional[asyncio.Task] = None
        self._running = False
        self._state = "off"
        self._boost_until = 0.0
        self._lock = threading.Lock()

    def set_llm(self, llm) -> None:
        """Set the LLM reference for audio-evidence delivery."""
        self._llm = llm

    def listen_boost(self, seconds: Optional[int] = None) -> None:
        """Vision saw something new: listen harder for the matching sound.

        Drops the effective VAD aggressiveness by one (more sensitive)
        until the boost window ends. A no-op when hearing is stopped.
        """
        if not self._running:
            return
        window = self.listen_boost_seconds if seconds is None else max(0, int(seconds))
        self._boost_until = time.monotonic() + window

    def _effective_aggressiveness(self) -> int:
        if time.monotonic() < self._boost_until:
            return max(0, self.vad_aggressiveness - 1)
        return self.vad_aggressiveness

    def _is_speech(self, frame: bytes) -> bool:
        mode = self._effective_aggressiveness()
        if self._vad is None or self._vad_mode != mode:
            import webrtcvad

            self._vad = webrtcvad.Vad(mode)
            self._vad_mode = mode
        return bool(self._vad.is_speech(frame, self.sample_rate))

    async def _post_state(self, state: str, text: str = "", kind: str = "") -> None:
        self._state = state
        payload: dict = {"type": "hearing", "state": state}
        if text:
            payload["text"] = text
        if kind:
            payload["kind"] = kind
        try:
            await self.bridge.post(payload)
        except Exception as exc:  # pragma: no cover - dashboard already gone
            logger.debug("hearing state post failed: %s", exc)

    @property
    def state(self) -> str:
        """Current capture state: off | listening | error."""
        return self._state

    async def announce_state(self) -> None:
        """Re-broadcast the current state (e.g. to a newly connected
        dashboard) so its mic indicator shows the real capture state,
        not its defaults."""
        await self._post_state(self._state)

    async def start(self) -> bool:
        """Start ambient capture. Returns False (hearing stays off) when
        the mic stack is unavailable — never raises into startup."""
        if self._running:
            return True
        try:
            import sounddevice as sd
            import webrtcvad  # noqa: F401  (import check; Vad is built lazily)
        except Exception as exc:
            logger.error("Ambient hearing unavailable (mic stack import failed): %s", exc)
            await self._post_state("error", f"mic stack unavailable: {exc}")
            return False

        # Load the Vosk model (reuse AudioSTT's when one was passed in).
        try:
            if self._vosk_model is None:
                from vosk import Model

                import os

                if not self.model_path or not os.path.isdir(self.model_path):
                    raise RuntimeError(f"Vosk model directory not found: {self.model_path}")
                self._vosk_model = Model(self.model_path)
                self._own_model = True
            from vosk import KaldiRecognizer

            self._recognizer = KaldiRecognizer(self._vosk_model, self.sample_rate)
        except Exception as exc:
            logger.error("Ambient hearing unavailable (Vosk load failed): %s", exc)
            await self._post_state("error", f"STT load failed: {exc}")
            return False

        self._segmenter = HearingSegmenter(
            self._is_speech,
            sample_rate=self.sample_rate,
            silence_ms=self.segment_silence_ms,
            max_segment_s=self.max_segment_s,
            noise_rms_threshold=self.noise_rms_threshold,
        )
        blocksize = (self.sample_rate * FRAME_MS) // 1000
        device = None if str(self.device).lower() == "default" else self.device
        try:
            if isinstance(device, str) and device.isdigit():
                device = int(device)
            self._stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=blocksize,
                device=device,
                callback=self._audio_callback,
            )
            self._stream.start()
        except Exception as exc:
            logger.error("Ambient hearing unavailable (mic open failed): %s", exc)
            await self._post_state("error", f"mic open failed: {exc}")
            return False

        self._running = True
        self._drain_task = asyncio.create_task(self._drain())
        logger.info("Ambient hearing started (device=%s, vad=%d)", self.device, self.vad_aggressiveness)
        await self._post_state("listening")
        return True

    async def stop(self) -> None:
        if not self._running:
            await self._post_state("off")
            return
        self._running = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception as exc:
            logger.debug("hearing stream close failed: %s", exc)
        finally:
            self._stream = None
        # The sentinel is the drain task's only exit signal when the queue
        # is empty, so it must get in even when the queue is full
        # (transcription behind): drop a stale frame to make room. If the
        # queue refills before the retry (an in-flight mic callback), the
        # drain still exits via the _running check on the next frame it
        # consumes, so a failed retry is safe to swallow.
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        if self._drain_task is not None:
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None
        self._recognizer = None
        logger.info("Ambient hearing stopped")
        await self._post_state("off")

    async def set_enabled(self, value: bool) -> None:
        value = bool(value)
        self.enabled = value
        if value:
            started = await self.start()
            if not started:
                self.enabled = False
        else:
            await self.stop()

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        # PortAudio thread: never block, never raise — drop when full.
        if status:
            logger.debug("hearing callback status: %s", status)
        try:
            self._queue.put_nowait(bytes(indata))
        except queue.Full:
            pass

    async def _drain(self) -> None:
        """Consume queued frames, segment, transcribe, deliver evidence."""
        while True:
            frame = await asyncio.to_thread(self._queue.get)
            if frame is None or not self._running:  # stop() sentinel / shutdown
                break
            if self._segmenter is None:
                continue
            try:
                events = self._segmenter.feed(frame)
            except Exception as exc:
                logger.debug("hearing segmentation failed: %s", exc)
                continue
            for kind, payload in events:
                if kind == "speech":
                    text = await asyncio.to_thread(self._transcribe, payload)
                    if text and len(text) >= self.min_transcript_chars:
                        await self._emit_evidence(text, "speech")
                elif kind == "noise":
                    await self._emit_evidence("a loud noise", "noise")

    def _transcribe(self, pcm: bytes) -> str:
        """Transcribe one finished segment; returns "" when undecipherable."""
        if self._recognizer is None:
            return ""
        try:
            # Feed in chunks; AcceptWaveform returns True when a final
            # result is ready, FinalResult flushes the tail.
            for offset in range(0, len(pcm), 4000):
                self._recognizer.AcceptWaveform(pcm[offset:offset + 4000])
            result = json.loads(self._recognizer.FinalResult() or "{}")
            if hasattr(self._recognizer, "Reset"):
                self._recognizer.Reset()
            return (result.get("text") or "").strip()
        except Exception as exc:
            logger.debug("hearing transcription failed: %s", exc)
            return ""

    async def _emit_evidence(self, text: str, kind: str) -> None:
        """Hand one heard thing to the idle loop + dashboard."""
        if self._cfg.get("retain_transcripts", True):
            logger.info("Hearing (%s): '%s'", kind, text[:80])
        else:
            # Retention off: room speech must not land in persistent logs
            # either — the bridge writes INFO to logs/nomous.log. Metadata
            # only; the transcript itself goes to the idle loop in memory.
            logger.info("Hearing (%s): %d chars (transcript retention off)", kind, len(text))
        await self._post_state("listening", text=text, kind=kind)
        if self._llm is not None:
            try:
                await self._llm.note_audio_evidence(text, kind)
            except Exception as exc:
                logger.debug("hearing evidence delivery failed: %s", exc)
