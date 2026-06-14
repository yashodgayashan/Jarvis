# VAD = Voice Activity Detection (is this audio chunk speech?); STT = Speech-to-Text.
"""Mic capture + VAD + Whisper STT — flat, free functions where possible.

Ported (and flattened) from jarvis/jarvis/audio/{mic,vad,stt}.py.
"""
from __future__ import annotations

import collections
import logging
import queue
import time
from functools import lru_cache 

import numpy as np
# sounddevice: reads raw audio from the microphone; webrtcvad: the VAD speech detector.
import sounddevice as sd
import webrtcvad

log = logging.getLogger(__name__)

# sample rate = audio samples per second; 16 kHz is what Whisper and the VAD expect.
SAMPLE_RATE = 16000  # 16Hz
# A "block"/"frame" is one small slice of audio; VAD only accepts 10/20/30 ms frames.
BLOCK_MS = 30  # 10/20/30 are the only VAD-legal frame sizes
# Number of audio samples in one block (sample rate × block duration).
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_MS / 1000)


def record_until_silence(
    *,
    aggressiveness: int = 2,  #0-3 (3 is more strict)
    silence_ms: int = 800, 
    max_utterance_s: float = 30.0,
    lead_in_ms: int = 300,
) -> bytes:
    """Block until the user speaks then stops talking. Return raw int16 PCM bytes.

    The lead-in ring buffer keeps ~300 ms of audio from just BEFORE speech was
    detected, so the first word survives.
    """
    # int16 PCM = raw uncompressed audio samples as 16-bit integers.
    vad = webrtcvad.Vad(aggressiveness)
    # How many consecutive silent blocks count as "done speaking".
    silence_blocks = max(1, silence_ms // BLOCK_MS)
    # How many blocks of pre-speech audio to retain (the lead-in).
    lead_in_blocks = max(1, lead_in_ms // BLOCK_MS)
    max_blocks = int(max_utterance_s * 1000 / BLOCK_MS)

    # ring buffer: a fixed-size deque; once full, old blocks fall off the front.
    # Holds the lead-in so the start of the first word isn't clipped.
    lead_in: collections.deque[bytes] = collections.deque(maxlen=lead_in_blocks)
    speech: list[bytes] = []
    silence_run = 0
    spoken = False
    started_at = time.time()

    # Audio arrives on a background thread; this queue hands frames to the main loop.
    q: queue.Queue[bytes] = queue.Queue(maxsize=200)

    # Called by sounddevice for each captured block; just enqueues the raw bytes.
    def on_audio(indata, frames, time_info, status):
        try:
            q.put_nowait(bytes(indata))
        except queue.Full:
            # Queue full: drop the oldest frame so we keep the freshest audio.
            try:
                q.get_nowait()
                q.put_nowait(bytes(indata))
            except queue.Empty:
                pass

    # Open the mic as a raw int16 mono stream at our chosen sample rate / block size.
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SAMPLES,
        dtype="int16",
        channels=1,
        callback=on_audio,
    ):
        while True:
            try:
                frame = q.get(timeout=0.5)
            except queue.Empty:
                if time.time() - started_at > 60 and not spoken:
                    log.warning("no speech detected in 60s — bailing")
                    break
                continue

            # Ask the VAD whether this frame contains speech.
            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if not spoken:
                # Still waiting for speech: keep buffering into the lead-in ring.
                lead_in.append(frame)
                if is_speech:
                    # Speech just started: prepend the buffered lead-in, then this frame.
                    spoken = True
                    speech.append(b"".join(lead_in))
                    speech.append(frame)
            else:
                speech.append(frame)
                if is_speech:
                    # Reset the silence counter whenever speech resumes.
                    silence_run = 0
                else:
                    # Enough trailing silence in a row → the utterance is over.
                    silence_run += 1
                    if silence_run >= silence_blocks:
                        break

            if len(speech) > max_blocks:
                log.warning("max utterance length reached — cutting off")
                break

    return b"".join(speech)


def _pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """int16 PCM bytes → float32 [-1, 1] mono. What faster-whisper wants."""
    # float32 = audio samples as floats in [-1, 1]; divide by 32768 (max int16) to scale.
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    return (arr.astype(np.float32) / 32768.0).copy()


# lru_cache: load each Whisper model once, then reuse the cached instance.
@lru_cache(maxsize=2)
def _load_whisper(model_name: str, compute_type: str):
    """First load takes a few seconds; cached after that."""
    # Whisper = the local speech-to-text model (here via the faster-whisper package).
    from faster_whisper import WhisperModel

    log.info(f"loading whisper {model_name!r} ({compute_type}) — this takes a few seconds")
    return WhisperModel(model_name, compute_type=compute_type, device="cpu")


class WhisperSTT:
    """Local Whisper transcriber. Default: small.en, int8 — CPU-friendly."""

    def __init__(self, model: str = "small.en", compute_type: str = "int8") -> None:
        self._model = _load_whisper(model, compute_type)

    def transcribe(self, pcm_bytes: bytes) -> str:
        # Convert raw PCM into the float32 array Whisper expects.
        arr = _pcm_to_float32(pcm_bytes)
        # Too few samples (< ~60 ms) to be real speech — skip it.
        if len(arr) < 1000:
            return ""
        # Run STT; we already gated on speech ourselves, so vad_filter is off here.
        segments, _ = self._model.transcribe(
            arr, language="en", vad_filter=False, beam_size=5
        )
        # Stitch the segment texts into one transcript string.
        return "".join(s.text for s in segments).strip()
