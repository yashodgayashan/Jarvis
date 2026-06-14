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
import sounddevice as sd
# webrtcvad: VAD = Voice Activity Detection — tells if a frame contains speech
import webrtcvad

log = logging.getLogger(__name__)

# sample rate: audio samples per second; 16 kHz is what Whisper expects
SAMPLE_RATE = 16000
# frame/block length in ms; VAD = Voice Activity Detection
BLOCK_MS = 30  # 10/20/30 are the only VAD-legal frame sizes
# samples per block = how many audio samples make up one 30 ms frame
BLOCK_SAMPLES = int(SAMPLE_RATE * BLOCK_MS / 1000)


def record_until_silence(
    *,
    aggressiveness: int = 2,
    silence_ms: int = 800,
    max_utterance_s: float = 30.0,
    lead_in_ms: int = 300,
) -> bytes:
    """Block until the user speaks then stops talking. Return raw int16 PCM bytes.

    The lead-in ring buffer keeps ~300 ms of audio from just BEFORE speech was
    detected, so the first word survives.
    """
    vad = webrtcvad.Vad(aggressiveness)
    # how many consecutive silent frames count as "done talking"
    silence_blocks = max(1, silence_ms // BLOCK_MS)
    lead_in_blocks = max(1, lead_in_ms // BLOCK_MS)
    max_blocks = int(max_utterance_s * 1000 / BLOCK_MS)

    # lead-in ring buffer: holds the most recent frames from just before speech
    lead_in: collections.deque[bytes] = collections.deque(maxlen=lead_in_blocks)
    speech: list[bytes] = []
    silence_run = 0
    spoken = False
    started_at = time.time()

    # queue bridges the audio callback thread to this loop
    q: queue.Queue[bytes] = queue.Queue(maxsize=200)

    # sounddevice calls this from a background thread for every captured block
    def on_audio(indata, frames, time_info, status):
        try:
            q.put_nowait(bytes(indata))
        except queue.Full:
            try:
                q.get_nowait()
                q.put_nowait(bytes(indata))
            except queue.Empty:
                pass

    # open the mic stream; dtype="int16" = 16-bit PCM samples, mono
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

            # ask the VAD whether this 30 ms frame contains speech
            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if not spoken:
                # not talking yet: keep recent frames in the lead-in buffer
                lead_in.append(frame)
                if is_speech:
                    spoken = True
                    # prepend the buffered lead-in so the first word isn't clipped
                    speech.append(b"".join(lead_in))
                    speech.append(frame)
            else:
                speech.append(frame)
                if is_speech:
                    silence_run = 0
                else:
                    # count trailing silence; enough of it ends the utterance
                    silence_run += 1
                    if silence_run >= silence_blocks:
                        break

            if len(speech) > max_blocks:
                log.warning("max utterance length reached — cutting off")
                break

    return b"".join(speech)


def _pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """int16 PCM bytes → float32 [-1, 1] mono. What faster-whisper wants."""
    arr = np.frombuffer(pcm_bytes, dtype=np.int16)
    return (arr.astype(np.float32) / 32768.0).copy()


# lru_cache so the (heavy) model is loaded once and reused
@lru_cache(maxsize=2)
def _load_whisper(model_name: str, compute_type: str):
    """First load takes a few seconds; cached after that."""
    # Whisper: the local STT model that turns audio into text
    from faster_whisper import WhisperModel

    log.info(f"loading whisper {model_name!r} ({compute_type}) — this takes a few seconds")
    return WhisperModel(model_name, compute_type=compute_type, device="cpu")


class WhisperSTT:
    """Local Whisper transcriber. Default: small.en, int8 — CPU-friendly."""

    def __init__(self, model: str = "small.en", compute_type: str = "int8") -> None:
        self._model = _load_whisper(model, compute_type)

    def transcribe(self, pcm_bytes: bytes) -> str:
        arr = _pcm_to_float32(pcm_bytes)
        if len(arr) < 1000:
            return ""
        # faster-whisper's mel matmul leaks spurious FP-exception flags on
        # numpy 2.0.x / macOS (matmul does no division, so "divide by zero in
        # matmul" can't be real). Silence them — the result is correct.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            segments, _ = self._model.transcribe(
                arr, language="en", vad_filter=False, beam_size=5
            )
            return "".join(s.text for s in segments).strip()
