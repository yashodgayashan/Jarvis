# TTS = Text-to-Speech. Piper = a local TTS engine (no internet); Edge = Microsoft's
# cloud TTS, used as a fallback.
"""Text-to-speech: Piper (local) with Edge (cloud) fallback, plus a sentence splitter.

The sentence splitter is what makes the assistant feel fast — we speak each
sentence as soon as the LLM finishes punctuating it, while it's still generating
the rest of the reply.
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
import tempfile
import wave
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

VOICES_DIR = Path(__file__).resolve().parent / "voices"


class TTSError(RuntimeError):
    pass


class TTSEngine(ABC):
    name: str

    @abstractmethod
    def speak(self, text: str) -> None:
        """Block until the sentence is fully spoken."""


class PiperTTS(TTSEngine):
    """Local Piper TTS — high quality, fast, no internet."""

    name = "piper"

    def __init__(self, voice: str = "en_US-lessac-medium") -> None:
        self.voice = voice
        self.model_path = VOICES_DIR / f"{voice}.onnx"
        if not self.model_path.exists():
            raise TTSError(
                f"Piper voice not found: {self.model_path}. "
                f"Download from https://huggingface.co/rhasspy/piper-voices."
            )
        if not shutil.which("piper"):
            raise TTSError(
                "`piper` CLI not on PATH. `pip install piper-tts` should install it."
            )

    def speak(self, text: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            subprocess.run(
                ["piper", "--model", str(self.model_path), "--output_file", wav_path],
                input=text,
                text=True,
                check=True,
                capture_output=True,
            )
            with wave.open(wav_path, "rb") as wf:
                sr = wf.getframerate()
                n_channels = wf.getnchannels()
                # Read the WAV's raw PCM (Pulse-Code Modulation) bytes as int16 samples.
                audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels)
            sd.play(audio, sr)
            sd.wait()
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
            raise TTSError(f"piper synth failed: {stderr[:200]}") from e
        finally:
            Path(wav_path).unlink(missing_ok=True)


class EdgeTTS(TTSEngine):
    """Cloud fallback via Microsoft Edge voices. Needs internet, no API key."""

    name = "edge"

    def __init__(self, voice: str = "en-US-JennyNeural") -> None:
        self.voice = voice

    def speak(self, text: str) -> None:
        try:
            import edge_tts
        except ImportError as e:
            raise TTSError("edge-tts not installed.") from e

        async def synth(out_path: Path) -> None:
            await edge_tts.Communicate(text, self.voice).save(str(out_path))

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            mp3_path = Path(f.name)
        try:
            asyncio.run(synth(mp3_path))
            try:
                subprocess.run(
                    ["ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet", str(mp3_path)],
                    check=True,
                )
            except FileNotFoundError as e:
                raise TTSError("ffplay not found (install ffmpeg).") from e
        finally:
            mp3_path.unlink(missing_ok=True)


class TTSPlayer:
    """Speak with a primary engine; fall back to another on failure."""

    def __init__(self, engine: TTSEngine, fallback: TTSEngine | None = None) -> None:
        self.engine = engine
        self.fallback = fallback

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        try:
            self.engine.speak(text)
        except TTSError as e:
            if self.fallback is None:
                raise
            log.warning(f"{self.engine.name} TTS failed ({e}); falling back to {self.fallback.name}")
            self.fallback.speak(text)


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


class SentenceSplitter:
    # token-deltas = the small text fragments the LLM streams out one at a time.
    """Feed token-deltas in; get back complete sentences as soon as they form."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: str) -> list[str]:
        if not chunk:
            return []
        self._buf += chunk
        parts = _SENTENCE_END.split(self._buf)
        if len(parts) <= 1:
            return []
        *complete, self._buf = parts
        return [s.strip() for s in complete if s.strip()]

    def flush(self) -> str:
        s, self._buf = self._buf.strip(), ""
        return s


def build_default_player() -> TTSPlayer:
    """Piper primary, Edge fallback. Falls back to Edge-only if Piper is missing."""
    try:
        return TTSPlayer(engine=PiperTTS(), fallback=EdgeTTS())
    except TTSError as e:
        log.warning(f"Piper unavailable ({e}); using Edge TTS only")
        return TTSPlayer(engine=EdgeTTS())
