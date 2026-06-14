# REPL = Read-Eval-Print Loop: the interactive prompt that reads input, runs it, prints output, repeats.
"""jarvis_4 — argparse + Agent + thin REPLs. The brains live in agent.py now."""
import argparse
import logging
import sys
from pathlib import Path

import httpx

from agent import Agent
# WhisperSTT = local Speech-to-Text (Whisper) wrapper; record_until_silence captures mic audio.
from audio import WhisperSTT, record_until_silence
# TTS = Text-to-Speech (turns the reply text into spoken audio).
from tts import build_default_player
import tools  # noqa: F401 — importing registers the @tool builtins

# The local LLM (Large Language Model) Ollama will run for us.
MODEL = "qwen2.5:7b"

SYSTEM = Path(__file__).resolve().parent.joinpath("prompts/jarvis.txt").read_text(encoding="utf-8").strip()


def make_agent() -> Agent:
    return Agent(model=MODEL, system=SYSTEM)


def text_repl(agent: Agent, tts):
    while True:
        try:
            user = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            return
        print("jarvis › ", end="", flush=True)
        # on_token fires per streamed text fragment (print live); on_sentence fires
        # per finished sentence (hand it to TTS so speech starts before the reply ends).
        agent.turn(
            user,
            on_token=lambda t: print(t, end="", flush=True),
            on_sentence=(tts.speak if tts else None),
        )
        print()


def voice_repl(agent: Agent, tts):
    stt = WhisperSTT()
    print("voice mode — press Enter to talk, Ctrl-C to quit.\n")
    while True:
        try:
            input("press Enter to talk › ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        print("(listening…)")
        # pcm = raw PCM (Pulse-Code Modulation) audio samples captured from the mic.
        pcm = record_until_silence()
        # STT (Speech-to-Text): turn the recorded audio into a text string.
        user = stt.transcribe(pcm)
        if not user:
            print("(heard nothing — try again)")
            continue
        print(f"you › {user}")
        print("jarvis › ", end="", flush=True)
        agent.turn(
            user,
            on_token=lambda t: print(t, end="", flush=True),
            on_sentence=tts.speak,
        )
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="store_true", help="text mode (no mic)")
    ap.add_argument("--mute", action="store_true", help="disable TTS")
    ap.add_argument("--verbose", "-v", action="store_true", help="log tool calls")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    tts = None if args.mute else build_default_player()
    agent = make_agent()
    try:
        if args.text:
            text_repl(agent, tts)
        else:
            voice_repl(agent, tts)
    except httpx.ConnectError:
        # Ollama is the local LLM server; the agent talks to it over HTTP on port 11434.
        print("\n[err] Could not reach Ollama at http://localhost:11434 — is `ollama serve` running?", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
