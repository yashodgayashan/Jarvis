# REPL = Read-Eval-Print Loop, the interactive prompt; here it can take voice input too.
"""jarvis_2 — adds push-to-talk speech-in to the streaming REPL."""
import argparse
import json
import sys

# httpx: HTTP client; we use its streaming mode to read the reply token-by-token.
import httpx

# WhisperSTT = local Speech-to-Text (transcription); record_until_silence captures mic audio.
from audio import WhisperSTT, record_until_silence

# Ollama = local LLM (Large Language Model) server; this is its chat endpoint.
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"

SYSTEM = """You are Jarvis, a dry, unflappable British butler who happens to have
a doctorate in computer science. Address the user as "sir". Keep replies short —
one to three sentences unless asked for detail. Never use emojis or exclamation
marks. Speak in prose, not bullet points. You have no tools, so never claim to
have done something you cannot actually do (set a reminder, send a message, save
a file) — if you can't do it, say so plainly. You are useful, and faintly
entertaining while you're at it."""


def chat(messages):
    """Stream one assistant reply. Prints tokens as they arrive; returns full text."""
    # token = the small text fragments the LLM emits one at a time.
    reply = ""
    # httpx streaming: keep the connection open and read the response as it trickles in.
    with httpx.stream(
        "POST",
        OLLAMA_URL,
        json={"model": MODEL, "messages": messages, "stream": True},
        timeout=120.0,
    ) as resp:
        resp.raise_for_status()
        # Each streamed line is one JSON chunk carrying a token (or the "done" flag).
        for line in resp.iter_lines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            # Pull this chunk's token text out; missing fields default to "".
            text = (chunk.get("message") or {}).get("content") or ""
            if text:
                print(text, end="", flush=True)
                reply += text
            if chunk.get("done"):
                break
    print()
    return reply


def text_repl(messages):
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
        messages.append({"role": "user", "content": user})
        print("jarvis › ", end="", flush=True)
        reply = chat(messages)
        messages.append({"role": "assistant", "content": reply})


def voice_repl(messages):
    # STT = Speech-to-Text. Load the Whisper model now so the first transcription is fast.
    stt = WhisperSTT()  # eagerly load — first transcribe should be fast
    print("voice mode — press Enter to talk, Ctrl-C to quit.\n")
    while True:
        try:
            input("press Enter to talk › ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        print("(listening…)")
        # Capture mic audio until silence; pcm = raw PCM (Pulse-Code Modulation) audio bytes.
        pcm = record_until_silence()
        # Transcribe the captured audio into text.
        user = stt.transcribe(pcm)
        if not user:
            print("(heard nothing — try again)")
            continue
        print(f"you › {user}")
        messages.append({"role": "user", "content": user})
        print("jarvis › ", end="", flush=True)
        reply = chat(messages)
        messages.append({"role": "assistant", "content": reply})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="store_true", help="text mode (no mic) — for debugging")
    args = ap.parse_args()

    messages = [{"role": "system", "content": SYSTEM}]
    try:
        if args.text:
            text_repl(messages)
        else:
            voice_repl(messages)
    except httpx.ConnectError:
        print("\n[err] Could not reach Ollama at http://localhost:11434 — is `ollama serve` running?", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
