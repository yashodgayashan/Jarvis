"""jarvis_3 — add speech out. Sentence-by-sentence TTS while the LLM streams."""
import argparse
import json
import sys
from pathlib import Path

# httpx: HTTP client used here for streaming the LLM response chunk by chunk
import httpx

# WhisperSTT = local Speech-to-Text (transcription); record_until_silence = mic capture
from audio import WhisperSTT, record_until_silence
from tts import SentenceSplitter, build_default_player

# Ollama = local LLM (Large Language Model) server we POST chat requests to
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"

SYSTEM = Path(__file__).resolve().parent.joinpath("prompts/jarvis.txt").read_text(encoding="utf-8").strip()


def chat(messages, *, on_sentence=None):
    """Stream one reply. Print tokens, hand each complete sentence to on_sentence."""
    reply = ""
    # splitter collects streamed text and emits whole sentences for TTS
    splitter = SentenceSplitter()
    # stream=True: read the reply incrementally instead of waiting for it all
    with httpx.stream(
        "POST",
        OLLAMA_URL,
        json={"model": MODEL, "messages": messages, "stream": True},
        timeout=120.0,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            # each chunk carries a token: a small text fragment of the reply
            text = (chunk.get("message") or {}).get("content") or ""
            if text:
                print(text, end="", flush=True)
                reply += text
                if on_sentence:
                    # speak each sentence as soon as it's complete (streaming TTS)
                    for sent in splitter.feed(text):
                        on_sentence(sent)
            if chunk.get("done"):
                if on_sentence:
                    # flush any leftover text that never got final punctuation
                    tail = splitter.flush()
                    if tail:
                        on_sentence(tail)
                break
    print()
    return reply


# REPL = Read-Eval-Print Loop: the interactive prompt loop (typed input here)
def text_repl(messages, tts):
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
        reply = chat(messages, on_sentence=tts.speak if tts else None)
        messages.append({"role": "assistant", "content": reply})


# Same loop as text_repl, but input comes from the mic via STT instead of typing
def voice_repl(messages, tts):
    stt = WhisperSTT()
    print("voice mode — press Enter to talk, Ctrl-C to quit.\n")
    while True:
        try:
            input("press Enter to talk › ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        print("(listening…)")
        # pcm = raw PCM (Pulse-Code Modulation) audio bytes captured from the mic
        pcm = record_until_silence()
        # turn the recorded audio into text
        user = stt.transcribe(pcm)
        if not user:
            print("(heard nothing — try again)")
            continue
        print(f"you › {user}")
        messages.append({"role": "user", "content": user})
        print("jarvis › ", end="", flush=True)
        reply = chat(messages, on_sentence=tts.speak)
        messages.append({"role": "assistant", "content": reply})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="store_true", help="text mode (no mic)")
    ap.add_argument("--mute", action="store_true", help="disable TTS")
    args = ap.parse_args()

    # tts = the Text-to-Speech player (None when --mute disables speech out)
    tts = None if args.mute else build_default_player()
    messages = [{"role": "system", "content": SYSTEM}]
    try:
        if args.text:
            text_repl(messages, tts)
        else:
            voice_repl(messages, tts)
    except httpx.ConnectError:
        print("\n[err] Could not reach Ollama at http://localhost:11434 — is `ollama serve` running?", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
