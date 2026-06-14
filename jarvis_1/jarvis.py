import json
import sys

# httpx: HTTP client; we use its streaming mode to read the reply token-by-token.
import httpx

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
            if not line.strip(): # Ignore empty lines
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


def main():
    messages = [{"role": "system", "content": SYSTEM}]
    print("jarvis_1 — Ctrl-D or `quit` to exit.\n")
    # REPL = Read-Eval-Print Loop: read a line, send it to the model, print the reply, repeat.
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
        try:
            reply = chat(messages)
        except httpx.ConnectError:
            print("\n[err] Could not reach Ollama at http://localhost:11434 — is `ollama serve` running?", file=sys.stderr)
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
