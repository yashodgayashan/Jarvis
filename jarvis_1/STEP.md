# jarvis_1 — streaming text REPL

## What's new
- Everything. This is the first snapshot.
- One file (`jarvis.py`), one dependency (`httpx`), ~50 lines of code.
- Streams replies from a local Ollama server token-by-token.
- Inline `SYSTEM` prompt — the butler personality lives in a string constant for now.

## What you learn
- HTTP streaming with `httpx.stream(...)` and `resp.iter_lines()`.
- The Ollama `/api/chat` JSON-line protocol: `{"message": {"content": "..."}, "done": false}`.
- A minimal chat loop: a `messages` list that grows on every turn.
- How little code it takes for an LLM to already feel "in character".

## Prereqs
1. Ollama running locally: `ollama serve` (macOS: `open -a Ollama`).
2. The model pulled: `ollama pull qwen2.5:7b` (or `qwen2.5:3b` if you have <16 GB RAM — edit `MODEL` in `jarvis.py`).

## Run
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python jarvis.py
```

Then type:
```
you › what's the capital of France?
jarvis › Paris, sir.
you › quit
```

Smoke-test it non-interactively:
```
printf "capital of France?\nquit\n" | python jarvis.py
```

## Files
- `jarvis.py` — entrypoint, the whole thing.
- `requirements.txt` — just `httpx`.

## Evolve into jarvis_2
Hour 2 adds **speech in** — press Enter to talk, see your transcript, get a reply.
1. Add a new file `audio.py` with `record_until_silence(...)` (mic + VAD) and a `WhisperSTT` class.
2. Add `sounddevice`, `numpy`, `webrtcvad-wheels`, `faster-whisper` to `requirements.txt`.
3. Add an `argparse` flag `--text` to `jarvis.py` (keeps today's text mode alive for debugging).
4. In voice mode: `input("press Enter to talk ")` → `record_until_silence()` → `stt.transcribe()` → feed into the same chat loop you already have. No TTS yet — you still read the reply.
