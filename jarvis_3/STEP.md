# jarvis_3 — speech out (and a real prompts file)

## What's new
- `tts.py` — `PiperTTS` (local), `EdgeTTS` (cloud fallback), `SentenceSplitter`, `TTSPlayer`.
- `prompts/jarvis.txt` — the butler persona, extracted from `jarvis.py`. The string got too big for a `SYSTEM = "..."` constant — it's a real prompt now.
- `voices/en_US-lessac-medium.onnx` — Piper voice files committed alongside the code.
- The streaming loop now feeds every token into `SentenceSplitter` and calls `tts.speak(sentence)` the moment a sentence completes. Jarvis starts speaking before he finishes writing.
- New `--mute` flag for testing without audio.

## What you learn
- The "sentence splitter" trick: punctuation-boundary detection on a streaming buffer.
- Why streaming TTS is night-and-day better than wait-for-full-reply TTS — perceived latency drops from "seconds" to "instant".
- Two TTS engines behind one interface (`TTSPlayer` with fallback) — the same pattern you'll use for LLM providers later.
- Why we shell out to `piper` and play with `sounddevice`, but shell out to `ffplay` for Edge — Piper gives us WAV (easy), Edge gives us MP3 (needs ffmpeg).

## Prereqs (in addition to jarvis_2)
- `pip install piper-tts` (the `piper` CLI lands in `.venv/bin`).
- `ffmpeg` on PATH — `brew install ffmpeg` / `apt install ffmpeg` / Chocolatey on Windows. Needed by the Edge fallback to play MP3.

## Run
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python jarvis.py            # voice + TTS (the full experience)
python jarvis.py --text     # type, but reply is spoken
python jarvis.py --text --mute  # silent debugging
```

Listen for: Jarvis says "Paris" before the full sentence has finished streaming on screen. That's the win.

## Files
- `jarvis.py` (~95 LOC) — chat now takes `on_sentence`; argparse adds `--mute`.
- `audio.py` (unchanged from jarvis_2).
- `tts.py` (~140 LOC) — engines + splitter + player.
- `prompts/jarvis.txt` — the persona, in prose.
- `voices/en_US-lessac-medium.onnx` + `.onnx.json` — Piper model.
- `requirements.txt` — adds `edge-tts`.

## Evolve into jarvis_4
Hour 4 introduces **tools**. Right now Jarvis hallucinates if you ask "what time is it?" — we'll fix that.
1. Add `tools.py`: a `Tool` dataclass, a `@tool(description)` decorator that introspects the function signature into a JSON Schema, and a `ToolRegistry` with `register/get/tool_specs/dispatch`. Define 3 builtins in the same file: `get_time`, `read_file`, `web_search`.
2. Add `agent.py`: move the chat loop into `Agent.turn(...)`. Add a tool-dispatch hop loop (`MAX_TOOL_ITERATIONS = 5`). Handle Ollama's `tool_calls` field in the streamed chunks.
3. Slim `jarvis.py` down to: argparse, instantiate `Agent`, thin REPL/voice wrappers.
4. Add `duckduckgo-search` to requirements (for `web_search`).
5. Smoke test: ask "what time is it?". You should see `→ tool get_time({})` log a line, then a reply with the real time.
