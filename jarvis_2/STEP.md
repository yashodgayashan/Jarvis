# jarvis_2 — push-to-talk speech in

## What's new
- `audio.py` — mic capture + WebRTC VAD + faster-whisper STT (small.en, int8).
- `record_until_silence()` blocks until you speak then stop; returns raw PCM bytes.
- `WhisperSTT.transcribe()` — local transcription, no network.
- `jarvis.py` gains a `--text` flag so the old keyboard mode still works.
- Default mode is now voice: press Enter, speak, hear silence, see your transcript and a reply.

## What you learn
- The mic → VAD → STT pipeline, end to end, in <100 lines.
- Why we use a 300 ms lead-in ring buffer (first-word survival).
- Why webrtcvad needs **exactly** 10/20/30 ms frames at 16 kHz.
- That all of this runs **locally**: no cloud STT, no API keys, no usage caps.

## Prereqs (in addition to jarvis_1)
- A working microphone, and OS-level mic permission for your terminal:
  - **macOS:** System Settings → Privacy & Security → Microphone → enable your terminal.
  - **Linux:** check `arecord -l`; user in the `audio` group.
  - **Windows:** Settings → Privacy → Microphone.
- The first run downloads the Whisper `small.en` model (~250 MB) into `~/.cache/huggingface/`.

## Run
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python jarvis.py            # voice mode (default)
python jarvis.py --text     # the jarvis_1 mode, kept alive for debugging
```

Voice flow:
```
press Enter to talk › <Enter>
(listening…)
you › what's the capital of France?
jarvis › Paris, sir.
```

Text mode smoke test (works without a mic):
```
printf "hi\nquit\n" | python jarvis.py --text
```

## Files
- `jarvis.py` (~90 LOC) — argparse, text REPL, voice REPL, chat.
- `audio.py` (~110 LOC) — mic, VAD, Whisper. Flat free functions plus one `WhisperSTT` class.
- `requirements.txt` — adds `sounddevice`, `numpy`, `webrtcvad-wheels`, `faster-whisper`.

## Evolve into jarvis_3
Hour 3 adds **speech out** — and (because the prompt is about to get long) extracts it to a file.
1. Create `prompts/jarvis.txt`, move the `SYSTEM` constant from `jarvis.py` into it. Load with `Path("prompts/jarvis.txt").read_text()`.
2. Add `tts.py` with `PiperTTS`, `EdgeTTS` (fallback), `SentenceSplitter`, `TTSPlayer`.
3. Add `edge-tts>=7.2,<8` to `requirements.txt`. Install `piper-tts` via your OS package manager / pip wheel.
4. In the streaming loop, feed every token into `SentenceSplitter` and call `tts.speak(sentence)` on each complete sentence. This is the big win: Jarvis starts SPEAKING before he finishes WRITING.
5. Drop the Piper voice files into `voices/` (the master scaffold has `en_US-lessac-medium`).
