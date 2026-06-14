"""jarvis_5 — async main loop with MCP bridge lifecycle."""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

import httpx
import yaml

from agent import Agent
# WhisperSTT = local Speech-to-Text (STT) engine; turns recorded audio into text.
from audio import WhisperSTT, record_until_silence
# MCP = Model Context Protocol: a standard for exposing external tools to an LLM,
# each tool server running as a separate subprocess. MCPBridge manages them.
from mcp_bridge import MCPBridge
from tts import build_default_player
import tools  # noqa: F401 — importing registers the @tool builtins

HERE = Path(__file__).resolve().parent
SYSTEM = HERE.joinpath("prompts/jarvis.txt").read_text(encoding="utf-8").strip()


def load_config() -> dict:
    return yaml.safe_load(HERE.joinpath("config.yaml").read_text(encoding="utf-8")) or {}


async def text_repl(agent: Agent, tts) -> None:
    # The event loop schedules all the async/await work on this thread.
    loop = asyncio.get_event_loop()
    while True:
        try:
            # input() blocks, so run_in_executor pushes it to a worker thread,
            # keeping the async event loop free instead of stalling on the prompt.
            user = (await loop.run_in_executor(None, input, "you › ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            return
        print("jarvis › ", end="", flush=True)
        await agent.turn(
            user,
            # on_token: print each token (a small text fragment) as the LLM streams it.
            on_token=lambda t: print(t, end="", flush=True),
            # on_sentence: hand each finished sentence to TTS (Text-to-Speech) to speak.
            on_sentence=(tts.speak if tts else None),
        )
        print()


async def voice_repl(agent: Agent, tts) -> None:
    stt = WhisperSTT()
    loop = asyncio.get_event_loop()
    print("voice mode — press Enter to talk, Ctrl-C to quit.\n")
    while True:
        try:
            await loop.run_in_executor(None, input, "press Enter to talk › ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        print("(listening…)")
        # Recording and transcribing both block, so run them off the event loop on
        # a worker thread. pcm = raw audio samples (Pulse-Code Modulation).
        pcm = await loop.run_in_executor(None, record_until_silence)
        # Whisper turns the recorded audio into text (STT).
        user = await loop.run_in_executor(None, stt.transcribe, pcm)
        if not user:
            print("(heard nothing — try again)")
            continue
        print(f"you › {user}")
        print("jarvis › ", end="", flush=True)
        await agent.turn(
            user,
            on_token=lambda t: print(t, end="", flush=True),
            on_sentence=tts.speak,
        )
        print()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", action="store_true", help="text mode (no mic)")
    ap.add_argument("--mute", action="store_true", help="disable TTS")
    ap.add_argument("--verbose", "-v", action="store_true", help="log tool calls")
    ap.add_argument("--no-mcp", action="store_true", help="skip MCP servers (faster startup)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    cfg = load_config()
    tts = None if args.mute else build_default_player()
    agent = Agent(
        model=cfg.get("model", "qwen2.5:7b"),
        ollama_url=cfg.get("ollama_url", "http://localhost:11434/api/chat"),
        system=SYSTEM,
    )

    bridge: MCPBridge | None = None
    if not args.no_mcp:
        # Spawn each configured MCP server subprocess and register its tools.
        bridge = MCPBridge(cfg.get("mcp", {}).get("servers") or [])
        await bridge.start_all()

    try:
        if args.text:
            await text_repl(agent, tts)
        else:
            await voice_repl(agent, tts)
    except httpx.ConnectError:
        # Ollama is the local LLM server we POST chat requests to.
        print("\n[err] Could not reach Ollama at http://localhost:11434 — is `ollama serve` running?", file=sys.stderr)
        sys.exit(1)
    finally:
        # Always tear down the MCP subprocesses and HTTP client on exit.
        if bridge:
            await bridge.stop_all()
        await agent.aclose()


if __name__ == "__main__":
    # asyncio.run sets up the event loop and runs the async main() to completion.
    asyncio.run(main())
