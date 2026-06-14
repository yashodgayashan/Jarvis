# jarvis_5 — MCP bridge

## What's new
- `mcp_bridge.py` — `MCPBridge` class. `AsyncExitStack` for clean lifecycle, `stdio_client` for spawning servers, `session.list_tools()` to discover them, and a closure factory that wraps each MCP tool as a `Tool` in the same `REGISTRY`.
- `config.yaml` — declarative server list. Add a server, restart, get dozens of new tools — without touching the agent loop.
- Everything went **async**. `Agent.turn`, `ToolRegistry.dispatch`, `_stream_one` (now `httpx.AsyncClient` + `aiter_lines`), and the main entrypoint (`asyncio.run(main())`).
- `--no-mcp` flag for faster startup when you don't want to spawn npx subprocesses.

## What you learn
- The `AsyncExitStack` pattern: every `enter_async_context` is unwound in reverse on a single `__aexit__` — no leaked subprocesses.
- The MCP stdio protocol: spawn process, talk JSON-RPC over stdin/stdout, `initialize`, `list_tools`, `call_tool`.
- That an MCP tool **and** a `@tool`-decorated Python function are interchangeable from the agent's point of view. The registry doesn't care; the model doesn't care.
- How sync code morphs into async — `with x.stream(...)` → `async with x.stream(...)`, `for line in resp.iter_lines()` → `async for line in resp.aiter_lines()`. Same shape, different keyword.
- Running blocking calls (mic, STT) from async code via `run_in_executor`.

## Prereqs (in addition to jarvis_4)
- **Node 20+** with `npx` on PATH (the MCP servers run via `npx`).
- A workspace dir to point the filesystem server at:
  ```
  mkdir -p ~/jarvis_workspace
  echo "hello jarvis" > ~/jarvis_workspace/marker.txt
  ```

## Run
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python jarvis.py --text --mute --verbose
```

Startup logs should show:
```
starting MCP server 'filesystem': npx -y @modelcontextprotocol/server-filesystem ~/jarvis_workspace
  ↳ filesystem: registered N tools
```

Smoke test the bridge:
```
printf "use mcp_filesystem_list_directory with path /Users/$USER/jarvis_workspace\nquit\n" \
  | python jarvis.py --text --mute --verbose 2>&1 | grep "mcp_filesystem"
```
Expect both `→ tool mcp_filesystem_list_directory(...)` AND the reply mentioning `marker.txt`.

## Files
- `jarvis.py` (~90 LOC) — async main, bridge lifecycle.
- `agent.py` (~110 LOC) — async version of jarvis_4's Agent.
- `tools.py` (~155 LOC) — `dispatch` is async; builtins unchanged.
- `mcp_bridge.py` (~95 LOC) — server lifecycle + proxy registration.
- `config.yaml` — declarative server list.
- `audio.py`, `tts.py`, `prompts/`, `voices/` — unchanged.
- `requirements.txt` — adds `mcp`, `PyYAML`.

## Evolve into jarvis_6
Hour 6 makes Jarvis truly **hands-free**. No more Enter-to-talk.
1. Add `wake.py` — `PorcupineWake` (real wake-word, needs `PORCUPINE_KEY` env), `SubstringWake` (fallback that watches STT transcripts), and a `build_wake(engine, keyword)` factory.
2. Add `state_machine.py` — a `Mode` enum (`IDLE / LISTENING / THINKING / SPEAKING`) and `run_forever(agent, wake, stt, tts)` that loops the four states.
3. Add `pvporcupine>=3.0.2` to requirements; sign up at https://console.picovoice.ai for a free key.
4. Add `assets/ding.wav` — a short notification sound played on wake-up. (On mac: `say "ding" -o assets/ding.wav --data-format=LEF32@22050`.)
5. Switch `jarvis.py` default mode to `state_machine.run_forever(...)`; keep `--push-to-talk` as the old behavior.
6. Smoke test: `python jarvis.py` → say "Jarvis" → hear ding → ask a question → hear reply → it returns to IDLE.
