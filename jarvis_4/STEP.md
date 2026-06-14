# jarvis_4 — tools + dispatch

## What's new
- `tools.py` — `Tool` dataclass, `ToolRegistry`, `@tool(description)` decorator that introspects the signature into a JSON Schema, plus three builtins: `get_time`, `read_file`, `web_search`.
- `agent.py` — `Agent.turn(...)` with a tool-dispatch hop loop. The model can call a tool, see the result, call another, and reply.
- `jarvis.py` is now thin: argparse, build Agent, run REPL.
- New `--verbose` flag — prints `→ tool name({args})` / `← result` lines so you can SEE the agent thinking.

## What you learn
- The whole tool calling pattern, written from scratch in ~100 lines:
  1. Decorator wraps the function and introspects its signature.
  2. JSON Schema is the contract the LLM sees.
  3. The model returns a `tool_calls` field instead of (or alongside) text.
  4. You dispatch, append a `role=tool` message, and stream again.
- Why we cap at `MAX_TOOL_ITERATIONS = 5` — protection against infinite loops.
- Why `dispatch` returns a **string** even on failure: the model can read errors and apologize, instead of crashing the agent.
- Sandboxing: `_safe_path` rejects `../../etc/passwd`-style escapes.

## Prereqs (in addition to jarvis_3)
- The Ollama model needs to support tool calling. `qwen2.5:7b` does; `qwen2.5:3b` does too. Llama 3.1+ models work as well.

## Run
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python jarvis.py --text --mute --verbose
```

Smoke test — the headline feature:
```
printf "what time is it right now? call get_time.\nquit\n" \
  | python jarvis.py --text --mute --verbose 2>&1 | grep "→ tool"
```
Expect a line like `→ tool get_time({})`.

Try also: "search the web for the world cup winner in 2022", "read the file `notes.txt`" (after `echo hello > ~/jarvis_workspace/notes.txt`).

## Files
- `jarvis.py` (~75 LOC) — entrypoint, REPLs, glue.
- `agent.py` (~110 LOC) — `Agent.turn` + `_stream_one`.
- `tools.py` (~145 LOC) — registry + decorator + 3 builtins.
- `audio.py`, `tts.py`, `prompts/`, `voices/` — unchanged.
- `requirements.txt` — adds `duckduckgo-search`.

## Evolve into jarvis_5
Hour 5 adds **MCP** — point Jarvis at an external tool server and watch dozens of new tools appear in the same registry, without changing the agent loop.
1. Add `mcp_bridge.py`: `MCPBridge` class with an `AsyncExitStack`. `start_all()` spawns each MCP server via `stdio_client`, calls `session.list_tools()`, and registers each as a `Tool` (proxy closure that calls `session.call_tool`).
2. Add `config.yaml` — list the MCP servers (`filesystem`, `everything`).
3. Add `mcp>=1.2`, `PyYAML>=6.0.2` to requirements.
4. Switch the main loop to **async** — `asyncio.run(main())`. `Agent.turn` becomes async because `dispatch` is now async (MCP tools are async). The text streaming inside `_stream_one` can stay using `httpx.AsyncClient` for a clean async story.
5. Smoke test: `python jarvis.py --text --verbose` then ask "use mcp_filesystem_list_directory on path /Users/you/jarvis_workspace". Expect a `→ tool mcp_filesystem_list_directory(...)` log line.
