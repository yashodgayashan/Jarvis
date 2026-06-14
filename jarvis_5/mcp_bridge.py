"""MCPBridge — spawn MCP stdio servers, list their tools, register them.

The whole point of MCP: a small `config.yaml` change → many new tools. The
agent loop doesn't care that they live in another process — they look like
any other entry in the registry.

The trick is `AsyncExitStack`: every `enter_async_context` we make gets unwound
in reverse order on `stop_all()`, so one call cleans up every subprocess and
session at shutdown.
"""
from __future__ import annotations

import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from tools import REGISTRY, Tool

log = logging.getLogger(__name__)


class MCPBridge:
    def __init__(self, servers_cfg: list[dict[str, Any]], log_dir: Path | None = None) -> None:
        self.servers_cfg = servers_cfg
        self._stack = AsyncExitStack()
        self.sessions: dict[str, Any] = {}
        self.log_dir = log_dir or Path("logs")

    async def start_all(self) -> None:
        # Open the AsyncExitStack: it records every context we enter below and
        # unwinds them all (subprocesses + sessions) in reverse on stop_all().
        await self._stack.__aenter__()
        try:
            # ClientSession speaks JSON-RPC (MCP's request/response format) to a
            # server; stdio_client launches it as a subprocess and talks over its
            # standard input/output (stdio).
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as e:
            raise RuntimeError("mcp SDK not installed (pip install mcp).") from e

        self.log_dir.mkdir(parents=True, exist_ok=True)

        for cfg in self.servers_cfg:
            name = cfg["name"]
            command = cfg["command"]
            args = [self._expand(a) for a in cfg.get("args", [])]
            env = {**os.environ, **cfg.get("env", {})}

            log.info(f"starting MCP server {name!r}: {command} {' '.join(args)}")
            # Capture the subprocess's stderr to a per-server log file for debugging.
            errlog = open(self.log_dir / f"mcp_{name}.log", "wb")
            try:
                params = StdioServerParameters(command=command, args=args, env=env)
                # Launch the subprocess; read/write are the stdio pipes to it.
                # enter_async_context registers it for automatic cleanup later.
                read, write = await self._stack.enter_async_context(
                    stdio_client(params, errlog=errlog)
                )
                session = await self._stack.enter_async_context(ClientSession(read, write))
                # MCP handshake: agree on protocol version/capabilities before use.
                await session.initialize()
                self.sessions[name] = session

                # Ask the server what tools it offers, then expose each one to the agent.
                tools_resp = await session.list_tools()
                self._register_tools(name, tools_resp.tools)
                log.info(f"  ↳ {name}: registered {len(tools_resp.tools)} tools")
            except Exception as e:
                log.error(f"MCP server {name!r} failed to start: {e}")
                log.error(f"  see {self.log_dir / f'mcp_{name}.log'} for stderr")
                continue

    def _register_tools(self, server_name: str, mcp_tools: list[Any]) -> None:
        for t in mcp_tools:
            REGISTRY.register(
                Tool(
                    # Namespace the name so tools from different servers can't collide.
                    name=f"mcp_{server_name}_{t.name}",
                    description=t.description or f"MCP tool from {server_name}",
                    # inputSchema is the tool's JSON Schema (its parameter description).
                    parameters=t.inputSchema or {"type": "object", "properties": {}},
                    # The proxy is a local wrapper that forwards the call to the subprocess.
                    func=self._make_proxy(server_name, t.name),
                    source=f"mcp:{server_name}",
                )
            )

    def _make_proxy(self, server_name: str, tool_name: str):
        # Returns an async proxy: when the agent calls this tool, forward the call
        # over the MCP session to the subprocess and return its text output.
        async def _call(**kwargs):
            session = self.sessions.get(server_name)
            if session is None:
                return f"MCP server {server_name!r} is not running."
            try:
                result = await session.call_tool(tool_name, kwargs)
            except Exception as e:
                return f"MCP tool {tool_name!r} on {server_name!r} failed: {e}"
            return "\n".join(c.text for c in result.content if hasattr(c, "text"))

        return _call

    @staticmethod
    def _expand(arg: str) -> str:
        # Expand ~ and $VARS in config args (e.g. "~/notes", "$HOME").
        return os.path.expanduser(os.path.expandvars(arg))

    async def stop_all(self) -> None:
        # One call closes every context entered in start_all, in reverse order:
        # sessions first, then their subprocesses.
        await self._stack.__aexit__(None, None, None)
        self.sessions.clear()
