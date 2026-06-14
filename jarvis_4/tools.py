# LLM = Large Language Model. A "tool" is a named function the LLM can ask to run
# (function calling); JSON Schema = the machine-readable description of a tool's
# parameters that the LLM reads to know how to call it.
"""Tool system: dataclass + @tool decorator + registry + 3 builtins.

This file is deliberately self-contained — you'll see the whole tool mechanism
in one place: how a Python function becomes a JSON schema the LLM can call.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

# Maps Python types to their JSON Schema type names (what the LLM expects).
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(py_type: Any) -> str:
    return _TYPE_MAP.get(py_type, "string")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    func: Callable[..., Any]
    source: str  # "builtin" | "mcp:<server>" | "user"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        log.debug(f"registered tool {tool.name!r} (source={tool.source})")

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def tool_specs(self) -> list[dict[str, Any]]:
        # This is the JSON Schema list sent to Ollama so the LLM knows each
        # tool's name, purpose, and parameters.
        """Render every tool in Ollama's function-calling schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool by name. ALWAYS returns a string — the model reads it as a message."""
        t = self._tools.get(name)
        if t is None:
            return f"Tool {name!r} does not exist. Available: {self.names()}"
        try:
            result = t.func(**arguments)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            log.exception(f"tool {name!r} failed")
            return f"Tool {name!r} failed: {type(e).__name__}: {e}"


REGISTRY = ToolRegistry()


def tool(description: str, *, name: str | None = None) -> Callable[[Callable], Callable]:
    """Turn a Python function into a registered LLM tool.

    Inspects the signature, builds JSON Schema, registers into REGISTRY,
    returns the original function unchanged (so you can also call it directly).
    """

    def decorator(fn: Callable) -> Callable:
        # Read the function's signature and build a JSON Schema from it:
        # each parameter becomes a typed property; params without a default are "required".
        sig = inspect.signature(fn)
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []
        for pname, p in sig.parameters.items():
            properties[pname] = {"type": _json_type(p.annotation)}
            if p.default is inspect.Parameter.empty:
                required.append(pname)
        REGISTRY.register(
            Tool(
                name=name or fn.__name__,
                description=description.strip(),
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
                func=fn,
                source="builtin",
            )
        )
        return fn

    return decorator


# ── Three builtin tools ────────────────────────────────────────────────────

@tool(
    "Get the current local time as a 12-hour clock string like '2:59 PM'. "
    "Relay this time to the user as-is; do not convert it to a different format."
)
def get_time() -> str:
    # Return 12-hour time directly. Small models botch 24h→12h conversion
    # (e.g. turning '14:59' into 'half past four'), so we hand them no math to do.
    return datetime.now().strftime("%I:%M %p").lstrip("0")


WORKSPACE = Path.home() / "jarvis_workspace"
WORKSPACE.mkdir(parents=True, exist_ok=True)
MAX_READ_BYTES = 200_000


def _safe_path(name: str) -> Path:
    """Resolve a user-supplied path inside the workspace; reject escapes."""
    p = (WORKSPACE / name).resolve()
    base = WORKSPACE.resolve()
    if not (str(p) == str(base) or str(p).startswith(str(base) + os.sep)):
        raise ValueError(f"path {name!r} escapes the workspace")
    return p


@tool(
    "Read a file in the user's jarvis workspace (~/jarvis_workspace). "
    "Path is relative to the workspace; do not include `..` or absolute paths."
)
def read_file(path: str) -> str:
    try:
        p = _safe_path(path)
    except ValueError as e:
        return str(e)
    if not p.exists():
        return f"File {path!r} does not exist."
    if not p.is_file():
        return f"{path!r} is not a file."
    try:
        data = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"Could not read {path!r}: {e}"
    if len(data) > MAX_READ_BYTES:
        data = data[:MAX_READ_BYTES] + "\n...[truncated]"
    return data


@tool(
    "Write text to a file in the user's jarvis workspace (~/jarvis_workspace). "
    "Path is relative to the workspace; do not include `..` or absolute paths. "
    "Creates parent folders as needed and overwrites any existing file."
)
def write_file(path: str, content: str) -> str:
    try:
        p = _safe_path(path)
    except ValueError as e:
        return str(e)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)  # make intermediate folders
        p.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"Could not write {path!r}: {e}"
    # Report the real, absolute location so the model doesn't invent a path.
    return f"Wrote {len(content)} bytes to {p}."


@tool("Search the web (DuckDuckGo) and return the top results as JSON.")
def web_search(query: str) -> str:
    try:
        from ddgs import DDGS  # duckduckgo_search was renamed; old backend is dead
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return json.dumps({"error": "ddgs not installed — run: pip install ddgs"})
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        return json.dumps(results, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})
