"""Tool system: dataclass + @tool decorator + registry + 3 builtins.

In jarvis_5 the registry's `dispatch` becomes async — MCP tools are coroutines,
and the agent loop awaits them. Builtins (sync functions) still work: we just
check whether the call result is awaitable.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

# Maps Python types to JSON Schema type names (used to describe tool parameters).
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
    parameters: dict[str, Any]
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
        # Build the tool list in the shape the LLM expects (sent to Ollama each turn).
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

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool by name. Awaits the result if it's a coroutine (MCP tools).
        ALWAYS returns a string — failures become text the model can read.
        """
        t = self._tools.get(name)
        if t is None:
            return f"Tool {name!r} does not exist. Available: {self.names()}"
        try:
            result = t.func(**arguments)
            # MCP tools are coroutines, so await them; builtins return plain values.
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, str):
                return result
            # Non-string results are serialized to JSON so the model gets text back.
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            log.exception(f"tool {name!r} failed")
            return f"Tool {name!r} failed: {type(e).__name__}: {e}"


REGISTRY = ToolRegistry()


def tool(description: str, *, name: str | None = None) -> Callable[[Callable], Callable]:
    """Decorator: introspect signature, build JSON Schema, register into REGISTRY."""

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []
        for pname, p in sig.parameters.items():
            properties[pname] = {"type": _json_type(p.annotation)}
            # A parameter with no default is required in the JSON Schema.
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
    # Sandbox check: resolve the path and reject anything (e.g. via "..") that
    # would land outside the workspace directory.
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


@tool(
    "Open a file, folder, or app on the user's computer with its default application "
    "— like double-clicking it (e.g. open a PDF, image, document, or a folder in the "
    "file browser). Give an absolute path or one starting with '~'. This launches the "
    "item in a GUI app; it does NOT return the file's contents — use read_file for that."
)
def open_path(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"Nothing to open — {path!r} does not exist."
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(p)], check=True, timeout=10)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(p)], check=True, timeout=10)
        elif sys.platform.startswith("win"):
            os.startfile(str(p))  # type: ignore[attr-defined]  # Windows-only
        else:
            return f"Don't know how to open files on platform {sys.platform!r}."
    except FileNotFoundError:
        return "No file-opener found (need `open` on macOS or `xdg-open` on Linux)."
    except subprocess.CalledProcessError as e:
        return f"The opener failed for {path!r} (exit {e.returncode})."
    except subprocess.TimeoutExpired:
        # `open`/`xdg-open` normally return at once; a timeout just means the app is
        # still launching — the file is on its way, so report best-effort success.
        return f"Opened {p} (the app is still starting)."
    except Exception as e:
        return f"Could not open {path!r}: {type(e).__name__}: {e}"
    return f"Opened {p}."
