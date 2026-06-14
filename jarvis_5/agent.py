"""Async Agent. The chat loop now uses httpx.AsyncClient and awaits dispatch."""
from __future__ import annotations

import json
import logging
from typing import Callable

import httpx

from tools import REGISTRY
from tts import SentenceSplitter

log = logging.getLogger(__name__)

# Cap on how many times the LLM may call tools and come back in one turn,
# so a model that keeps requesting tools can't loop forever.
MAX_TOOL_ITERATIONS = 5


class Agent:
    def __init__(
        self,
        model: str,
        ollama_url: str = "http://localhost:11434/api/chat",
        system: str = "",
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        # Low temperature keeps the 7B model on the structured tool-call format and
        # makes it use each tool's EXACT parameter names (e.g. search_repositories
        # needs `query`, not `q`). Ollama's default (0.8) lets it improvise and miss.
        self.temperature = temperature
        self.messages: list[dict] = []
        if system:
            self.messages.append({"role": "system", "content": system})
        # Async HTTP client used to stream chat completions from Ollama (local LLM).
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def turn(
        self,
        user_text: str,
        *,
        on_token: Callable[[str], None] | None = None,
        on_sentence: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, dict, str], None] | None = None,
    ) -> str:
        self.messages.append({"role": "user", "content": user_text})

        full_reply = ""
        # Each "hop": stream one LLM response; if it asks for tools, run them and
        # loop so the model can see the results and continue.
        for hop in range(MAX_TOOL_ITERATIONS):
            reply_text, tool_calls = await self._stream_one(
                on_token=on_token, on_sentence=on_sentence
            )

            # No tool requests → this is the final answer; record it and return.
            if not tool_calls:
                self.messages.append({"role": "assistant", "content": reply_text})
                full_reply += reply_text
                return full_reply or reply_text

            self.messages.append(
                {"role": "assistant", "content": reply_text or "", "tool_calls": tool_calls}
            )
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                # Arguments may arrive as a dict or as a JSON string; normalize to a dict.
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                log.info(f"→ tool {name}({args})")
                # dispatch is async now: builtin tools run sync, MCP tools are awaited.
                result = await REGISTRY.dispatch(name, args)
                preview = (result[:200] + "…") if len(result) > 200 else result
                log.info(f"← {preview}")
                if on_tool_call:
                    on_tool_call(name, args, result)
                self.messages.append({"role": "tool", "name": name, "content": result})
            full_reply += reply_text

        log.warning(f"hit MAX_TOOL_ITERATIONS={MAX_TOOL_ITERATIONS}; bailing")
        return full_reply or "(no response — too many tool hops)"

    async def _stream_one(self, *, on_token, on_sentence):
        # Build the Ollama chat request: history + available tools, streamed back.
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            "tools": REGISTRY.tool_specs() or None,
            "options": {"temperature": self.temperature},
        }
        if not payload["tools"]:
            payload.pop("tools")

        text = ""
        tool_calls: list[dict] = []
        # Groups streamed tokens into whole sentences so TTS can speak them as they form.
        splitter = SentenceSplitter() if on_sentence else None

        # Stream the response line by line; Ollama sends one JSON object per chunk.
        async with self._client.stream("POST", self.ollama_url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message") or {}
                # delta = the new token(s) in this chunk.
                delta = msg.get("content") or ""
                if delta:
                    text += delta
                    if on_token:
                        on_token(delta)
                    if splitter:
                        for sent in splitter.feed(delta):
                            on_sentence(sent)
                for raw in msg.get("tool_calls") or []:
                    tool_calls.append(raw)
                if chunk.get("done"):
                    # Stream finished: flush any leftover partial sentence to TTS.
                    if splitter:
                        tail = splitter.flush()
                        if tail:
                            on_sentence(tail)
                    break
        return text, tool_calls
