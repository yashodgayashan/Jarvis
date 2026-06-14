# LLM = Large Language Model. tool / function calling = the LLM emitting a structured
# request to run a named function instead of just replying in prose.
"""Agent.turn — streams an LLM reply and runs tool-dispatch hops as needed.

The shape of a turn:
    user says X
    → stream reply; collect any tool_calls; emit on_token/on_sentence as it goes
    → if no tool calls: record reply, return
    → if tool calls: dispatch each, append results as `role=tool` messages, loop
       (the next stream lets the model react to the tool results)
    → bail after MAX_TOOL_ITERATIONS to avoid runaway loops
"""
from __future__ import annotations

import json
import logging
from typing import Callable

import httpx

from tools import REGISTRY
from tts import SentenceSplitter

log = logging.getLogger(__name__)

# Cap on the tool-dispatch hop loop: how many times we may re-call the LLM after
# feeding it tool results before giving up (guards against runaway tool loops).
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
        # Low temperature keeps the model on the structured tool-call format.
        # At Ollama's default (0.8) a 7B model wanders off it — emitting the
        # call as plain text ("get_time()") or just guessing the answer.
        self.temperature = temperature
        self.messages: list[dict] = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def turn(
        self,
        user_text: str,
        *,
        on_token: Callable[[str], None] | None = None,
        on_sentence: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, dict, str], None] | None = None,
    ) -> str:
        """Run one user→assistant cycle, dispatching tools as needed. Returns final text."""
        self.messages.append({"role": "user", "content": user_text})

        full_reply = ""
        # Each iteration is one "hop": call the LLM, and if it asked for tools,
        # run them and loop so the model can react to the results.
        for hop in range(MAX_TOOL_ITERATIONS):
            reply_text, tool_calls = self._stream_one(on_token=on_token, on_sentence=on_sentence)

            if not tool_calls:
                self.messages.append({"role": "assistant", "content": reply_text})
                full_reply += reply_text
                return full_reply or reply_text

            # Record the model's tool-call decision, then run each tool
            self.messages.append(
                {"role": "assistant", "content": reply_text or "", "tool_calls": tool_calls}
            )
            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments") or {}
                if isinstance(args, str):  # some models send a JSON string
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                log.info(f"→ tool {name}({args})")
                # Actually run the requested function and capture its string result.
                result = REGISTRY.dispatch(name, args)
                preview = (result[:200] + "…") if len(result) > 200 else result
                log.info(f"← {preview}")
                if on_tool_call:
                    on_tool_call(name, args, result)
                # Feed the result back as a `role=tool` message so the next hop's
                # LLM call can read it and continue the answer.
                self.messages.append({"role": "tool", "name": name, "content": result})
            full_reply += reply_text

        log.warning(f"hit MAX_TOOL_ITERATIONS={MAX_TOOL_ITERATIONS}; bailing")
        return full_reply or "(no response — too many tool hops)"

    def _stream_one(self, *, on_token, on_sentence):
        """One Ollama stream. Returns (text, list_of_raw_tool_calls)."""
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
            # The tool specs (each tool's JSON Schema) tell the LLM what it may call.
            "tools": REGISTRY.tool_specs() or None,
            "options": {"temperature": self.temperature},
        }
        # Ollama dislikes "tools": None; drop it
        if not payload["tools"]:
            payload.pop("tools")

        text = ""
        tool_calls: list[dict] = []
        splitter = SentenceSplitter() if on_sentence else None

        # httpx streaming: read Ollama's reply incrementally instead of waiting
        # for the whole thing — each line is one JSON chunk of the response.
        with httpx.stream("POST", self.ollama_url, json=payload, timeout=120.0) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message") or {}
                # delta = the new token(s) (small text fragments) in this chunk.
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
                    if splitter:
                        tail = splitter.flush()
                        if tail:
                            on_sentence(tail)
                    break
        return text, tool_calls


# [
#     {"role": "user", "content": "What time is it?"},
#     {
#         "role": "assistant",
#         "content": "",
#         "tool_calls": [
#             {"function": {"name": "get_time", "arguments": {}}}
#         ]
#     },
#     {
#         "role": "tool",
#         "name": "get_time",
#         "content": "10:30 AM"
#     }
# ]