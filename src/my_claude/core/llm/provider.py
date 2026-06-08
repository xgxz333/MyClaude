"""Streaming Anthropic-compatible LLM provider with prompt cache anchors."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import anthropic

from my_claude.agent.tools import ToolDefinition
from my_claude.core.context import (
    AnthropicMessage,
    MessageContentBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from my_claude.llm.client import LLMResponse

ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_PROMPT_CACHING_BETA = "prompt-caching-2024-07-31"
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Use the available tools to complete the user's goal. "
    "When the goal is fully achieved, respond with a final answer and do not call any more tools."
)


@dataclass(frozen=True)
class AnthropicProviderConfig:
    """Connection settings for Anthropic's Messages API."""

    api_key: str
    model: str
    base_url: str = "https://api.anthropic.com/v1/messages"
    timeout_seconds: float = 30.0
    max_tokens: int = 1024


class AnthropicStreamingProvider:
    """Call Anthropic Messages with streaming and normalize the response blocks."""

    def __init__(self, config: AnthropicProviderConfig, client: Any | None = None) -> None:
        self._config = config
        os.environ["ANTHROPIC_API_KEY"] = config.api_key
        os.environ["ANTHROPIC_BASE_URL"] = _anthropic_sdk_base_url(config.base_url)
        self._client = client or anthropic.AsyncAnthropic(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=0,
        )

    async def complete(
        self,
        messages: Sequence[AnthropicMessage],
        tools: Sequence[ToolDefinition],
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "system": _system_payload(),
            "messages": _messages_payload(messages),
        }
        if tools:
            kwargs["tools"] = _tools_payload(tools)

        text_parts: list[str] = []

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    text_parts.append(text)
                final_message = await stream.get_final_message()
        except anthropic.APIError as error:
            raise RuntimeError(f"LLM request failed: {error}") from error

        return _response_from_final_message(
            final_message,
            fallback_text="".join(text_parts),
        )


def normalize_anthropic_stream(events: Sequence[Mapping[str, Any]]) -> LLMResponse:
    """Convert Anthropic stream events into the standard LLMResponse."""

    builders: dict[int, _ContentBlockBuilder] = {}
    raw_message: dict[str, Any] = {}

    for event in events:
        event_type = event.get("type")
        if event_type == "message_start":
            message = event.get("message")
            if isinstance(message, dict):
                raw_message = dict(message)
        elif event_type == "content_block_start":
            index = _event_index(event)
            content_block = event.get("content_block")
            if isinstance(content_block, dict):
                builders[index] = _ContentBlockBuilder.from_start(content_block)
        elif event_type == "content_block_delta":
            index = _event_index(event)
            delta = event.get("delta")
            if isinstance(delta, dict) and index in builders:
                builders[index].apply_delta(delta)
        elif event_type == "content_block_stop":
            index = _event_index(event)
            if index in builders:
                builders[index].mark_stopped()
        elif event_type == "message_delta":
            delta = event.get("delta")
            usage = event.get("usage")
            if isinstance(delta, dict):
                raw_message.update(delta)
            if isinstance(usage, dict):
                raw_message["usage"] = usage

    content_blocks = tuple(
        block
        for index in sorted(builders)
        if (block := builders[index].build()) is not None
    )
    content = "".join(block.text for block in content_blocks if isinstance(block, TextBlock))

    return LLMResponse(
        content=content,
        content_blocks=content_blocks,
        raw={"message": raw_message, "events": list(events)},
    )


@dataclass
class _ContentBlockBuilder:
    block_type: str
    text: str = ""
    tool_use_id: str = ""
    tool_name: str = ""
    input_json: str = ""
    stopped: bool = False

    @classmethod
    def from_start(cls, content_block: Mapping[str, Any]) -> _ContentBlockBuilder:
        block_type = content_block.get("type")
        if block_type == "text":
            return cls(block_type="text", text=str(content_block.get("text", "")))
        if block_type == "tool_use":
            raw_input = content_block.get("input", {})
            return cls(
                block_type="tool_use",
                tool_use_id=str(content_block.get("id", "")),
                tool_name=str(content_block.get("name", "")),
                input_json=json.dumps(raw_input) if raw_input else "",
            )

        return cls(block_type=str(block_type or "unknown"))

    def apply_delta(self, delta: Mapping[str, Any]) -> None:
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            self.text += str(delta.get("text", ""))
        elif delta_type == "input_json_delta":
            self.input_json += str(delta.get("partial_json", ""))

    def mark_stopped(self) -> None:
        self.stopped = True

    def build(self) -> MessageContentBlock | None:
        if self.block_type == "text":
            return TextBlock(text=self.text)
        if self.block_type == "tool_use":
            return ToolUseBlock(
                id=self.tool_use_id,
                name=self.tool_name,
                input=_loads_tool_input(self.input_json),
            )

        return None


def _messages_payload(messages: Sequence[AnthropicMessage]) -> list[dict[str, Any]]:
    return [
        {
            "role": message.role,
            "content": _message_content_payload(message),
        }
        for message in messages
    ]


def _system_payload() -> list[dict[str, Any]]:
    return [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _message_content_payload(message: AnthropicMessage) -> str | list[dict[str, Any]]:
    if len(message.content) == 1 and isinstance(message.content[0], TextBlock):
        return message.content[0].text

    return [_block_payload(block) for block in message.content]


def _anthropic_sdk_base_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/v1/messages"):
        return stripped.removesuffix("/v1/messages")
    if stripped.endswith("/v1"):
        return stripped.removesuffix("/v1")
    return stripped


def _response_from_final_message(final_message: Any, *, fallback_text: str) -> LLMResponse:
    content_blocks = tuple(
        block
        for raw_block in getattr(final_message, "content", ())
        if (block := _content_block_from_anthropic_block(raw_block)) is not None
    )
    content = "".join(block.text for block in content_blocks if isinstance(block, TextBlock))
    if not content:
        content = fallback_text

    return LLMResponse(
        content=content,
        content_blocks=content_blocks,
        raw={"message": _raw_final_message(final_message)},
    )


def _content_block_from_anthropic_block(block: Any) -> MessageContentBlock | None:
    block_type = getattr(block, "type", None)
    if block_type == "text":
        return TextBlock(text=str(getattr(block, "text", "")))
    if block_type == "tool_use":
        raw_input = getattr(block, "input", {})
        tool_input = dict(raw_input) if isinstance(raw_input, Mapping) else {}
        return ToolUseBlock(
            id=str(getattr(block, "id", "")),
            name=str(getattr(block, "name", "")),
            input=tool_input,
        )

    return None


def _raw_final_message(final_message: Any) -> Mapping[str, Any]:
    model_dump = getattr(final_message, "model_dump", None)
    if not callable(model_dump):
        return {}

    raw = model_dump(mode="json")
    if isinstance(raw, Mapping):
        return dict(raw)

    return {}


def _block_payload(block: MessageContentBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": block.content,
            "is_error": block.is_error,
        }

    raise TypeError(f"unsupported message block: {type(block).__name__}")


def _tools_payload(tools: Sequence[ToolDefinition]) -> list[dict[str, Any]]:
    payload = [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema or {"type": "object", "properties": {}},
        }
        for tool in tools
    ]
    if payload:
        payload[-1]["cache_control"] = {"type": "ephemeral"}
    return payload


def _iter_sse_events(response: Any) -> Iterator[dict[str, Any]]:
    event_type: str | None = None
    data_lines: list[str] = []

    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line:
            if data_lines:
                data = "\n".join(data_lines)
                if data != "[DONE]":
                    event = json.loads(data)
                    if event_type is not None:
                        event.setdefault("event", event_type)
                    yield event
            event_type = None
            data_lines = []
            continue

        if line.startswith("event:"):
            event_type = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())

    if data_lines:
        data = "\n".join(data_lines)
        if data != "[DONE]":
            event = json.loads(data)
            if event_type is not None:
                event.setdefault("event", event_type)
            yield event


def _event_index(event: Mapping[str, Any]) -> int:
    index = event.get("index")
    if isinstance(index, int):
        return index
    if isinstance(index, str) and index.isdigit():
        return int(index)
    return 0


def _loads_tool_input(raw: str) -> dict[str, Any]:
    if not raw:
        return {}

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid streamed tool input JSON: {error}") from error

    if not isinstance(value, dict):
        raise RuntimeError("streamed tool input must be a JSON object")

    return value
