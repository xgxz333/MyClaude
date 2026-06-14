from __future__ import annotations

import asyncio
import io
import os
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest

from my_claude.agent.tools import ToolDefinition
from my_claude.core.context import AnthropicMessage
from my_claude.core.llm.provider import (
    AnthropicProviderConfig,
    AnthropicStreamingProvider,
    _anthropic_sdk_base_url,
    _iter_sse_events,
    _messages_payload,
    _system_payload,
    _tools_payload,
    normalize_anthropic_stream,
)


def test_provider_adds_cache_control_to_system_and_tool_anchors() -> None:
    system = _system_payload()
    patched_system = _system_payload("Long-term session notes:\n- fact: repo uses uv")
    messages = _messages_payload([AnthropicMessage.user_text("ship it")])
    tools = _tools_payload(
        [
            ToolDefinition(
                name="echo",
                description="Echo a value.",
                input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
            )
        ]
    )

    assert system[-1]["cache_control"] == {"type": "ephemeral"}
    assert patched_system[0]["text"].startswith("You are a helpful AI assistant")
    assert patched_system[1]["text"] == "Long-term session notes:\n- fact: repo uses uv"
    assert patched_system[-1]["cache_control"] == {"type": "ephemeral"}
    assert messages[-1]["content"] == "ship it"
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}


def test_provider_normalizes_base_url_for_anthropic_sdk() -> None:
    assert _anthropic_sdk_base_url("https://ai.prism.uno") == "https://ai.prism.uno"
    assert _anthropic_sdk_base_url("https://ai.prism.uno/v1") == "https://ai.prism.uno"
    assert _anthropic_sdk_base_url("https://ai.prism.uno/v1/messages") == (
        "https://ai.prism.uno"
    )


def test_provider_syncs_config_to_anthropic_sdk_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)

    AnthropicStreamingProvider(
        AnthropicProviderConfig(
            api_key="test-key",
            model="claude-test",
            base_url="https://ai.prism.uno/v1/messages",
        ),
        client=SimpleNamespace(
            messages=FakeMessages(
                FakeStream(texts=[], final_message=SimpleNamespace()),
            ),
        ),
    )

    assert os.environ["ANTHROPIC_API_KEY"] == "test-key"
    assert os.environ["ANTHROPIC_BASE_URL"] == "https://ai.prism.uno"


def test_provider_uses_anthropic_sdk_stream_and_parses_final_message() -> None:
    final_message = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Use the tool."),
            SimpleNamespace(type="tool_use", id="tool-1", name="echo", input={"value": "ok"}),
        ]
    )
    fake_messages = FakeMessages(
        FakeStream(texts=["Use ", "the tool."], final_message=final_message)
    )
    fake_client = SimpleNamespace(messages=fake_messages)
    provider = AnthropicStreamingProvider(
        AnthropicProviderConfig(
            api_key="test-key",
            model="claude-test",
            base_url="https://ai.prism.uno/v1/messages",
            max_tokens=256,
        ),
        client=fake_client,
    )

    response = asyncio.run(
        provider.complete(
            [AnthropicMessage.user_text("ship it")],
            [
                ToolDefinition(
                    name="echo",
                    description="Echo a value.",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
            system_prompt_patch="Long-term session notes:\n- fact: repo uses uv",
        )
    )

    kwargs = cast(dict[str, Any], fake_messages.kwargs)
    assert kwargs["model"] == "claude-test"
    assert kwargs["max_tokens"] == 256
    assert kwargs["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert kwargs["system"][-1]["text"] == "Long-term session notes:\n- fact: repo uses uv"
    assert kwargs["messages"][0]["role"] == "user"
    assert kwargs["messages"][0]["content"] == "ship it"
    assert kwargs["tools"][0]["name"] == "echo"
    assert response.content == "Use the tool."
    assert len(response.content_blocks) == 2
    assert response.tool_uses()[0].input == {"value": "ok"}


def test_provider_normalizes_streamed_text_and_tool_use_blocks() -> None:
    response = normalize_anthropic_stream(
        [
            {"type": "message_start", "message": {"id": "msg_1"}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Use a tool."},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "tool-1", "name": "echo", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"value":"ok"}'},
            },
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        ]
    )

    tool_uses = response.tool_uses()

    assert response.content == "Use a tool."
    assert len(response.content_blocks) == 2
    assert len(tool_uses) == 1
    assert tool_uses[0].id == "tool-1"
    assert tool_uses[0].name == "echo"
    assert tool_uses[0].input == {"value": "ok"}
    assert response.raw is not None
    raw = cast(dict[str, Any], response.raw)
    assert raw["message"]["stop_reason"] == "tool_use"


def test_provider_reads_sse_json_events() -> None:
    stream = io.BytesIO(
        b'event: content_block_delta\n'
        b'data: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"text_delta","text":"hi"}}\n'
        b"\n"
    )

    events = list(_iter_sse_events(stream))

    assert events == [
        {
            "event": "content_block_delta",
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        }
    ]


class FakeStream:
    def __init__(self, *, texts: list[str], final_message: SimpleNamespace) -> None:
        self._texts = texts
        self._final_message = final_message

    async def __aenter__(self) -> FakeStream:
        return self

    async def __aexit__(self, *_args: object) -> None:
        pass

    @property
    def text_stream(self) -> AsyncIterator[str]:
        async def stream_text() -> AsyncIterator[str]:
            for text in self._texts:
                yield text

        return stream_text()

    async def get_final_message(self) -> SimpleNamespace:
        return self._final_message


class FakeMessages:
    def __init__(self, stream: FakeStream) -> None:
        self._stream = stream
        self.kwargs: dict[str, object] = {}

    def stream(self, **kwargs: object) -> FakeStream:
        self.kwargs = kwargs
        return self._stream
