"""LLM client protocol and built-in client implementations."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from my_claude.agent.events import EventHandler
from my_claude.agent.tools import ToolDefinition
from my_claude.core.config import AppConfig
from my_claude.core.context import AnthropicMessage, MessageContentBlock, TextBlock, ToolUseBlock

LLMMessage = AnthropicMessage
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"


@dataclass(frozen=True)
class LLMResponse:
    """Text response returned by an LLM client, optionally with raw provider data."""

    content: str
    content_blocks: tuple[MessageContentBlock, ...] = ()
    raw: Mapping[str, Any] | None = None

    def assistant_message(self) -> AnthropicMessage:
        if self.content_blocks:
            return AnthropicMessage.assistant_blocks(list(self.content_blocks))

        return AnthropicMessage.assistant_text(self.content)

    def tool_uses(self) -> list[ToolUseBlock]:
        return [block for block in self.content_blocks if isinstance(block, ToolUseBlock)]


class LLMClient(Protocol):
    """Protocol implemented by asynchronous LLM clients."""

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolDefinition],
    ) -> LLMResponse: ...


class LocalLLMClient:
    """Deterministic local placeholder client used before real provider configuration."""

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolDefinition],
    ) -> LLMResponse:
        user_messages = [_message_text(message) for message in messages if message.role == "user"]
        goal = user_messages[-1] if user_messages else ""
        tool_count = len(tools)
        return LLMResponse(
            content=(
                f"Local LLM placeholder accepted goal: {goal}"
                f"\nRegistered tools: {tool_count}"
            )
        )


@dataclass(frozen=True)
class OpenAICompatibleLLMClient:
    """Minimal client for providers exposing an OpenAI Responses-compatible endpoint."""

    api_key: str
    model: str
    base_url: str
    timeout_seconds: float

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolDefinition],
    ) -> LLMResponse:
        return await asyncio.to_thread(self._complete_blocking, messages, tools)

    def _complete_blocking(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolDefinition],
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": message.role,
                    "content": [
                        block.model_dump(mode="json", exclude_none=True)
                        for block in message.content
                    ],
                }
                for message in messages
            ],
        }
        if tools:
            payload["tools"] = [_tool_to_openai_payload(tool) for tool in tools]

        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as error:
            raise RuntimeError(f"LLM request failed: {error}") from error

        if not isinstance(response_data, dict):
            raise RuntimeError("LLM response must be a JSON object")

        return LLMResponse(
            content=_extract_response_text(response_data),
            raw=response_data,
        )


def create_llm_client(
    config: AppConfig,
    *,
    event_handler: EventHandler | None = None,
    run_id: str | None = None,
) -> LLMClient:
    if config.llm_provider == "local":
        return LocalLLMClient()

    if config.llm_provider == "anthropic":
        if config.llm_api_key is None:
            raise RuntimeError("MYCLAUDE_LLM_API_KEY is required for Anthropic LLM")

        os.environ["ANTHROPIC_API_KEY"] = config.llm_api_key
        os.environ["ANTHROPIC_BASE_URL"] = _anthropic_sdk_base_url(config.llm_base_url)

        from my_claude.core.llm.provider import AnthropicProviderConfig, AnthropicStreamingProvider

        return AnthropicStreamingProvider(
            AnthropicProviderConfig(
                api_key=config.llm_api_key,
                model=config.llm_model,
                base_url=_anthropic_messages_url(config.llm_base_url),
                timeout_seconds=config.llm_timeout_seconds,
                max_tokens=config.llm_max_tokens,
            ),
            event_handler=event_handler,
            run_id=run_id,
        )

    if config.llm_api_key is None:
        raise RuntimeError("MYCLAUDE_LLM_API_KEY is required for openai-compatible LLM")

    return OpenAICompatibleLLMClient(
        api_key=config.llm_api_key,
        model=config.llm_model,
        base_url=config.llm_base_url,
        timeout_seconds=config.llm_timeout_seconds,
    )


def _anthropic_messages_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped == OPENAI_RESPONSES_URL:
        return ANTHROPIC_MESSAGES_URL
    if stripped.endswith("/messages"):
        return stripped
    if stripped.endswith("/v1"):
        return f"{stripped}/messages"
    return f"{stripped}/v1/messages"


def _anthropic_sdk_base_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped == OPENAI_RESPONSES_URL:
        return "https://api.anthropic.com"
    if stripped.endswith("/v1/messages"):
        return stripped.removesuffix("/v1/messages")
    if stripped.endswith("/v1"):
        return stripped.removesuffix("/v1")
    return stripped


def _tool_to_openai_payload(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema or {"type": "object", "properties": {}},
    }


def _message_text(message: LLMMessage) -> str:
    return "".join(block.text for block in message.content if isinstance(block, TextBlock))


def _extract_response_text(response_data: Mapping[str, Any]) -> str:
    output_text = response_data.get("output_text")
    if isinstance(output_text, str):
        return output_text

    output = response_data.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        if chunks:
            return "".join(chunks)

    raise RuntimeError("LLM response did not contain text output")
