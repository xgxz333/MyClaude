"""Compact long in-memory run history into a structured handoff summary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from my_claude.agent.tools import ToolDefinition
from my_claude.core.bus.events import ContextCompactedEvent
from my_claude.core.context import (
    AnthropicMessage,
    MessageContentBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    WorkingMemory,
)
from my_claude.llm.client import LLMClient, LLMResponse

COMPACTION_SYSTEM_PROMPT = "You are a helpful assistant that summarizes conversations."
COMPACTION_ASSISTANT_ACK = "Understood, I'll continue from this summary."

class ChatProvider(Protocol):
    async def chat(
        self,
        messages: Sequence[AnthropicMessage],
        tool_schemas: Sequence[ToolDefinition],
        *,
        run_id: str,
        step: int,
        system_prompt: str,
    ) -> LLMResponse: ...


LLMClientFactory = Callable[[], LLMClient | ChatProvider]


@dataclass(frozen=True)
class CompactResult:
    """Successful compaction result."""

    summary_text: str
    original_token_estimate: int
    summary_tokens: int


class Compactor:
    """Summarize current run memory without modifying persisted session history."""

    def __init__(
        self,
        bus: Any,
        session_dir: Path,
        session_id: str = "",
        *,
        llm_client: LLMClient | ChatProvider | None = None,
        llm_client_factory: LLMClientFactory | None = None,
    ) -> None:
        self._bus = bus
        self._session_dir = session_dir
        self._session_id = session_id
        self._llm_client = llm_client
        self._llm_client_factory = llm_client_factory

    async def compact(
        self,
        context: WorkingMemory,
        provider: LLMClient | ChatProvider,
    ) -> str | None:
        original_messages = context.llm_messages()
        result = await self.compact_messages(original_messages, provider)
        if result is None:
            return None

        summary_text = result.summary_text
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
        summary_path = self._session_dir / f"summary_{timestamp}.md"
        try:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(summary_text + "\n", encoding="utf-8")
        except OSError:
            return None

        context.messages = [
            AnthropicMessage.user_text(summary_text),
            AnthropicMessage.assistant_text(COMPACTION_ASSISTANT_ACK),
        ]
        await self._publish_compacted(
            run_id=context.run_id,
            original_tokens=result.original_token_estimate,
            summary_tokens=result.summary_tokens,
            ts=timestamp,
        )
        return summary_text

    async def compact_messages(
        self,
        messages: Sequence[AnthropicMessage],
        provider: LLMClient | ChatProvider,
        *,
        focus: str = "",
    ) -> CompactResult | None:
        compact_provider = self._compact_provider(provider)
        try:
            response = await self._summarize(
                compact_provider,
                list(messages),
                focus=focus,
            )
            summary_text = response.content.strip()
        except Exception:
            return None

        if not summary_text:
            return None
        original_token_estimate = _estimate_tokens(_messages_markdown(list(messages)))
        summary_tokens = (
            response.usage.output_tokens
            if response.usage is not None
            else _estimate_tokens(summary_text)
        )
        return CompactResult(
            summary_text=summary_text,
            original_token_estimate=original_token_estimate,
            summary_tokens=summary_tokens,
        )

    async def _summarize(
        self,
        provider: LLMClient | ChatProvider,
        messages: list[AnthropicMessage],
        *,
        focus: str = "",
    ) -> LLMResponse:
        prompt = _compaction_prompt(messages, focus=focus)
        request_messages = [AnthropicMessage.user_text(prompt)]
        chat = getattr(provider, "chat", None)
        if callable(chat):
            return await chat(
                request_messages,
                [],
                run_id="compact",
                step=0,
                system_prompt=COMPACTION_SYSTEM_PROMPT,
            )
        return await provider.complete(
            request_messages,
            [],
            step=0,
            system_prompt_patch=COMPACTION_SYSTEM_PROMPT,
        )

    async def _publish_compacted(
        self,
        *,
        run_id: str,
        original_tokens: int,
        summary_tokens: int,
        ts: str,
    ) -> None:
        event = ContextCompactedEvent(
            session_id=self._session_id,
            run_id=run_id,
            original_tokens=original_tokens,
            summary_tokens=summary_tokens,
            ts=ts,
        )
        publish = getattr(self._bus, "publish", None)
        if callable(publish):
            try:
                await publish(event)
            except Exception:
                return

    def _compact_provider(
        self,
        fallback: LLMClient | ChatProvider,
    ) -> LLMClient | ChatProvider:
        if self._llm_client is not None:
            return self._llm_client
        if self._llm_client_factory is not None:
            self._llm_client = self._llm_client_factory()
            return self._llm_client
        return fallback


def _compaction_prompt(messages: list[AnthropicMessage], *, focus: str = "") -> str:
    focus_text = focus.strip()
    focus_part = (
        f"Additional focus requested by the user:\n{focus_text}"
        if focus_text
        else "No additional focus was requested."
    )
    return "\n\n".join(
        [
            "Summarize the conversation history below into a structured handoff summary.",
            focus_part,
            "The output must be Markdown and must include exactly these section headings:",
            "## 1. Original Goal\n"
            "## 2. Completed Steps\n"
            "## 3. Key Constraints & Discoveries\n"
            "## 4. Current File State\n"
            "## 5. Remaining TODOs\n"
            "## 6. Critical Data",
            "Requirements:\n"
            "- Preserve the user's original goal.\n"
            "- Preserve completed steps.\n"
            "- Preserve key constraints and discoveries.\n"
            "- Preserve current file modification state.\n"
            "- Preserve remaining TODOs.\n"
            "- Preserve critical data such as file paths, commands, error messages, "
            "configuration values, IDs, and test results.\n"
            "- Do not write a generic summary.\n"
            "- Do not omit information needed to continue the task.",
            "Conversation history:\n\n" + _messages_markdown(messages),
        ]
    )


def _messages_markdown(messages: list[AnthropicMessage]) -> str:
    chunks: list[str] = []
    for index, message in enumerate(messages, 1):
        chunks.append(f"### Message {index}: {message.role}\n{_blocks_markdown(message.content)}")
    return "\n\n".join(chunks)


def _blocks_markdown(blocks: list[MessageContentBlock]) -> str:
    chunks: list[str] = []
    for block in blocks:
        if isinstance(block, TextBlock):
            chunks.append(block.text)
        elif isinstance(block, ToolUseBlock):
            chunks.append(
                "```tool_use\n"
                f"id={block.id}\nname={block.name}\ninput={block.input}\n"
                "```"
            )
        elif isinstance(block, ToolResultBlock):
            chunks.append(
                "```tool_result\n"
                f"tool_use_id={block.tool_use_id}\nis_error={block.is_error}\n"
                f"{block.content}\n"
                "```"
            )
    return "\n\n".join(chunks)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0
