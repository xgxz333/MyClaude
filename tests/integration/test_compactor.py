from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

from my_claude.agent.tools import ToolDefinition
from my_claude.core.bus.events import ContextCompactedEvent
from my_claude.core.compaction.compactor import COMPACTION_ASSISTANT_ACK, Compactor
from my_claude.core.context import AnthropicMessage, WorkingMemory
from my_claude.llm.client import LLMResponse, LLMUsage


class RecordingBus:
    def __init__(self) -> None:
        self.events: list[ContextCompactedEvent] = []

    async def publish(self, event: ContextCompactedEvent) -> None:
        self.events.append(event)


class CompactLLM:
    def __init__(self, response: LLMResponse | Exception) -> None:
        self._response = response
        self.messages: Sequence[AnthropicMessage] | None = None
        self.tools: Sequence[ToolDefinition] | None = None
        self.step: int | None = None
        self.system_prompt_patch: str | None = None

    async def complete(
        self,
        messages: Sequence[AnthropicMessage],
        tools: Sequence[ToolDefinition],
        *,
        step: int | None = None,
        system_prompt_patch: str | None = None,
    ) -> LLMResponse:
        self.messages = messages
        self.tools = tools
        self.step = step
        self.system_prompt_patch = system_prompt_patch
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def test_compactor_replaces_memory_writes_summary_and_publishes_event(
    tmp_path: Path,
) -> None:
    summary = "\n".join(
        [
            "## 1. Original Goal",
            "Ship compact.",
            "## 2. Completed Steps",
            "Read files.",
            "## 3. Key Constraints & Discoveries",
            "Do not edit thread.jsonl.",
            "## 4. Current File State",
            "Changed compactor.",
            "## 5. Remaining TODOs",
            "Run tests.",
            "## 6. Critical Data",
            "tests/integration/test_compactor.py",
        ]
    )
    bus = RecordingBus()
    llm = CompactLLM(LLMResponse(content=summary, usage=LLMUsage(output_tokens=42)))
    memory = WorkingMemory.from_goal("ship compact", run_id="run-1", max_steps=3)
    memory.append_assistant_text("I inspected the code.")
    compactor = Compactor(bus, tmp_path, "sess-1", llm_client=llm)

    result = asyncio.run(compactor.compact(memory, CompactLLM(LLMResponse(content="unused"))))

    assert result == summary
    assert llm.tools == []
    assert llm.step == 0
    assert llm.system_prompt_patch == "You are a helpful assistant that summarizes conversations."
    assert llm.messages is not None
    assert "## 1. Original Goal" in llm.messages[0].text
    assert "ship compact" in llm.messages[0].text
    assert memory.messages == [
        AnthropicMessage.user_text(summary),
        AnthropicMessage.assistant_text(COMPACTION_ASSISTANT_ACK),
    ]
    summary_files = list(tmp_path.glob("summary_*.md"))
    assert len(summary_files) == 1
    assert summary_files[0].read_text(encoding="utf-8") == summary + "\n"
    assert len(bus.events) == 1
    event = bus.events[0]
    assert event.session_id == "sess-1"
    assert event.run_id == "run-1"
    assert event.original_tokens > 0
    assert event.summary_tokens == 42


def test_compactor_failure_leaves_memory_and_files_unchanged(tmp_path: Path) -> None:
    bus = RecordingBus()
    memory = WorkingMemory.from_goal("ship compact", run_id="run-1", max_steps=3)
    original_messages = memory.llm_messages()
    compactor = Compactor(bus, tmp_path, llm_client=CompactLLM(RuntimeError("boom")))

    result = asyncio.run(compactor.compact(memory, CompactLLM(LLMResponse(content="unused"))))

    assert result is None
    assert memory.messages == original_messages
    assert list(tmp_path.glob("summary_*.md")) == []
    assert bus.events == []


def test_compactor_empty_summary_leaves_memory_and_files_unchanged(tmp_path: Path) -> None:
    bus = RecordingBus()
    memory = WorkingMemory.from_goal("ship compact", run_id="run-1", max_steps=3)
    original_messages = memory.llm_messages()
    compactor = Compactor(bus, tmp_path, llm_client=CompactLLM(LLMResponse(content=" \n ")))

    result = asyncio.run(compactor.compact(memory, CompactLLM(LLMResponse(content="unused"))))

    assert result is None
    assert memory.messages == original_messages
    assert list(tmp_path.glob("summary_*.md")) == []
    assert bus.events == []
