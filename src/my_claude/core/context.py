"""Run context and working memory models for the agent loop."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(StrEnum):
    """High-level lifecycle status tracked by working memory."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TextBlock(BaseModel):
    """Anthropic-compatible text content block."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    """Anthropic-compatible assistant tool request block."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    """Anthropic-compatible user tool result block."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


MessageContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


class AnthropicMessage(BaseModel):
    """Single Anthropic-compatible conversation message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    content: list[MessageContentBlock]

    @classmethod
    def user_text(cls, text: str) -> AnthropicMessage:
        return cls(role="user", content=[TextBlock(text=text)])

    @classmethod
    def assistant_text(cls, text: str) -> AnthropicMessage:
        return cls(role="assistant", content=[TextBlock(text=text)])

    @classmethod
    def assistant_blocks(cls, blocks: list[MessageContentBlock]) -> AnthropicMessage:
        return cls(role="assistant", content=blocks)

    @property
    def text(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))


class WorkingMemory(BaseModel):
    """Mutable run state and Anthropic-compatible message history for one agent run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    goal: str
    max_steps: int = Field(gt=0)
    step: int = Field(default=0, ge=0)
    status: RunStatus = RunStatus.PENDING
    reason: str | None = None
    status_transition_reason: str | None = None
    messages: list[AnthropicMessage] = Field(default_factory=list)

    @classmethod
    def from_goal(cls, goal: str, *, run_id: str = "local", max_steps: int = 1) -> WorkingMemory:
        return cls(
            run_id=run_id,
            goal=goal,
            max_steps=max_steps,
            messages=[AnthropicMessage.user_text(goal)],
        )

    def mark_step_started(self, *, reason: str) -> int:
        if self.step >= self.max_steps:
            raise RuntimeError("working memory max_steps exceeded")

        self.step += 1
        self.set_status(
            RunStatus.RUNNING,
            reason=f"step {self.step} started",
            transition_reason=reason,
        )
        return self.step

    def set_status(
        self,
        status: RunStatus,
        *,
        reason: str,
        transition_reason: str,
    ) -> None:
        self.status = status
        self.reason = reason
        self.status_transition_reason = transition_reason

    def llm_messages(self) -> list[AnthropicMessage]:
        return list(self.messages)

    def append_assistant_text(self, text: str) -> None:
        self.messages.append(AnthropicMessage.assistant_text(text))

    def append_assistant_message(self, message: AnthropicMessage) -> None:
        self.messages.append(message)

    def append_tool_result(
        self,
        *,
        tool_use_id: str,
        content: str,
        is_error: bool = False,
    ) -> None:
        self.messages.append(
            AnthropicMessage(
                role="user",
                content=[
                    ToolResultBlock(
                        tool_use_id=tool_use_id,
                        content=content,
                        is_error=is_error,
                    )
                ],
            )
        )
