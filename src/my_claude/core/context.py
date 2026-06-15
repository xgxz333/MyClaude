"""Run context and working memory models for the agent loop."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

BASE_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Use the available tools to complete the user's goal. "
    "When the goal is fully achieved, respond with a final answer and do not call any more tools."
)


class RunStatus(StrEnum):
    """High-level lifecycle status tracked by working memory."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ExecutionMode(StrEnum):
    """Whether a run is isolated or restored from a persistent session."""

    ISOLATED = "isolated"
    SESSION = "session"


class SemanticMemoryKind(StrEnum):
    """Kinds of durable semantic memory attached to a session."""

    FACT = "fact"
    DECISION = "decision"
    NOTE = "note"


class SemanticMemoryItem(BaseModel):
    """One fact, decision, or note injected into a session-backed run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: SemanticMemoryKind
    content: str


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


class ExecutionContext(BaseModel):
    """LLM-facing context assembled before a run starts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: ExecutionMode
    run_id: str
    goal: str
    session_id: str | None = None
    episodic_messages: list[AnthropicMessage] = Field(default_factory=list)
    semantic_memory: list[SemanticMemoryItem] = Field(default_factory=list)
    global_context: str = ""
    project_context: str = ""

    @classmethod
    def isolated(
        cls,
        *,
        goal: str,
        run_id: str,
        global_context: str = "",
        project_context: str = "",
    ) -> ExecutionContext:
        return cls(
            mode=ExecutionMode.ISOLATED,
            run_id=run_id,
            goal=goal,
            global_context=global_context,
            project_context=project_context,
        )

    def llm_messages(self) -> list[AnthropicMessage]:
        if self.mode == ExecutionMode.ISOLATED:
            return [AnthropicMessage.user_text(self.goal)]

        messages = list(self.episodic_messages)
        if not messages:
            messages.append(AnthropicMessage.user_text(self.goal))
        return messages

    def system_prompt(self, base: str = BASE_SYSTEM_PROMPT) -> str:
        parts = [base]
        if self.global_context.strip():
            parts.append("\n\n## Global Context\n" + self.global_context.strip())
        if self.project_context.strip():
            parts.append("\n\n## Project Context\n" + self.project_context.strip())
        if session_notes := self.session_notes():
            parts.append("\n\n" + session_notes)
        return "".join(parts)

    def session_notes(self) -> str:
        if not self.semantic_memory:
            return ""

        lines = ["## Session Notes"]
        for item in self.semantic_memory:
            content = item.content.strip()
            if item.kind == SemanticMemoryKind.NOTE and content.startswith("## Note"):
                lines.append(content)
            else:
                label = item.kind.value
                lines.append(f"- {label}: {content}")
        lines.extend(["", "Remember important durable facts by calling note_save."])
        return "\n".join(lines)

    def system_prompt_patch(self) -> str | None:
        session_notes = self.session_notes().strip()
        return session_notes or None


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
    system_prompt_patch: str | None = None
    messages: list[AnthropicMessage] = Field(default_factory=list)

    @classmethod
    def from_goal(cls, goal: str, *, run_id: str = "local", max_steps: int = 1) -> WorkingMemory:
        return cls(
            run_id=run_id,
            goal=goal,
            max_steps=max_steps,
            messages=[AnthropicMessage.user_text(goal)],
        )

    @classmethod
    def from_execution_context(
        cls,
        execution_context: ExecutionContext,
        *,
        max_steps: int,
    ) -> WorkingMemory:
        return cls(
            run_id=execution_context.run_id,
            goal=execution_context.goal,
            max_steps=max_steps,
            system_prompt_patch=execution_context.system_prompt(),
            messages=execution_context.llm_messages(),
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

    def is_done(self) -> bool:
        return self.status in {
            RunStatus.COMPLETED,
            RunStatus.CANCELLED,
            RunStatus.FAILED,
        }

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
