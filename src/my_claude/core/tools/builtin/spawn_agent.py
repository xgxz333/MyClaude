"""Tool for spawning an isolated child agent."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from my_claude.agent.agent import Agent
from my_claude.agent.control import LoopController
from my_claude.agent.events import (
    AgentEvent,
    SubagentFinishedEvent,
    SubagentStartedEvent,
)
from my_claude.agent.memory import WorkingMemory
from my_claude.agent.tools import ToolRegistry
from my_claude.core.agents import AgentProfile, AgentProfileLoader, BackgroundTaskRegistry
from my_claude.core.context import ExecutionContext, ExecutionMode
from my_claude.core.events.bus import EventBus
from my_claude.core.tools.base import ParamsModel, ToolDefinition, ToolResult
from my_claude.llm.client import LLMClient

MAX_SUBAGENT_DEPTH = 2


class SpawnAgentParams(BaseModel):
    """Arguments accepted by the spawn_agent tool."""

    model_config = ConfigDict(extra="ignore")

    description: Annotated[str, Field(strict=True)]
    prompt: Annotated[str, Field(strict=True)]
    run_in_background: bool = False
    subagent_type: str = ""


ChildRegistryBuilder = Callable[
    [EventBus[AgentEvent], str, int, AgentProfile | None],
    ToolRegistry,
]
RunIdFactory = Callable[[], str]


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class SpawnAgentTool:
    """Spawn a child agent with isolated memory, events, and tools."""

    params_model: ClassVar[ParamsModel] = SpawnAgentParams

    llm_client: LLMClient
    workspace_root: Path
    max_steps: int
    parent_event_bus: EventBus[AgentEvent] | None = None
    parent_run_id: str | None = None
    depth: int = 0
    profile_loader: AgentProfileLoader = field(default_factory=AgentProfileLoader)
    background_registry: BackgroundTaskRegistry = field(default_factory=BackgroundTaskRegistry)
    child_registry_builder: ChildRegistryBuilder | None = None
    run_id_factory: RunIdFactory = field(default=_new_run_id)

    @property
    def name(self) -> str:
        return "spawn_agent"

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=(
                "Spawn an isolated child agent for a subtask and return its final output. "
                "Parent conversation history is not inherited; include all needed context "
                "in the prompt."
            ),
            input_schema=SpawnAgentParams.model_json_schema(),
        )

    async def run(self, arguments: dict[str, object]) -> ToolResult:
        params = SpawnAgentParams.model_validate(arguments)
        if self.depth >= MAX_SUBAGENT_DEPTH:
            return ToolResult(
                content="Subagent nesting limit (2) reached; cannot spawn further subagents.",
                is_error=True,
                error="subagent nesting limit reached",
                error_type="nesting_limit",
            )
        child_run_id = self.run_id_factory()
        profile = self._resolve_profile(params.subagent_type)
        child_context = self._create_child_context(
            child_run_id=child_run_id,
            prompt=params.prompt,
            profile=profile,
        )
        await self._publish_parent(
            SubagentStartedEvent(
                run_id=child_run_id,
                parent_run_id=self.parent_run_id or "",
                description=params.description,
                subagent_type=params.subagent_type or None,
            )
        )
        if params.run_in_background:
            task = asyncio.create_task(
                self._run_child_with_lifecycle(
                    params=params,
                    execution_context=child_context,
                    profile=profile,
                )
            )
            task.add_done_callback(_consume_background_exception)
            self.background_registry.register(child_run_id, task, child_context)
            return ToolResult.success(
                "Subagent started in background. "
                f"run_id={child_run_id}. "
                f"Use agent_result(run_id='{child_run_id}') to retrieve result."
            )

        try:
            result = await self._run_child_with_lifecycle(
                params=params,
                execution_context=child_context,
                profile=profile,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return ToolResult.failure(str(error))
        return ToolResult.success(result)

    def _create_child_context(
        self,
        *,
        child_run_id: str,
        prompt: str,
        profile: AgentProfile | None,
    ) -> ExecutionContext:
        return ExecutionContext(
            mode=ExecutionMode.ISOLATED,
            run_id=child_run_id,
            goal=prompt,
            system_prompt_override=profile.system_prompt if profile else None,
            tool_whitelist=profile.allowed_tools if profile and profile.allowed_tools else None,
        )

    async def _run_child_with_lifecycle(
        self,
        *,
        params: SpawnAgentParams,
        execution_context: ExecutionContext,
        profile: AgentProfile | None,
    ) -> str:
        try:
            result = await self._run_child_agent(
                execution_context=execution_context,
                profile=profile,
            )
        except asyncio.CancelledError:
            message = "Subagent was cancelled."
            _set_context_result(execution_context, message)
            await self._publish_parent(
                SubagentFinishedEvent(
                    run_id=execution_context.run_id,
                    parent_run_id=self.parent_run_id or "",
                    description=params.description,
                    subagent_type=params.subagent_type or None,
                    result=message,
                    is_error=True,
                )
            )
            raise
        except Exception as error:
            error_text = str(error)
            _set_context_result(execution_context, error_text)
            await self._publish_parent(
                SubagentFinishedEvent(
                    run_id=execution_context.run_id,
                    parent_run_id=self.parent_run_id or "",
                    description=params.description,
                    subagent_type=params.subagent_type or None,
                    result=error_text,
                    is_error=True,
                )
            )
            raise

        _set_context_result(execution_context, result)
        await self._publish_parent(
            SubagentFinishedEvent(
                run_id=execution_context.run_id,
                parent_run_id=self.parent_run_id or "",
                description=params.description,
                subagent_type=params.subagent_type or None,
                result=result,
                is_error=False,
            )
        )
        return result

    async def _run_child_agent(
        self,
        *,
        execution_context: ExecutionContext,
        profile: AgentProfile | None,
    ) -> str:
        working_memory = WorkingMemory.from_execution_context(
            execution_context,
            max_steps=self.max_steps,
        )
        child_bus: EventBus[AgentEvent] = EventBus()
        child_bus.subscribe(self._bridge_child_event)
        child_registry = self._build_child_registry(
            child_bus,
            execution_context.run_id,
            profile,
        )
        agent = Agent(
            llm_client=self.llm_client,
            tools=child_registry,
            working_memory=working_memory,
            loop_controller=LoopController(max_iterations=self.max_steps),
            event_handler=child_bus.publish,
        )
        result = await agent.run(emit_run_started=True)
        return result.final_response

    def _build_child_registry(
        self,
        child_bus: EventBus[AgentEvent],
        child_run_id: str,
        profile: AgentProfile | None,
    ) -> ToolRegistry:
        if self.child_registry_builder is None:
            return ToolRegistry()
        return self.child_registry_builder(
            child_bus,
            child_run_id,
            self.depth + 1,
            profile,
        )

    def _resolve_profile(self, subagent_type: str) -> AgentProfile | None:
        if not subagent_type:
            return None
        return self.profile_loader.resolve(subagent_type)

    async def _publish_parent(self, event: AgentEvent) -> None:
        if self.parent_event_bus is not None:
            await self.parent_event_bus.publish(event)

    async def _bridge_child_event(self, event: AgentEvent) -> None:
        if self.parent_event_bus is not None:
            await self.parent_event_bus.publish(event)


def _set_context_result(context: ExecutionContext, result: str) -> None:
    object.__setattr__(context, "result", result)


def _consume_background_exception(task: asyncio.Task[str]) -> None:
    if task.cancelled():
        return
    try:
        task.exception()
    except asyncio.CancelledError:
        return
