"""Agent loop implementation that coordinates memory, LLM calls, tools, and events."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from my_claude.agent.control import LoopController
from my_claude.agent.events import (
    AgentEvent,
    EventHandler,
    RunCancelledEvent,
    RunFailedEvent,
    RunStartedEvent,
    dispatch_event,
)
from my_claude.agent.memory import RunStatus, WorkingMemory
from my_claude.agent.tools import ToolRegistry
from my_claude.core.loop import AgentLoop
from my_claude.llm.client import LLMClient


@dataclass(frozen=True)
class AgentResult:
    """The user-visible result of an agent run."""

    goal: str
    final_response: str


class Agent:
    """Run the agent loop with injected LLM, tools, memory, controller, and event handler."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tools: ToolRegistry,
        working_memory: WorkingMemory | None = None,
        loop_controller: LoopController | None = None,
        event_handler: EventHandler | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._tools = tools
        self._working_memory = working_memory
        self._loop_controller = loop_controller or LoopController()
        self._event_handler = event_handler

    async def run(self, goal: str | None = None) -> AgentResult:
        working_memory = self._working_memory
        if working_memory is None:
            if goal is None:
                raise ValueError("goal is required when working memory is not provided")
            working_memory = WorkingMemory.from_goal(goal)

        run_goal = working_memory.goal
        await self._emit(RunStartedEvent(goal=run_goal))

        try:
            loop = AgentLoop(
                llm_client=self._llm_client,
                tools=self._tools,
                working_memory=working_memory,
                loop_controller=self._loop_controller,
                event_handler=self._event_handler,
            )
            result = await loop.run()
            return AgentResult(goal=run_goal, final_response=result.final_response)
        except asyncio.CancelledError:
            if working_memory.status != RunStatus.CANCELLED:
                working_memory.set_status(
                    RunStatus.CANCELLED,
                    reason="agent run cancelled",
                    transition_reason="async cancellation requested",
                )
                await self._emit(RunCancelledEvent())
            raise
        except Exception as error:
            if working_memory.status != RunStatus.FAILED:
                working_memory.set_status(
                    RunStatus.FAILED,
                    reason=str(error),
                    transition_reason="agent loop raised an exception",
                )
                await self._emit(RunFailedEvent(message=str(error), error=str(error)))
            raise

    async def _emit(self, event: AgentEvent) -> None:
        await dispatch_event(self._event_handler, event)
