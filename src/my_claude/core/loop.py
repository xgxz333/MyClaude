"""ReAct agent loop: Plan, Observe, Act, then terminate or continue."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

from my_claude.agent.control import LoopController
from my_claude.agent.events import (
    AgentEvent,
    EventHandler,
    LLMRequestStartedEvent,
    LLMResponseCompletedEvent,
    RunCancelledEvent,
    RunCompletedEvent,
    RunFailedEvent,
    StepFinishedEvent,
    StepStartedEvent,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
    dispatch_event,
)
from my_claude.agent.tools import ToolRegistry
from my_claude.core.context import RunStatus, ToolUseBlock, WorkingMemory
from my_claude.llm.client import LLMClient, LLMResponse

CancelRequested = Callable[[], bool]


@dataclass(frozen=True)
class AgentLoopResult:
    """Final result produced after the ReAct loop terminates."""

    final_response: str


class AgentLoop:
    """Drive the ReAct while-loop over working memory, LLM responses, and tool calls."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tools: ToolRegistry,
        working_memory: WorkingMemory,
        loop_controller: LoopController,
        event_handler: EventHandler | None = None,
        cancel_requested: CancelRequested | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._tools = tools
        self._working_memory = working_memory
        self._loop_controller = loop_controller
        self._event_handler = event_handler
        self._cancel_requested = cancel_requested

    async def run(self) -> AgentLoopResult:
        last_response = ""

        try:
            while True:
                self._raise_if_cancelled("cancelled before planning")
                await self._start_next_step()

                response = await self._plan()
                self._raise_if_cancelled("cancelled after planning")

                assistant_message = response.assistant_message()
                self._working_memory.append_assistant_message(assistant_message)
                last_response = response.content
                await self._emit(
                    LLMResponseCompletedEvent(
                        run_id=self._working_memory.run_id,
                        message=response.content,
                        content=response.content,
                        has_raw_response=response.raw is not None,
                    )
                )

                tool_uses = response.tool_uses()
                if tool_uses:
                    await self._act(tool_uses)
                    self._raise_if_cancelled("cancelled after acting")

                    if not self._loop_controller.should_continue():
                        self._working_memory.set_status(
                            RunStatus.FAILED,
                            reason="agent loop reached max_steps",
                            transition_reason="safety fuse triggered after tool calls",
                        )
                        await self._finish_step()
                        raise RuntimeError("agent loop reached max_steps")

                    await self._finish_step()
                    continue

                self._working_memory.set_status(
                    RunStatus.COMPLETED,
                    reason="agent run completed",
                    transition_reason="llm response did not request tool calls",
                )
                await self._finish_step()
                await self._emit(
                    RunCompletedEvent(
                        goal=self._working_memory.goal,
                        run_id=self._working_memory.run_id,
                        steps=self._working_memory.step,
                    )
                )
                return AgentLoopResult(final_response=last_response)
        except asyncio.CancelledError:
            await self._mark_cancelled("async cancellation requested")
            raise
        except Exception as error:
            if self._working_memory.status != RunStatus.FAILED:
                self._working_memory.set_status(
                    RunStatus.FAILED,
                    reason=str(error),
                    transition_reason="agent loop raised an exception",
                )
            await self._emit(RunFailedEvent(message=str(error), error=str(error)))
            raise

    async def _plan(self) -> LLMResponse:
        messages = self._working_memory.llm_messages()
        tools = self._tools.definitions()
        await self._emit(
            LLMRequestStartedEvent(
                run_id=self._working_memory.run_id,
                model_input_messages=len(messages),
                tools=len(tools),
            )
        )
        return await self._llm_client.complete(
            messages,
            tools,
            step=self._working_memory.step,
            system_prompt_patch=self._working_memory.system_prompt_patch,
        )

    async def _act(self, tool_uses: list[ToolUseBlock]) -> None:
        for tool_use in tool_uses:
            self._raise_if_cancelled(f"cancelled before tool call {tool_use.name}")
            await self._emit(
                ToolCallStartedEvent(
                    run_id=self._working_memory.run_id,
                    tool_use_id=tool_use.id,
                    tool_name=tool_use.name,
                    arguments=tool_use.input,
                )
            )

            started_at = time.monotonic()
            result = await self._tools.call(tool_use.name, tool_use.input)
            elapsed_ms = int((time.monotonic() - started_at) * 1000)

            self._working_memory.append_tool_result(
                tool_use_id=tool_use.id,
                content=result.content,
                is_error=result.is_error,
            )
            await self._emit(
                ToolCallCompletedEvent(
                    run_id=self._working_memory.run_id,
                    tool_use_id=tool_use.id,
                    tool_name=tool_use.name,
                    result=None if result.is_error else result.content,
                    error=result.error,
                    error_type=result.error_type,
                    elapsed_ms=elapsed_ms,
                )
            )

    async def _start_next_step(self) -> None:
        iteration = self._loop_controller.mark_iteration_started()
        step = self._working_memory.mark_step_started(
            reason=f"agent loop iteration {iteration} started",
        )
        await self._emit(
            StepStartedEvent(run_id=self._working_memory.run_id, step=step)
        )

    async def _finish_step(self) -> None:
        await self._emit(
            StepFinishedEvent(
                run_id=self._working_memory.run_id,
                step=self._working_memory.step,
            )
        )

    def _raise_if_cancelled(self, reason: str) -> None:
        if self._cancel_requested is not None and self._cancel_requested():
            self._working_memory.set_status(
                RunStatus.CANCELLED,
                reason="agent run cancelled",
                transition_reason=reason,
            )
            raise asyncio.CancelledError(reason)

    async def _mark_cancelled(self, transition_reason: str) -> None:
        if self._working_memory.status != RunStatus.CANCELLED:
            self._working_memory.set_status(
                RunStatus.CANCELLED,
                reason="agent run cancelled",
                transition_reason=transition_reason,
            )
        await self._emit(RunCancelledEvent(reason=self._working_memory.status_transition_reason))

    async def _emit(self, event: AgentEvent) -> None:
        await dispatch_event(self._event_handler, event)
