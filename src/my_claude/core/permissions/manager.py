"""Async permission manager for tool approval requests."""

from __future__ import annotations

import asyncio
import datetime
import inspect
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

from my_claude.core.permissions.policy import (
    DEFAULT_POLICIES,
    PermissionDecision,
    ToolPolicy,
    matches_outside_cwd,
    param_preview,
)
from my_claude.core.permissions.policy import (
    evaluate as evaluate_policy,
)
from my_claude.core.permissions.storage import (
    AlwaysDecision,
    load_policy_file,
    save_policy_file,
)

logger = logging.getLogger(__name__)

PermissionEvent = dict[str, Any]
PermissionEventEmitter = Callable[[PermissionEvent], Awaitable[None] | None]


def _now() -> str:
    return datetime.datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class PendingPermissionRequest:
    """State kept while a tool call waits for external user approval."""

    future: asyncio.Future[str]
    session_id: str
    tool_name: str


async def _noop_event_emitter(_event: PermissionEvent) -> None:
    return None


class PermissionManager:
    """Evaluate static policy, suspend ASK requests, and persist always decisions."""

    def __init__(
        self,
        policies: Mapping[str, ToolPolicy] | None = None,
        *,
        event_emitter: PermissionEventEmitter | None = None,
        policy_file: str | Path | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        self.event_emitter = event_emitter or _noop_event_emitter
        self._policies: dict[str, ToolPolicy] = dict(policies or DEFAULT_POLICIES)
        self._pending: dict[str, PendingPermissionRequest] = {}
        self._session_always: dict[tuple[str, str], AlwaysDecision] = {}
        self._policy_file = Path(policy_file) if policy_file is not None else None
        self._persistent_always: dict[str, AlwaysDecision] = (
            load_policy_file(self._policy_file) if self._policy_file is not None else {}
        )
        self._timeout_s = timeout_s

    def evaluate(self, tool_name: str, params: dict[str, Any]) -> PermissionDecision:
        """Evaluate static policy without async user interaction."""

        return evaluate_policy(tool_name, params, self._policies.get(tool_name))

    async def authorize(
        self,
        *,
        tool_use_id: str,
        tool_name: str,
        params: dict[str, Any],
        session_id: str | None = None,
        preview: str | None = None,
        policy: ToolPolicy | None = None,
        event_emitter: PermissionEventEmitter | None = None,
    ) -> PermissionDecision:
        """Compatibility wrapper returning a coarse allow/deny enum."""

        allowed, _decision = await self.check_and_wait(
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            params=params,
            session_id=session_id or "",
            preview=preview,
            policy=policy,
            event_emitter=event_emitter,
        )
        return PermissionDecision.ALLOW if allowed else PermissionDecision.DENY

    async def check_and_wait(
        self,
        *,
        tool_use_id: str,
        tool_name: str,
        params: dict[str, Any],
        session_id: str,
        preview: str | None = None,
        policy: ToolPolicy | None = None,
        event_emitter: PermissionEventEmitter | None = None,
    ) -> tuple[bool, str]:
        """Check permission and wait for external approval when policy asks.

        The order is security-sensitive and mirrors KamaClaude stage/s5:
        deny patterns run first, bash outside-CWD detection forces ASK before
        cache lookup, then session and persistent always caches may apply, and
        only then do allow/default rules decide the automatic fallback.
        """

        resolved_policy = policy or self._policies.get(tool_name)
        command = _bash_command(tool_name, params)

        if command and resolved_policy is not None:
            for pattern in resolved_policy.deny_patterns:
                if re.search(pattern, command):
                    logger.debug("permission: deny_pattern hit tool=%s", tool_name)
                    return False, "auto_deny"

        outside_cwd = bool(command and matches_outside_cwd(command))

        if not outside_cwd:
            cached = self._cached_decision(session_id, tool_name)
            if cached is not None:
                logger.debug(
                    "permission: cache hit tool=%s decision=%s",
                    tool_name,
                    cached,
                )
                return cached == "allow", f"auto_{cached}"

            if command and resolved_policy is not None:
                for pattern in resolved_policy.allow_patterns:
                    if re.search(pattern, command):
                        return True, "auto_allow"

            if resolved_policy is not None:
                if resolved_policy.default is PermissionDecision.ALLOW:
                    return True, "auto_allow"
                if resolved_policy.default is PermissionDecision.DENY:
                    return False, "auto_deny"

        return await self._ask_user(
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            params=params,
            session_id=session_id,
            preview=preview,
            event_emitter=event_emitter,
        )

    async def request_approval(
        self,
        *,
        tool_use_id: str,
        tool_name: str,
        params: dict[str, Any],
        session_id: str | None = None,
        preview: str | None = None,
        event_emitter: PermissionEventEmitter | None = None,
    ) -> bool:
        """Compatibility wrapper that only performs the ASK workflow."""

        allowed, _decision = await self._ask_user(
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            params=params,
            session_id=session_id or "",
            preview=preview,
            event_emitter=event_emitter,
        )
        return allowed

    async def _ask_user(
        self,
        *,
        tool_use_id: str,
        tool_name: str,
        params: dict[str, Any],
        session_id: str,
        preview: str | None = None,
        event_emitter: PermissionEventEmitter | None = None,
    ) -> tuple[bool, str]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending[tool_use_id] = PendingPermissionRequest(
            future=future,
            session_id=session_id,
            tool_name=tool_name,
        )

        await self._emit(
            {
                "type": "permission.requested",
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "params": dict(params),
                "param_preview": preview or param_preview(tool_name, params),
                "session_id": session_id,
                "ts": _now(),
            },
            event_emitter=event_emitter,
        )

        try:
            if self._timeout_s > 0:
                raw = await asyncio.wait_for(future, timeout=self._timeout_s)
            else:
                raw = await future
        except TimeoutError:
            self._pending.pop(tool_use_id, None)
            logger.info(
                "permission: timeout tool_use_id=%s tool=%s",
                tool_use_id,
                tool_name,
            )
            return False, "timeout"
        except asyncio.CancelledError:
            self._pending.pop(tool_use_id, None)
            if not future.done():
                future.cancel()
            raise

        return self._apply_response(raw, session_id, tool_name), raw

    def respond(self, tool_use_id: str, decision: str) -> None:
        """Resolve a pending approval request from an external user decision."""

        request = self._pending.pop(tool_use_id, None)
        if request is None:
            logger.warning("permission.respond: unknown tool_use_id=%s", tool_use_id)
            return

        if not request.future.done():
            request.future.set_result(decision)

    def cancel_session(self, session_id: str, reason: str = "client_disconnected") -> None:
        """Deny all pending permission requests for a disconnected session."""

        to_cancel = [
            tool_use_id
            for tool_use_id, request in self._pending.items()
            if request.session_id == session_id
        ]
        for tool_use_id in to_cancel:
            request = self._pending.pop(tool_use_id)
            if not request.future.done():
                logger.debug(
                    "permission: cancel pending tool_use_id=%s reason=%s",
                    tool_use_id,
                    reason,
                )
                request.future.set_result("deny_once")

    def pending(self) -> dict[str, PendingPermissionRequest]:
        """Return a shallow copy of pending approval requests."""

        return dict(self._pending)

    async def _emit(
        self,
        event: PermissionEvent,
        *,
        event_emitter: PermissionEventEmitter | None = None,
    ) -> None:
        result = (event_emitter or self.event_emitter)(event)
        if inspect.isawaitable(result):
            await result

    def _cached_decision(
        self,
        session_id: str,
        tool_name: str,
    ) -> AlwaysDecision | None:
        session_decision = self._session_always.get((session_id, tool_name))
        if session_decision is not None:
            return session_decision
        return self._persistent_always.get(tool_name)

    def _apply_response(
        self,
        decision: str | PermissionDecision,
        session_id: str,
        tool_name: str,
    ) -> bool:
        """Apply a user response, update always caches, and return allow/deny."""

        normalized = _normalize_response(decision)
        if normalized is None:
            logger.warning("invalid permission decision %r; denying request", decision)
            return False

        allow = normalized in ("allow_once", "always_allow")
        if normalized in ("always_allow", "always_deny"):
            always_decision = "allow" if normalized == "always_allow" else "deny"
            self._session_always[(session_id, tool_name)] = always_decision
            self._persistent_always[tool_name] = always_decision
            logger.info(
                "permission: always %s tool=%s policy_file=%s",
                always_decision,
                tool_name,
                self._policy_file,
            )
            if self._policy_file is not None:
                save_policy_file(self._persistent_always, self._policy_file)

        return allow


def _bash_command(tool_name: str, params: dict[str, Any]) -> str:
    if tool_name != "bash":
        return ""
    command = params.get("command")
    return command if isinstance(command, str) else ""


def _normalize_response(decision: str | PermissionDecision) -> str | None:
    if isinstance(decision, PermissionDecision):
        return "allow_once" if decision is PermissionDecision.ALLOW else "deny_once"

    normalized = decision.strip().lower()
    if normalized in {"allow", "allow_once", "always_allow"}:
        return "allow_once" if normalized == "allow" else normalized
    if normalized in {"deny", "deny_once", "always_deny"}:
        return "deny_once" if normalized == "deny" else normalized
    return None


__all__ = [
    "PendingPermissionRequest",
    "PermissionEvent",
    "PermissionEventEmitter",
    "PermissionManager",
    "load_policy_file",
    "save_policy_file",
]
