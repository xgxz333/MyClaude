"""Trace record model shared by all S3 instrumentation layers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

TraceDirection = Literal[
    "CLIENT→CORE",
    "CORE→CLIENT",
    "CORE",
    "CORE→LLM",
    "LLM→CORE",
]
TraceLayer = Literal["ipc", "event", "llm"]


class TraceRecord(BaseModel):
    """Single trace entry on the daemon timeline.

    Control-flow fields are intentionally fixed. Business-specific payload stays
    inside ``data`` so producers can evolve without changing the trace envelope.
    """

    ts: str
    direction: TraceDirection
    layer: TraceLayer
    kind: str
    run_id: str | None = None
    step: int | None = None
    client_id: str | None = None
    data: dict[str, Any]
