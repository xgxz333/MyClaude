"""Loop control primitives for limiting agent iterations."""

from __future__ import annotations


class LoopController:
    """Track iteration count and decide whether the agent loop may continue."""

    def __init__(self, max_iterations: int = 1) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be greater than 0")
        self._max_iterations = max_iterations
        self._iterations = 0

    def should_continue(self) -> bool:
        return self._iterations < self._max_iterations

    def mark_iteration_started(self) -> int:
        if not self.should_continue():
            raise RuntimeError("agent loop cannot continue")

        self._iterations += 1
        return self._iterations
