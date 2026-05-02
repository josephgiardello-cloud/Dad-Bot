"""Frozen turn-spine contract markers for the Phase R1 migration.

This module is a zero-runtime-dependency anchor for the turn-spine refactor.
It records the canonical production entrypoint and the legacy surfaces that may
remain as compatibility wrappers until caller migration completes.
"""

from __future__ import annotations

CANONICAL_TURN_ENTRYPOINT = "dadbot.core.dadbot.DadBot.execute_turn"

LEGACY_FACADE_ENTRYPOINTS: tuple[str, ...] = (
    "process_user_message",
    "process_user_message_async",
    "process_user_message_stream",
    "process_user_message_stream_async",
    "handle_turn_sync",
    "handle_turn_async",
    "run_turn",
)

LEGACY_SERVICE_ENTRYPOINTS: tuple[str, ...] = (
    "TurnService.process_user_message",
    "TurnService.process_user_message_async",
    "TurnService.process_user_message_stream",
    "TurnService.process_user_message_stream_async",
)

INTERNAL_KERNEL_EXECUTORS: tuple[str, ...] = (
    "DadBotTurnMixin._run_graph_turn_async",
    "DadBotOrchestrator.handle_turn",
    "ExecutionControlPlane.submit_turn",
)

PRODUCTION_CALL_POLICY = (
    "Production callers under dadbot/ must call the canonical execute_turn adapter. "
    "Legacy entrypoints remain compatibility wrappers only."
)


def canonical_turn_entrypoint() -> str:
    return CANONICAL_TURN_ENTRYPOINT


__all__ = [
    "CANONICAL_TURN_ENTRYPOINT",
    "INTERNAL_KERNEL_EXECUTORS",
    "LEGACY_FACADE_ENTRYPOINTS",
    "LEGACY_SERVICE_ENTRYPOINTS",
    "PRODUCTION_CALL_POLICY",
    "canonical_turn_entrypoint",
]