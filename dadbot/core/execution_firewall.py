from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ExecutionBlockedInvariantError(RuntimeError):
    """Raised when the execution firewall blocks a forbidden runtime path."""


_DEFAULT_LEGACY_REGISTRY = {
    "handle_graph_failure",
    "legacy",
    "direct_path",
    "fallback",
    "graph_turns_enabled",
}


@dataclass
class FirewallContext:
    """Normalized context used by the execution firewall."""

    trace_id: str = ""
    stage: str = ""
    mutation_outside_save_node: bool = False
    temporal_missing: bool = False
    metadata: dict[str, Any] | None = None


class ExecutionFirewall:
    """Central firewall that blocks quarantined or forbidden execution paths."""

    def __init__(self, *, quarantine_path: Path | None = None) -> None:
        self.quarantine_path = quarantine_path or Path("runtime") / "phase4_quarantine_registry.json"
        self._legacy_registry = set(_DEFAULT_LEGACY_REGISTRY)
        self._quarantined_symbols: set[str] = set()
        self._load_quarantine_registry()

    @staticmethod
    def _normalize_symbol(value: str) -> str:
        return str(value or "").strip().lower()

    def _load_quarantine_registry(self) -> None:
        self._quarantined_symbols = set()
        try:
            if not self.quarantine_path.exists():
                return
            payload = json.loads(self.quarantine_path.read_text(encoding="utf-8"))
            symbols = payload.get("quarantined_symbols", []) if isinstance(payload, dict) else []
            if isinstance(symbols, list):
                for item in symbols:
                    symbol = self._normalize_symbol(str(item or ""))
                    if symbol:
                        self._quarantined_symbols.add(symbol)
        except Exception:  # noqa: BLE001
            # Firewall must stay fail-safe and never crash loading quarantine state.
            self._quarantined_symbols = set()

    def is_blocked(self, call_site: str) -> bool:
        symbol = self._normalize_symbol(call_site)
        if not symbol:
            return False
        if symbol in self._legacy_registry:
            return True
        # Match exact symbol and dotted suffix matches.
        if symbol in self._quarantined_symbols:
            return True
        return any(symbol.endswith(f".{blocked}") for blocked in self._quarantined_symbols)

    def enforce_execution_firewall(
        self,
        call_site: str,
        context: FirewallContext,
    ) -> None:
        symbol = self._normalize_symbol(call_site)
        if self.is_blocked(symbol):
            raise ExecutionBlockedInvariantError(
                f"Legacy/quarantined execution blocked by Phase 4 firewall: {call_site!r}",
            )

        if bool(context.mutation_outside_save_node):
            raise ExecutionBlockedInvariantError(
                f"Mutation outside SaveNode blocked by Phase 4 firewall "
                f"(trace_id={context.trace_id!r}, stage={context.stage!r})",
            )

        if bool(context.temporal_missing):
            raise ExecutionBlockedInvariantError(
                f"TemporalNode violation blocked by Phase 4 firewall "
                f"(trace_id={context.trace_id!r}, stage={context.stage!r})",
            )


__all__ = ["ExecutionBlockedInvariantError", "ExecutionFirewall", "FirewallContext"]
