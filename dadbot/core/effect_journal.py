from __future__ import annotations

import hashlib
from typing import Any

from dadbot.core.ledger_index import LedgerIndex
from dadbot.core.ledger_writer_adapter import LedgerWriterAdapter


class EffectJournal:
    """Durable effect begin/commit journal anchored in the execution ledger."""

    def __init__(self, *, writer: LedgerWriterAdapter, index: LedgerIndex) -> None:
        self._writer = writer
        self._index = index

    @staticmethod
    def derive_effect_id(*, session_id: str, request_id: str, trace_token: str = "", **legacy_kwargs: Any) -> str:
        legacy_trace = legacy_kwargs.pop("trace_id", "")
        if legacy_kwargs:
            unknown = ", ".join(sorted(str(name) for name in legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        token = str(request_id or "").strip()
        if token:
            return f"eff:{token}"
        seed = f"{str(session_id or 'default')}|{str(trace_token or legacy_trace or '')}"
        return f"eff:{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:20]}"

    def begin(
        self,
        *,
        session_id: str,
        trace_token: str = "",
        effect_id: str,
        request_id: str,
        step_key: str = "scheduler.execute.effect.begin",
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        legacy_trace = legacy_kwargs.pop("trace_id", "")
        legacy_step = legacy_kwargs.pop("kernel_step_id", "")
        if legacy_kwargs:
            unknown = ", ".join(sorted(str(name) for name in legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        event = self._writer.write_event(
            event_type="EFFECT_BEGIN",
            session_id=str(session_id or "default"),
            trace_token=str(trace_token or legacy_trace or "").strip(),
            step_key=str(step_key or legacy_step or "scheduler.execute.effect.begin"),
            payload={
                "effect_id": str(effect_id or "").strip(),
                "request_id": str(request_id or "").strip(),
            },
            committed=False,
        )
        self._index.refresh(force=True)
        return event

    def commit(
        self,
        *,
        session_id: str,
        trace_token: str = "",
        effect_id: str,
        request_id: str,
        step_key: str = "scheduler.execute.effect.commit",
        **legacy_kwargs: Any,
    ) -> dict[str, Any]:
        legacy_trace = legacy_kwargs.pop("trace_id", "")
        legacy_step = legacy_kwargs.pop("kernel_step_id", "")
        if legacy_kwargs:
            unknown = ", ".join(sorted(str(name) for name in legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        event = self._writer.write_event(
            event_type="EFFECT_COMMIT",
            session_id=str(session_id or "default"),
            trace_token=str(trace_token or legacy_trace or "").strip(),
            step_key=str(step_key or legacy_step or "scheduler.execute.effect.commit"),
            payload={
                "effect_id": str(effect_id or "").strip(),
                "request_id": str(request_id or "").strip(),
            },
            committed=True,
        )
        self._index.refresh(force=True)
        return event

    def is_committed(self, *, session_id: str, effect_id: str) -> bool:
        return self._index.is_effect_committed(session_id=session_id, effect_id=effect_id)

    def is_ambiguous(self, *, session_id: str, effect_id: str) -> bool:
        return self._index.has_ambiguous_effect_inflight(session_id=session_id, effect_id=effect_id)


__all__ = ["EffectJournal"]
