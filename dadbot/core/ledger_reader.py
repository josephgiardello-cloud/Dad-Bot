from __future__ import annotations

from dadbot.core.execution_ledger import ExecutionLedger

POLICY_TRACE_EVENT_TYPE = "PolicyTraceEvent"


class LedgerReader:
    """Read-only helper for querying execution lifecycle state."""

    def __init__(self, ledger: ExecutionLedger):
        self._ledger = ledger

    def events(self) -> list[dict[str, object]]:
        return self._ledger.read()

    def events_for_job(self, job_id: str) -> list[dict[str, object]]:
        jid = str(job_id or "")
        return [
            e
            for e in self._ledger.read()
            if str((e.get("payload") or {}).get("job_id") or e.get("job_id") or "") == jid
        ]

    def events_by_type(self, event_type: str) -> list[dict[str, object]]:
        target = str(event_type or "").strip()
        if not target:
            return []
        return [e for e in self._ledger.read() if str(e.get("type") or "") == target]

    @staticmethod
    def _resolve_policy_trace_filters(
        *,
        session_id: str,
        trace_token: str,
        legacy_kwargs: dict[str, object],
    ) -> tuple[str, str]:
        legacy_trace = str(legacy_kwargs.pop("trace_id", "") or "")
        if legacy_kwargs:
            unknown = ", ".join(sorted(str(name) for name in legacy_kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        sid = str(session_id or "").strip()
        tid = str(trace_token or legacy_trace or "").strip()
        return sid, tid

    @staticmethod
    def _filter_policy_events(
        events: list[dict[str, object]],
        *,
        session_id: str,
        trace_token: str,
    ) -> list[dict[str, object]]:
        if session_id:
            events = [event for event in events if str(event.get("session_id") or "") == session_id]
        if trace_token:
            events = [event for event in events if str(event.get("trace_id") or "") == trace_token]
        return events

    def policy_trace_events(
        self,
        *,
        session_id: str = "",
        trace_token: str = "",
        limit: int = 0,
        **legacy_kwargs: object,
    ) -> list[dict[str, object]]:
        sid, tid = self._resolve_policy_trace_filters(
            session_id=session_id,
            trace_token=trace_token,
            legacy_kwargs=dict(legacy_kwargs),
        )
        events = self.events_by_type(POLICY_TRACE_EVENT_TYPE)
        events = self._filter_policy_events(events, session_id=sid, trace_token=tid)
        if limit and limit > 0:
            return events[-int(limit):]
        return events

    @staticmethod
    def _aggregate_policy_events_summary(
        events: list[dict[str, object]],
    ) -> tuple[dict[str, int], set[str], str, str, str]:
        action_counts: dict[str, int] = {}
        policies_seen: set[str] = set()
        latest_action = ""
        latest_step_name = ""
        latest_trace_id = ""

        for event in events:
            latest_trace_id = str(event.get("trace_id") or latest_trace_id)
            payload = dict(event.get("payload") or {})
            summary = dict(payload.get("summary") or {})
            policy = str(summary.get("policy") or payload.get("policy") or "").strip()
            if policy:
                policies_seen.add(policy)
            action = str(
                summary.get("decision_action")
                or summary.get("action")
                or payload.get("action")
                or "",
            ).strip()
            if action:
                action_counts[action] = action_counts.get(action, 0) + 1
                latest_action = action
            step_name = str(
                summary.get("step_name")
                or payload.get("step_name")
                or "",
            ).strip()
            if step_name:
                latest_step_name = step_name

        return action_counts, policies_seen, latest_action, latest_step_name, latest_trace_id

    @staticmethod
    def _build_policy_summary_dict(
        events: list[dict[str, object]],
        action_counts: dict[str, int],
        policies_seen: set[str],
        latest_action: str,
        latest_step_name: str,
        latest_trace_id: str,
    ) -> dict[str, object]:
        return {
            "event_type": POLICY_TRACE_EVENT_TYPE,
            "event_count": len(events),
            "policies": sorted(policies_seen),
            "action_counts": action_counts,
            "latest_action": latest_action,
            "latest_step_name": latest_step_name,
            "latest_trace_id": latest_trace_id,
        }

    def summarize_policy_trace_events(
        self,
        *,
        session_id: str = "",
        trace_token: str = "",
        limit: int = 0,
        **legacy_kwargs: object,
    ) -> dict[str, object]:
        _, resolved_trace_token = self._resolve_policy_trace_filters(
            session_id=session_id,
            trace_token=trace_token,
            legacy_kwargs=dict(legacy_kwargs),
        )
        events = self.policy_trace_events(
            session_id=session_id,
            trace_token=resolved_trace_token,
            limit=limit,
        )

        action_counts, policies_seen, latest_action, latest_step_name, latest_trace_id = (
            self._aggregate_policy_events_summary(events)
        )

        return self._build_policy_summary_dict(
            events,
            action_counts,
            policies_seen,
            latest_action,
            latest_step_name,
            latest_trace_id,
        )

    def is_terminal(self, job_id: str) -> bool:
        for event in reversed(self.events_for_job(job_id)):
            event_type = str(event.get("type") or "")
            if event_type in {"JOB_COMPLETED", "JOB_FAILED"}:
                return True
            if event_type in {"JOB_STARTED", "JOB_QUEUED", "JOB_SUBMITTED"}:
                return False
        return False
