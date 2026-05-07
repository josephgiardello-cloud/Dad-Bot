from __future__ import annotations

from typing import Any, Callable

from dadbot.core.tool_executor import execute_tool as _execute_tool

# Tool names that may be executed via the agentic tool pipeline.
# This frozenset is the SINGLE authority for tool authorization policy.
# The service layer MUST NOT define its own tool allowlists.
_AUTHORIZED_TOOL_NAMES: frozenset[str] = frozenset({"set_reminder", "web_search"})
_WEB_SEARCH_ONLY_TOOL_NAME = "web_search"


class DadBotActionMixin:
    """Convenience actions layered on top of the manager-owned runtime surface."""

    @staticmethod
    def _service_allowed_tools(policy: Any) -> set[str] | None:
        if not isinstance(policy, dict):
            return None
        raw_tools = policy.get("allowed_tools")
        if raw_tools is None:
            return None
        return {str(tool_name or "").strip() for tool_name in list(raw_tools or []) if str(tool_name or "").strip()}

    def authorize_tool_execution(self, tool_name: str) -> bool:
        """Single authority for whether a named tool may be executed.

        All policy decisions about which tools are executable flow through
        this method.  The service layer MUST NOT maintain its own tool
        allowlists — this is the only decision point.
        """
        normalized_tool_name = str(tool_name or "").strip()
        if normalized_tool_name not in _AUTHORIZED_TOOL_NAMES:
            return False
        service_policy = getattr(self, "_service_request_policy", None)
        allowed_tools = self._service_allowed_tools(service_policy)
        if allowed_tools is None:
            return True
        return normalized_tool_name in allowed_tools

    def authorize_tool_execution_for_bias(self, tool_name: str, tool_bias: str) -> bool:
        """Bias-aware tool authorization owned by the runtime contract.

        The service layer may ask whether a candidate tool is allowed under the
        current Bayesian bias, but it must not encode named-tool policy itself.
        """
        normalized_tool = str(tool_name or "").strip()
        normalized_bias = str(tool_bias or "planner_default").strip() or "planner_default"
        if not self.authorize_tool_execution(normalized_tool):
            return False
        if normalized_bias == "defer_tools_unless_explicit":
            return False
        if normalized_bias == "minimal_tools":
            return normalized_tool != _WEB_SEARCH_ONLY_TOOL_NAME
        return True

    def execute_tool(
        self,
        *,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        executor: Callable[[], Any],
        compensating_action: Callable[[], None] | None = None,
    ):
        """Runtime-owned entrypoint for the canonical tool execution spine."""
        return _execute_tool(
            tool_name=tool_name,
            parameters=parameters,
            executor=executor,
            compensating_action=compensating_action,
        )

    def _queue_or_apply_memory_patch(self, **patch):
        normalized_patch = dict(patch or {})
        # Apply immediately for test visibility and correctness.
        # During graph execution commits, this ensures mutations are visible to observers.
        self.memory.mutate_memory_store(**normalized_patch, save=True)
        return "applied"

    def record_relationship_history_point(
        self,
        *,
        trust_level,
        openness_level,
        source="turn",
    ):
        history = list(self.memory.relationship_history(limit=180))
        point = {
            "recorded_at": self.runtime_timestamp(),
            "trust_level": self.clamp_score(trust_level),
            "openness_level": self.clamp_score(openness_level),
            "source": str(source or "turn").strip().lower() or "turn",
        }
        history.append(point)
        self._queue_or_apply_memory_patch(relationship_history=history[-180:])
        return point

    def soft_reset_session_context(self, preserve_recent_summary=True):
        preserved_summary = str(self.session_summary or "").strip()
        if not preserved_summary:
            transcript = self.transcript_from_messages(self.history[-8:])
            if transcript:
                preserved_summary = transcript[:380]
        self.reset_session_state()
        if preserve_recent_summary and preserved_summary:
            self.session_summary = preserved_summary
            self.session_summary_updated_at = self.runtime_timestamp()
            self.session_summary_covered_messages = 0
        if hasattr(self.runtime_state_manager, "sync_active_thread_snapshot"):
            self.runtime_state_manager.sync_active_thread_snapshot()
        else:
            self._apply_thread_snapshot_unlocked(
                getattr(self.runtime_state_manager, "thread_snapshot", dict)(),
            )
        return {
            "mode": "soft",
            "preserved_summary": preserved_summary if preserve_recent_summary else "",
        }
