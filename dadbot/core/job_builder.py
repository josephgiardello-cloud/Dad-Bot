"""dadbot.core.job_builder — TurnContext construction authority.

Builds a :class:`~dadbot.core.graph.TurnContext` from an
:class:`~dadbot.core.control_plane.ExecutionJob`.  No policy logic, no graph
references, no session state — pure data assembly.
"""

from __future__ import annotations

from typing import Any

from dadbot.contracts import AttachmentList
from dadbot.core.graph import TurnContext


class JobBuilder:
    """Assembles a :class:`TurnContext` from job/session inputs.

    Authority contract
    ------------------
    - Receives raw job fields (user_input, attachments, session_id, metadata).
    - Returns a fully initialised :class:`TurnContext` with metadata bound.
    - Zero policy logic, zero graph imports beyond :class:`TurnContext`.
    """

    def build(
        self,
        *,
        user_input: str,
        attachments: AttachmentList | None,
        session_id: str,
        metadata: dict[str, Any] | None,
    ) -> TurnContext:
        """Build and return a new :class:`TurnContext` for *job*.

        Parameters
        ----------
        user_input:
            Raw user message.
        attachments:
            Optional list of attachments.
        session_id:
            Session identifier; defaults to ``"default"`` if empty.
        metadata:
            Caller-supplied metadata dict; ``trace_id`` is promoted to the
            context's ``trace_id`` field when present.

        """
        md: dict[str, Any] = dict(metadata or {})
        trace_id = str(md.get("trace_id") or "")
        context_kwargs: dict[str, Any] = {
            "user_input": user_input,
            "attachments": attachments,
            "metadata": {"session_id": str(session_id or "default"), **md},
        }
        if trace_id:
            context_kwargs["trace_id"] = trace_id
        return TurnContext(**context_kwargs)
