from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from dadbot.contracts import AttachmentList, FinalizedTurnResult


def thin_turn_handler_enabled() -> bool:
    # Canonical default is enabled; alternate paths are no longer runtime-selectable.
    value = str(os.environ.get("DADBOT_USE_THIN_TURN_HANDLER", "1") or "").strip().lower()
    return value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class TurnContext:
    user_input: str
    session_id: str
    attachments: AttachmentList | None = None
    confluence_key: str | None = None
    metadata: Mapping[str, Any] | None = None
    timeout_seconds: float | None = None


class TurnHandler:
    """Thin, deterministic turn spine adapter.

    This wrapper does not decide policy or mutate global state; it only normalizes
    submit payload shape and delegates into the authoritative execution entrypoint.
    """

    def __init__(
        self,
        *,
        submit_turn: Callable[..., Awaitable[FinalizedTurnResult]],
    ) -> None:
        self._submit_turn = submit_turn

    async def process_turn(self, ctx: TurnContext) -> FinalizedTurnResult:
        outbound_metadata: dict[str, Any] = {
            "confluence_mode": "enforce",
            "confluence_key": str(ctx.confluence_key or "").strip(),
        }
        if ctx.metadata:
            outbound_metadata.update(dict(ctx.metadata))

        return await self._submit_turn(
            str(ctx.user_input or ""),
            attachments=ctx.attachments,
            session_id=str(ctx.session_id or "default"),
            confluence_key=str(ctx.confluence_key or ""),
            metadata=outbound_metadata,
            timeout_seconds=ctx.timeout_seconds,
        )
