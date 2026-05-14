"""dadbot/core/equivalence_validator.py — Live convergence validation wrapper.

Intercepts turn execution to validate hard equivalence between thin-spine
and legacy paths during actual runtime.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Mapping

from dadbot.contracts import AttachmentList, FinalizedTurnResult
from dadbot.core.behavior_contract_lock import get_behavior_contract_lock

logger = logging.getLogger(__name__)


def _result_text(result: FinalizedTurnResult | None) -> str:
    if result is None:
        return ""
    if isinstance(result, tuple) and len(result) >= 1:
        return str(result[0] or "").strip()
    return ""


class EquivalenceValidator:
    """Wraps turn submission to validate path equivalence.
    
    In "dual-run" mode: executes both thin-spine and legacy path, compares outputs.
    In "verify-only" mode: executes one path, validates against captured reference.
    """

    def __init__(self, *, mode: str = "verify-only"):
        """
        Args:
            mode: "verify-only" (execute one path, compare to reference),
                  "dual-run" (execute both, compare in real-time)
        """
        self.mode = mode
        self.contract_lock = get_behavior_contract_lock(execution_mode="strict")
        self._reference_outputs: dict[str, FinalizedTurnResult] = {}

    async def validate_turn_equivalence(
        self,
        *,
        user_input: str,
        attachments: AttachmentList | None = None,
        thin_spine_executor: Callable | None = None,
        legacy_executor: Callable | None = None,
        session_id: str = "default",
    ) -> FinalizedTurnResult:
        """Execute and validate turn with equivalence checking.
        
        Args:
            user_input: Raw user message
            attachments: Attachments (if any)
            thin_spine_executor: Callable that executes thin-spine path
            legacy_executor: Callable that executes legacy path
            session_id: Session identifier for tracking
            
        Returns:
            FinalizedTurnResult from primary executor
            
        Raises:
            AssertionError: If contract invariants violated (strict mode)
        """
        input_hash = self._hash_input(user_input, session_id)

        if thin_spine_executor is None:
            raise RuntimeError("Equivalence validation requires thin_spine_executor")

        if self.mode == "dual-run":
            if legacy_executor is None:
                raise RuntimeError("Dual-run equivalence validation requires legacy_executor")
            return await self._validate_dual_run(
                input_hash=input_hash,
                user_input=user_input,
                attachments=attachments,
                thin_spine_executor=thin_spine_executor,
                legacy_executor=legacy_executor,
                session_id=session_id,
            )
        else:  # verify-only
            return await self._validate_verify_only(
                input_hash=input_hash,
                user_input=user_input,
                attachments=attachments,
                thin_spine_executor=thin_spine_executor,
                session_id=session_id,
            )

    async def _validate_dual_run(
        self,
        *,
        input_hash: str,
        user_input: str,
        attachments: AttachmentList | None,
        thin_spine_executor: Callable,
        legacy_executor: Callable,
        session_id: str,
    ) -> FinalizedTurnResult:
        """Execute both paths, validate equivalence, return thin-spine result."""
        logger.info(
            "DUAL_RUN validation starting [input_hash=%s, session=%s]",
            input_hash,
            session_id,
        )

        # Execute both paths (sequentially for determinism)
        thin_result = await thin_spine_executor(
            user_input,
            attachments=attachments,
            session_id=session_id,
        )
        legacy_result = await legacy_executor(
            user_input,
            attachments=attachments,
            session_id=session_id,
        )

        # Validate semantic output equivalence
        thin_text = _result_text(thin_result)
        legacy_text = _result_text(legacy_result)

        self.contract_lock.lock_semantic_output(
            input_hash=input_hash,
            legacy_output=legacy_text,
            thin_spine_output=thin_text,
        )

        # TODO: Validate tool call sequence
        # TODO: Validate state mutation graph

        logger.info(
            "DUAL_RUN validation passed [input_hash=%s, equivalence=OK]",
            input_hash,
        )

        return thin_result  # Return thin-spine result

    async def _validate_verify_only(
        self,
        *,
        input_hash: str,
        user_input: str,
        attachments: AttachmentList | None,
        thin_spine_executor: Callable,
        session_id: str,
    ) -> FinalizedTurnResult:
        """Execute primary path, validate against cached reference."""
        logger.debug(
            "VERIFY_ONLY validation [input_hash=%s, session=%s]",
            input_hash,
            session_id,
        )

        result = await thin_spine_executor(
            user_input,
            attachments=attachments,
            session_id=session_id,
        )

        # Check if we have a reference for this input
        if input_hash in self._reference_outputs:
            reference = self._reference_outputs[input_hash]
            result_text = _result_text(result)
            ref_text = _result_text(reference)

            self.contract_lock.lock_semantic_output(
                input_hash=input_hash,
                legacy_output=ref_text,
                thin_spine_output=result_text,
            )

        return result

    def record_reference(self, *, input_hash: str, result: FinalizedTurnResult) -> None:
        """Record reference output for future comparisons."""
        self._reference_outputs[input_hash] = result

    @staticmethod
    def _hash_input(user_input: str, session_id: str) -> str:
        """Create deterministic input hash for equivalence tracking."""
        normalized = f"{session_id}::{user_input}".strip()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get_validation_status(self) -> dict[str, Any]:
        """Return current validation status."""
        return {
            "mode": self.mode,
            "contract_status": self.contract_lock.get_contract_status(),
            "cached_references": len(self._reference_outputs),
        }

    def reset_references(self) -> None:
        """Clear cached reference outputs."""
        self._reference_outputs.clear()


__all__ = ["EquivalenceValidator"]
