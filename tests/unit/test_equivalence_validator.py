from __future__ import annotations

import pytest

from dadbot.core.equivalence_validator import EquivalenceValidator

pytestmark = [pytest.mark.dev, pytest.mark.asyncio]


async def _thin_ok(_user_input: str, **_kwargs):
    return ("same output", False)


async def _legacy_ok(_user_input: str, **_kwargs):
    return ("same output", False)


async def _legacy_mismatch(_user_input: str, **_kwargs):
    return ("different output", False)


async def test_dual_run_equivalence_passes_when_outputs_match() -> None:
    validator = EquivalenceValidator(mode="dual-run")
    validator.contract_lock.reset()

    result = await validator.validate_turn_equivalence(
        user_input="hello",
        thin_spine_executor=_thin_ok,
        legacy_executor=_legacy_ok,
        session_id="s1",
    )

    assert result[0] == "same output"
    status = validator.get_validation_status()
    assert status["contract_status"]["violations"] == 0


async def test_dual_run_equivalence_fails_when_outputs_mismatch() -> None:
    validator = EquivalenceValidator(mode="dual-run")
    validator.contract_lock.reset()

    with pytest.raises(AssertionError, match="SEMANTIC_OUTPUT"):
        await validator.validate_turn_equivalence(
            user_input="hello",
            thin_spine_executor=_thin_ok,
            legacy_executor=_legacy_mismatch,
            session_id="s1",
        )


async def test_dual_run_requires_legacy_executor() -> None:
    validator = EquivalenceValidator(mode="dual-run")
    validator.contract_lock.reset()

    with pytest.raises(RuntimeError, match="requires legacy_executor"):
        await validator.validate_turn_equivalence(
            user_input="hello",
            thin_spine_executor=_thin_ok,
            session_id="s1",
        )
