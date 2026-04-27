"""Unit tests for TurnFidelity — full_pipeline property, to_dict."""
from __future__ import annotations

import pytest

from dadbot.core.graph import TurnFidelity


def _fidelity(**kwargs: bool) -> TurnFidelity:
    """Construct TurnFidelity with all flags false by default."""
    defaults = {
        "temporal": False,
        "inference": False,
        "reflection": False,
        "save": False,
    }
    defaults.update(kwargs)
    return TurnFidelity(**defaults)


class TestTurnFidelityFullPipeline:
    def test_all_true_is_full(self):
        f = _fidelity(temporal=True, inference=True, reflection=True, save=True)
        assert f.full_pipeline is True

    def test_all_false_is_not_full(self):
        assert _fidelity().full_pipeline is False

    @pytest.mark.parametrize("missing", [
        "temporal", "inference", "reflection", "save"
    ])
    def test_one_missing_stage_breaks_full(self, missing):
        all_flags = dict(temporal=True, inference=True, reflection=True, save=True)
        all_flags[missing] = False
        f = TurnFidelity(**all_flags)
        assert f.full_pipeline is False

    def test_partial_run_not_full(self):
        f = _fidelity(temporal=True)
        assert f.full_pipeline is False


class TestTurnFidelityToDict:
    def test_to_dict_contains_all_stage_flags(self):
        f = _fidelity(temporal=True, inference=True)
        d = f.to_dict()
        for stage in ("temporal", "inference", "reflection", "save"):
            assert stage in d

    def test_to_dict_bool_values(self):
        f = _fidelity(temporal=True)
        d = f.to_dict()
        assert d["temporal"] is True
        assert d["inference"] is False

    def test_to_dict_full_pipeline_key_present(self):
        f = _fidelity()
        d = f.to_dict()
        assert "full_pipeline" in d

    def test_to_dict_full_pipeline_reflects_state(self):
        all_true = _fidelity(temporal=True, inference=True, reflection=True, save=True)
        assert all_true.to_dict()["full_pipeline"] is True
        none_true = _fidelity()
        assert none_true.to_dict()["full_pipeline"] is False


class TestTurnFidelityMutability:
    def test_fidelity_is_mutable(self):
        f = _fidelity(temporal=True)
        f.temporal = False
        assert f.temporal is False

    def test_equality(self):
        a = _fidelity(temporal=True, save=True)
        b = _fidelity(temporal=True, save=True)
        assert a == b

    def test_inequality_on_flag_difference(self):
        a = _fidelity(temporal=True)
        b = _fidelity(temporal=False)
        assert a != b


class TestTurnFidelityHarnessFactory:
    """Smoke-test that TurnFactory produces sane fidelity defaults."""

    def test_default_fidelity_all_false(self):
        """TurnContext created without running graph has no stage flags set."""
        from harness.turn_factory import TurnFactory
        ctx = TurnFactory().build_turn(seed=42)
        assert ctx.fidelity.full_pipeline is False
        assert ctx.fidelity.temporal is False
