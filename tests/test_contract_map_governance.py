from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CONTRACT_MAP = ROOT / "docs" / "system_contract_map.md"
CI_GATE = ROOT / "_ci_gate_check.py"


def test_contract_map_declares_required_update_bundle_rule():
    text = CONTRACT_MAP.read_text(encoding="utf-8")
    assert "### D-1 Required Update Bundle" in text
    assert "must include" in text


def test_contract_map_declares_ci_contract_gate_rule():
    text = CONTRACT_MAP.read_text(encoding="utf-8")
    assert "### D-2 CI Contract Gate" in text
    assert "tools/contract_test_compiler.py" in text


def test_contract_map_declares_documentation_staleness_rule():
    text = CONTRACT_MAP.read_text(encoding="utf-8")
    assert "### D-3 Documentation Staleness Rule" in text
    assert "this map is authoritative" in text


def test_ci_gate_exposes_contract_gate_flag():
    text = CI_GATE.read_text(encoding="utf-8")
    assert "--contract-gate" in text
    assert "--fail-on-untested" in text
