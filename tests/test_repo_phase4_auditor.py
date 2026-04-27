from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from tools import repo_phase4_auditor


def test_repo_phase4_auditor_subchecks_return_expected_shape() -> None:
    structural = repo_phase4_auditor._structural_checks()
    architecture = repo_phase4_auditor._architectural_checks()
    observability = repo_phase4_auditor._observability_checks()
    security = repo_phase4_auditor._security_checks()

    assert "ok" in structural
    assert "missing_files" in structural
    assert "ok" in architecture
    assert "violations" in architecture
    assert "ok" in observability
    assert "trace_levels" in observability
    assert "ok" in security
    assert "receipt_chain_valid" in security


def test_repo_phase4_auditor_main_emits_json_report() -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exit_code = repo_phase4_auditor.main()

    payload = json.loads(buffer.getvalue())
    assert exit_code in {0, 1}
    assert payload["phase4_status"] in {"PASS", "FAIL"}
    assert "coverage" in payload
    assert "critical_gaps" in payload
    assert "risk_score" in payload
