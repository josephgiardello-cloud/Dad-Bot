from __future__ import annotations

from pathlib import Path

from tools import architectural_shallow_audit as audit


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_detects_string_dispatch_and_fake_compile_pattern(tmp_path: Path) -> None:
    root = tmp_path
    code = """
from typing import Any

def compile_safety(service: Any, candidate: Any) -> Any:
    steps = ["binary", "unary"]
    for step in steps:
        if step == "binary":
            return candidate
        if step == "unary":
            return candidate
    return candidate
""".strip()
    file_path = root / "dadbot" / "core" / "policy_compiler.py"
    _write(file_path, code)

    result = audit.analyze_file(file_path, root)

    assert result is not None
    assert result.score >= 6
    assert "string_dispatch" in result.flags
    assert "fake_compile_pattern" in result.flags


def test_audit_repo_ranks_files_by_score(tmp_path: Path) -> None:
    root = tmp_path
    low_risk = """

def stable() -> int:
    return 1
""".strip()
    high_risk = """
from typing import Any

def build_plan(x: Any):
    items = ["a", "b"]
    for i in items:
        if i == "a":
            return x
        if i == "b":
            return x
    return x
""".strip()

    _write(root / "dadbot" / "core" / "safe.py", low_risk)
    _write(root / "dadbot" / "core" / "risky.py", high_risk)

    report = audit.audit_repo(root, min_score=1, top_n=10)

    ranked = list(report.get("ranked") or [])
    assert ranked
    assert ranked[0]["file"].endswith("risky.py")
    assert int(ranked[0]["score"] or 0) >= int(ranked[-1]["score"] or 0)


def test_skips_tests_directory_by_default(tmp_path: Path) -> None:
    root = tmp_path
    risky_test = """
from typing import Any

def plan_dispatch(x: Any):
    for kind in ["a"]:
        if kind == "a":
            return x
""".strip()
    _write(root / "tests" / "test_risky.py", risky_test)

    report_default = audit.audit_repo(root, min_score=1, top_n=10, include_tests=False)
    report_with_tests = audit.audit_repo(root, min_score=1, top_n=10, include_tests=True)

    default_files = {item["file"] for item in list(report_default.get("ranked") or [])}
    with_tests_files = {item["file"] for item in list(report_with_tests.get("ranked") or [])}

    assert "tests/test_risky.py" not in default_files
    assert "tests/test_risky.py" in with_tests_files


def test_name_reality_mismatch_is_flagged_for_arch_claims(tmp_path: Path) -> None:
    root = tmp_path
    claimed = """
from typing import Any

class PolicyCompiler:
    pass

def compile_policy(x: Any) -> Any:
    for mode in ["fast", "safe"]:
        if mode == "fast":
            return x
        if mode == "safe":
            return x
    return x
""".strip()
    file_path = root / "dadbot" / "core" / "policy_compiler.py"
    _write(file_path, claimed)

    result = audit.analyze_file(file_path, root)

    assert result is not None
    assert "name_reality_mismatch" in result.flags
    name_reality = dict(result.details.get("name_reality") or {})
    assert bool(name_reality.get("mismatch")) is True
    assert "compiler" in list(name_reality.get("claims") or [])
    assert int(name_reality.get("mismatch_score") or 0) >= 2


def test_manual_review_queue_contains_name_reality_candidates(tmp_path: Path) -> None:
    root = tmp_path
    risky = """
from typing import Any

def build_planner(x: Any):
    for kind in ["a", "b"]:
        if kind == "a":
            return x
        if kind == "b":
            return x
    return x
""".strip()
    _write(root / "dadbot" / "core" / "task_planner.py", risky)

    report = audit.audit_repo(root, min_score=1, top_n=10)
    queue = list(report.get("manual_review_queue") or [])

    assert queue
    assert any(str(item.get("file") or "").endswith("task_planner.py") for item in queue)
