import hashlib
import json

import pytest

from tools import contract_test_compiler


@pytest.mark.parametrize("run_tests", [False, True])
def test_check_mode_does_not_rewrite_manifest(tmp_path, monkeypatch, run_tests):
    root = tmp_path
    contract_path = root / "docs" / "system_contract_map.md"
    output_path = root / "artifacts" / "contract_test_manifest.json"
    contract_path.parent.mkdir(parents=True)
    output_path.parent.mkdir(parents=True)
    contract_text = (
        "Version: test.v1\n\n"
        "### G-1 Sample contract\n\n"
        "Test anchors:\n"
        "- tests/sample.py::test_sample\n"
    )
    contract_path.write_text(contract_text, encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "sample.py").write_text("def test_sample(): pass\n", encoding="utf-8")
    contracts = contract_test_compiler._parse_contracts(contract_text)
    manifest = contract_test_compiler._manifest_payload(
        contract_path.relative_to(root),
        contract_text,
        contracts,
    )
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    monkeypatch.setattr(contract_test_compiler, "_repo_root", lambda: root)
    monkeypatch.setattr(contract_test_compiler, "_run_compiled_tests", lambda *_: (0, ""))
    original_digest = hashlib.sha256(output_path.read_bytes()).hexdigest()

    arguments = ["--contract", str(contract_path), "--output", str(output_path), "--check"]
    if run_tests:
        arguments.append("--run-tests")
    result = contract_test_compiler.main(arguments)

    assert result == 0
    assert hashlib.sha256(output_path.read_bytes()).hexdigest() == original_digest
