"""Compile system contract map test anchors into an executable test manifest.

This is a lightweight contract-to-test compiler layer:
- Parses contract IDs and test anchors from docs/system_contract_map.md.
- Compiles a deterministic manifest of contract -> pytest nodeids.
- Optionally validates nodeids with pytest --collect-only.
- Optionally executes compiled nodeids.

Usage examples:
  python tools/contract_test_compiler.py
  python tools/contract_test_compiler.py --validate-nodeids
  python tools/contract_test_compiler.py --run-tests
  python tools/contract_test_compiler.py --check
  python tools/contract_test_compiler.py --json

Exit codes:
  0 - success
  1 - compile/validation/test failure
  2 - contract map missing or unreadable
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_CONTRACT_MAP = Path("docs/system_contract_map.md")
DEFAULT_OUTPUT = Path("artifacts/contract_test_manifest.json")

CONTRACT_HEADER_RE = re.compile(r"^###\s+([A-Z]-\d+)\s+(.+?)\s*$")
VERSION_RE = re.compile(r"^Version:\s*(.+?)\s*$")
ANCHOR_LINE_RE = re.compile(r"^\s*-\s+(tests/.+?)\s*$")


@dataclass
class ContractEntry:
    contract_id: str
    title: str
    anchors: list[str]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile contract map test anchors")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Fail if output manifest differs from compiled result")
    parser.add_argument("--validate-nodeids", action="store_true", help="Validate compiled nodeids via pytest --collect-only")
    parser.add_argument("--run-tests", action="store_true", help="Execute compiled nodeids with pytest")
    parser.add_argument("--fail-on-untested", action="store_true", help="Fail if any contract has no test anchors")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser.parse_args(argv)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _contract_version(lines: list[str]) -> str:
    for line in lines:
        m = VERSION_RE.match(line.strip())
        if m:
            return str(m.group(1)).strip()
    return "unknown"


def _parse_contracts(markdown: str) -> list[ContractEntry]:
    lines = markdown.splitlines()
    contracts: list[ContractEntry] = []

    current_id: str | None = None
    current_title: str | None = None
    current_anchors: list[str] = []
    in_test_anchor_block = False

    def flush_current() -> None:
        nonlocal current_id, current_title, current_anchors, in_test_anchor_block
        if current_id is None or current_title is None:
            return
        contracts.append(
            ContractEntry(
                contract_id=current_id,
                title=current_title,
                anchors=sorted(set(current_anchors)),
            ),
        )
        current_id = None
        current_title = None
        current_anchors = []
        in_test_anchor_block = False

    for raw in lines:
        line = raw.rstrip("\n")

        header_match = CONTRACT_HEADER_RE.match(line)
        if header_match:
            flush_current()
            current_id = str(header_match.group(1)).strip()
            current_title = str(header_match.group(2)).strip()
            continue

        if current_id is None:
            continue

        normalized = line.strip()
        if normalized == "Test anchors:":
            in_test_anchor_block = True
            continue

        if in_test_anchor_block:
            anchor_match = ANCHOR_LINE_RE.match(line)
            if anchor_match:
                current_anchors.append(str(anchor_match.group(1)).strip())
                continue

            # End of anchor block once a non-bullet, non-empty line appears.
            if normalized and not normalized.startswith("-"):
                in_test_anchor_block = False

    flush_current()

    seen: set[str] = set()
    for contract in contracts:
        if contract.contract_id in seen:
            raise ValueError(f"Duplicate contract id: {contract.contract_id}")
        seen.add(contract.contract_id)

    if not contracts:
        raise ValueError("No contract entries found in contract map")

    return contracts


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _python_executable(root: Path) -> str:
    venv_python = root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    completed = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    output = (completed.stdout or "") + (completed.stderr or "")
    return int(completed.returncode), output


def _validate_nodeids(root: Path, nodeids: list[str]) -> dict[str, Any]:
    py = _python_executable(root)
    missing: list[str] = []
    details: dict[str, str] = {}

    for nodeid in nodeids:
        cmd = [py, "-m", "pytest", "--collect-only", "-q", nodeid]
        rc, output = _run(cmd, cwd=root)
        lowered = output.lower()
        if rc != 0 or "not found" in lowered or "no tests collected" in lowered:
            missing.append(nodeid)
            details[nodeid] = output.strip()

    return {
        "ok": len(missing) == 0,
        "missing_nodeids": missing,
        "details": details,
    }


def _run_compiled_tests(root: Path, nodeids: list[str]) -> tuple[int, str]:
    py = _python_executable(root)
    cmd = [py, "-m", "pytest", "-q", *nodeids]
    return _run(cmd, cwd=root)


def _manifest_payload(contract_path: Path, contract_text: str, contracts: list[ContractEntry]) -> dict[str, Any]:
    contract_lines = contract_text.splitlines()
    version = _contract_version(contract_lines)

    all_nodeids: list[str] = []
    for item in contracts:
        all_nodeids.extend(item.anchors)

    unique_nodeids = sorted(set(all_nodeids))
    untested_contracts = [item.contract_id for item in contracts if not item.anchors]

    return {
        "schema_version": "contract-test-manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "contract_map": {
            "path": contract_path.as_posix(),
            "version": version,
            "sha256": _sha256_text(contract_text),
        },
        "contracts": [
            {
                "id": item.contract_id,
                "title": item.title,
                "test_anchors": item.anchors,
                "anchor_count": len(item.anchors),
            }
            for item in contracts
        ],
        "compiled": {
            "nodeids": unique_nodeids,
            "nodeid_count": len(unique_nodeids),
            "untested_contract_ids": untested_contracts,
            "untested_contract_count": len(untested_contracts),
        },
    }


def _normalized_manifest_for_check(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize manifest for deterministic equivalence checks.

    Ignore volatile fields and optional runtime-only sections.
    """
    return {
        "schema_version": payload.get("schema_version"),
        "contract_map": payload.get("contract_map"),
        "contracts": payload.get("contracts"),
        "compiled": payload.get("compiled"),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = _repo_root()
    contract_path = args.contract if args.contract.is_absolute() else (root / args.contract)
    output_path = args.output if args.output.is_absolute() else (root / args.output)

    if not contract_path.exists():
        print(f"[contract-compiler] ERROR: contract map not found: {contract_path}", file=sys.stderr)
        return 2

    try:
        contract_text = _read_text(contract_path)
        contracts = _parse_contracts(contract_text)
    except Exception as exc:
        print(f"[contract-compiler] ERROR: failed to compile contract map: {exc}", file=sys.stderr)
        return 1

    manifest = _manifest_payload(contract_path.relative_to(root), contract_text, contracts)

    if args.fail_on_untested and int(manifest["compiled"]["untested_contract_count"]) > 0:
        print(
            "[contract-compiler] ERROR: untested contracts found: "
            f"{manifest['compiled']['untested_contract_ids']}",
            file=sys.stderr,
        )
        return 1

    if args.check:
        if not output_path.exists():
            print(f"[contract-compiler] ERROR: manifest missing for --check: {output_path}", file=sys.stderr)
            return 1
        existing = json.loads(_read_text(output_path))
        existing_norm = _normalized_manifest_for_check(existing)
        current_norm = _normalized_manifest_for_check(manifest)
        if existing_norm != current_norm:
            print("[contract-compiler] ERROR: manifest drift detected. Rebuild manifest.", file=sys.stderr)
            return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    nodeids = list(manifest["compiled"]["nodeids"])

    if args.validate_nodeids:
        validation = _validate_nodeids(root, nodeids)
        manifest["validation"] = validation
        output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if not bool(validation.get("ok")):
            print(
                "[contract-compiler] ERROR: one or more test anchors are not collectable:",
                file=sys.stderr,
            )
            for bad in validation.get("missing_nodeids", []):
                print(f"  - {bad}", file=sys.stderr)
            return 1

    if args.run_tests:
        rc, output = _run_compiled_tests(root, nodeids)
        manifest["last_test_run"] = {
            "return_code": rc,
            "nodeid_count": len(nodeids),
        }
        output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if rc != 0:
            print(output)
            print("[contract-compiler] ERROR: compiled contract tests failed", file=sys.stderr)
            return 1

    if args.json_output:
        print(json.dumps(manifest, indent=2))
    else:
        print(
            "[contract-compiler] compiled "
            f"contracts={len(contracts)} nodeids={manifest['compiled']['nodeid_count']} "
            f"untested={manifest['compiled']['untested_contract_count']}",
        )
        print(f"[contract-compiler] manifest -> {output_path.relative_to(root).as_posix()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
