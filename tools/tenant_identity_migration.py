"""Read-only dry-run inventory for legacy tenant identity migration."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from dadbot_system.tenant_identity import parse_tenant_id

_NORMALIZER = re.compile(r"[^a-zA-Z0-9._-]+")
_TENANT_KEYS = {"tenant", "tenant_id", "tenantid"}
_JSON_COLUMNS = {"payload", "state", "data", "metadata"}


@dataclass(frozen=True)
class TenantObservation:
    value: str | None
    source: str


def legacy_normalized_form(value: str | None) -> str:
    normalized = _NORMALIZER.sub("-", str(value or "").strip().lower()).strip("-._")
    return normalized or "default"


def _literal(node: ast.AST) -> str | object | None:
    if isinstance(node, ast.Constant) and (node.value is None or isinstance(node.value, str)):
        return node.value
    return _UNSET


_UNSET = object()


def _source_observations(path: Path, root: Path) -> list[TenantObservation]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        return []
    observations: list[TenantObservation] = []
    seen: set[tuple[str | None, str]] = set()
    for node in ast.walk(tree):
        candidates: list[ast.AST] = []
        if isinstance(node, ast.Call):
            candidates.extend(keyword.value for keyword in node.keywords if keyword.arg == "tenant_id")
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=False):
                if isinstance(key, ast.Constant) and key.value in _TENANT_KEYS:
                    candidates.append(value)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "DADBOT_TENANT_ID"
                ):
                    candidates.append(node.value)
        for candidate in candidates:
            value = _literal(candidate)
            if value is _UNSET:
                continue
            source = f"{path.relative_to(root)}:{getattr(candidate, 'lineno', 1)}"
            item = (value, source)
            if item not in seen:
                observations.append(TenantObservation(value, source))
                seen.add(item)
    if path.name == "contracts.py":
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                names = {target.id for target in node.targets if isinstance(target, ast.Name)}
                if "DEFAULT_TENANT_ID" in names:
                    value = _literal(node.value)
                    if isinstance(value, str):
                        observations.append(
                            TenantObservation(value, f"{path.relative_to(root)}:{node.lineno}")
                        )
    return observations


def _json_observations(value: object, source: str) -> list[TenantObservation]:
    found: list[TenantObservation] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in _TENANT_KEYS and isinstance(item, str):
                found.append(TenantObservation(item, source))
            found.extend(_json_observations(item, source))
    elif isinstance(value, list):
        for item in value:
            found.extend(_json_observations(item, source))
    return found


def _json_file_observations(path: Path, root: Path) -> list[TenantObservation]:
    found: list[TenantObservation] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return found
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        found.extend(_json_observations(value, f"{path.relative_to(root)}:{line_number}"))
    return found


def _sqlite_observations(path: Path, root: Path) -> list[TenantObservation]:
    found: list[TenantObservation] = []
    uri = f"file:{quote(str(path.resolve()))}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return found
    try:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        for (table,) in tables:
            columns = connection.execute(f'PRAGMA table_info("{table.replace(chr(34), chr(34) * 2)}")').fetchall()
            tenant_columns = [column[1] for column in columns if str(column[1]).lower() in _TENANT_KEYS]
            json_columns = [
                column[1] for column in columns if str(column[1]).lower() in _JSON_COLUMNS
            ]
            for column in tenant_columns:
                quoted_table = '"' + str(table).replace('"', '""') + '"'
                quoted_column = '"' + str(column).replace('"', '""') + '"'
                rows = connection.execute(
                    f"SELECT DISTINCT {quoted_column} FROM {quoted_table} WHERE {quoted_column} IS NOT NULL"
                ).fetchall()
                found.extend(
                    TenantObservation(str(row[0]), f"{path.relative_to(root)}:{table}.{column}")
                    for row in rows
                )
            for column in json_columns:
                quoted_table = '"' + str(table).replace('"', '""') + '"'
                quoted_column = '"' + str(column).replace('"', '""') + '"'
                rows = connection.execute(
                    f"SELECT {quoted_column} FROM {quoted_table} WHERE {quoted_column} IS NOT NULL"
                ).fetchall()
                for row_index, (raw,) in enumerate(rows, start=1):
                    try:
                        value = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
                    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
                        continue
                    found.extend(
                        _json_observations(value, f"{path.relative_to(root)}:{table}.{column}[{row_index}]")
                    )
    except sqlite3.Error:
        return found
    finally:
        connection.close()
    return found


def collect_tenant_observations(root: Path) -> list[TenantObservation]:
    observations: list[TenantObservation] = []
    for source_root in (root / "dadbot", root / "dadbot_system", root / "tests"):
        if source_root.exists():
            for path in source_root.rglob("*.py"):
                observations.extend(_source_observations(path, root))

    for path in root.glob(".env*"):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for line_number, line in enumerate(lines, start=1):
            key, separator, value = line.partition("=")
            if separator and key.strip() == "DADBOT_TENANT_ID":
                observations.append(TenantObservation(value.strip().strip("\"'"), f"{path.name}:{line_number}"))

    state_roots = [root / "runtime", root / "session_logs"]
    state_files = [
        root / name
        for name in (
            "dad_memory.json",
            "dad_memory.json.bak",
            "dad_profile.json",
            "dad_memory_graph.sqlite3",
            "dad_memory_semantic.sqlite3",
            "dad_memory.citation-approvals.sqlite3",
        )
    ]
    for state_root in state_roots:
        if state_root.exists():
            state_files.extend(path for path in state_root.rglob("*") if path.is_file())
    for path in state_files:
        if not path.is_file():
            continue
        if path.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
            observations.extend(_json_file_observations(path, root))
        elif path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}:
            observations.extend(_sqlite_observations(path, root))
    return observations


def render_dry_run(observations: list[TenantObservation]) -> str:
    grouped: dict[str, list[TenantObservation]] = defaultdict(list)
    for observation in observations:
        grouped[legacy_normalized_form(observation.value)].append(observation)

    lines = [
        "# Tenant identity migration inventory — dry run",
        "",
        "Status: **DRY RUN ONLY — no tenant IDs, aliases, memberships, tokens, or state were changed.**",
        "",
        "This inventory uses the prior lossy normalizer only to identify legacy collision groups. "
        "It does not issue identifiers or resolve ownership.",
        "",
        f"Observed tenant-bearing values: {len(observations)}; normalized forms: {len(grouped)}.",
        "",
        "| Legacy normalized form | Observed exact inputs | Proposed disposition | Sources |",
        "|---|---|---|---|",
    ]
    for normalized, entries in sorted(grouped.items()):
        raw_inputs = sorted({"<missing>" if item.value is None else repr(item.value) for item in entries})
        collision = len(raw_inputs) > 1
        try:
            parse_tenant_id(normalized)
            retained = True
        except ValueError:
            retained = False
        if collision:
            disposition = "UNRESOLVED COLLISION; require operator mapping to distinct issued IDs"
        elif retained:
            disposition = f"Candidate retained legacy ID `{normalized}`; explicit sign-off required"
        else:
            disposition = "Provision a new opaque ID; exact alias mapping requires sign-off"
        sources = sorted({entry.source for entry in entries})
        sources_text = "<br>".join(sources)
        lines.append(
            f"| `{normalized}` | {', '.join(raw_inputs)} | {disposition} | {sources_text} |"
        )
    lines.extend(
        [
            "",
            "## Collision groups",
            "",
        ]
    )
    collisions = [
        (normalized, entries)
        for normalized, entries in sorted(grouped.items())
        if len({"<missing>" if item.value is None else repr(item.value) for item in entries}) > 1
    ]
    if collisions:
        for normalized, entries in collisions:
            inputs = sorted({"<missing>" if item.value is None else repr(item.value) for item in entries})
            lines.append(f"- `{normalized}`: {', '.join(inputs)}")
    else:
        lines.append("- None detected in scanned sources.")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "Scanned Python tenant-bearing literals in `dadbot/`, `dadbot_system/`, and `tests/`; "
            "the `DADBOT_TENANT_ID` key in root `.env*` files; tenant keys in JSON/JSONL under "
            "`runtime/` and `session_logs/`; and tenant columns/JSON payloads in the known local "
            "SQLite state files. Secrets and unrelated JSON properties are not emitted.",
            "",
            "Unresolved collisions must not be merged automatically. Issue opaque IDs through the "
            "provisioning boundary and add exact alias records only after an operator identifies "
            "the owner of each legacy value. The literal `default` is listed as a candidate legacy "
            "identity, never as a missing-value fallback.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    report = render_dry_run(collect_tenant_observations(root))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Dry-run report written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
