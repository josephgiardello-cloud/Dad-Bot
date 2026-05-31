from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_input(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows
    return json.loads(text)


def _normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    scenario = str(row.get("scenario") or row.get("scenario_id") or row.get("id") or "").strip()
    if not scenario:
        return None

    response = row.get("response")
    if response is None:
        response = row.get("final_response")
    if response is None:
        response = row.get("output")

    tools = row.get("tools_executed")
    if tools is None:
        tools = row.get("tool_calls")
    if tools is None:
        tools = []

    memory = row.get("memory_accessed")
    if memory is None:
        memory = row.get("memory_reads")
    if memory is None:
        memory = []

    planner_output = row.get("planner_output")
    if not isinstance(planner_output, dict):
        planner_output = None

    return {
        "scenario": scenario,
        "response": str(response or ""),
        "completed": bool(row.get("completed", True)),
        "error": str(row.get("error")) if row.get("error") else None,
        "planner_output": planner_output,
        "tools_executed": [str(v) for v in list(tools or [])],
        "memory_accessed": [str(v) for v in list(memory or [])],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize competitor output into external benchmark schema.",
    )
    parser.add_argument("--input", required=True, help="Path to competitor json/jsonl output.")
    parser.add_argument("--output", required=True, help="Path to write normalized artifact json.")
    parser.add_argument("--agent", required=True, help="Competitor agent label.")
    parser.add_argument("--model", default="", help="Optional model label.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    in_path = Path(str(args.input).strip())
    if not in_path.is_absolute():
        in_path = ROOT / in_path
    out_path = Path(str(args.output).strip())
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}")
        return 2

    try:
        payload = _load_input(in_path)
    except Exception as exc:
        print(f"ERROR: failed to parse input: {type(exc).__name__}: {exc}")
        return 2

    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("responses"), list):
        rows = [item for item in payload.get("responses") if isinstance(item, dict)]
    elif isinstance(payload, list):
        rows = [item for item in payload if isinstance(item, dict)]
    else:
        print("ERROR: input must be a list of rows or an object with 'responses' list")
        return 2

    normalized = []
    for row in rows:
        mapped = _normalize_row(row)
        if mapped is not None:
            normalized.append(mapped)

    out_payload = {
        "agent": str(args.agent),
        "model": str(args.model),
        "responses": normalized,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2, sort_keys=True), encoding="utf-8")

    rel_path = out_path
    try:
        rel_path = out_path.relative_to(ROOT)
    except ValueError:
        rel_path = out_path
    print(f"WROTE_COMPETITOR_ENTRY={str(rel_path).replace('\\', '/')}")
    print(f"ROWS_NORMALIZED={len(normalized)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
