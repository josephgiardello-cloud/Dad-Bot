from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "external_benchmark"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.scenario_suite import SCENARIOS, Scenario
from tests.scoring_engine import ScoringEngine, aggregate_capability_profile
from tests.trace_schema import NormalizedTrace


@dataclass
class ArtifactSource:
    name: str
    path: Path


@dataclass
class ScenarioResponse:
    scenario: str
    response: str
    completed: bool
    error: str | None
    planner_output: dict[str, Any] | None
    tools_executed: list[str]
    memory_accessed: list[str]
    raw_state: dict[str, Any]


@dataclass
class AgentScorecard:
    name: str
    source_path: str
    scenario_scores: list[dict[str, Any]]
    category_profile: dict[str, float]
    overall_average: float
    pass_rate: float
    scorer_mode: str
    external_metrics: dict[str, float]
    internal_sim_overall_average: float
    internal_sim_pass_rate: float
    scenario_count_expected: int
    scenario_count_scored: int
    missing_scenarios: list[str]


class ExternalBenchmarkError(RuntimeError):
    """Raised when benchmark input artifacts are invalid."""


class ExternalBenchmarkRunner:
    def __init__(self, cert_mode: bool = False) -> None:
        self._scoring = ScoringEngine()
        self._scenario_map: dict[str, Scenario] = {s.name: s for s in SCENARIOS}
        self._cert_mode = cert_mode

    def run(self, sources: list[ArtifactSource]) -> dict[str, Any]:
        if not sources:
            raise ExternalBenchmarkError("No benchmark artifacts provided.")

        run_started = utc_now_iso()
        entrants: list[AgentScorecard] = []

        input_hashes: dict[str, str] = {}
        entrant_roster: list[str] = []
        for source in sources:
            payload = _load_json(source.path)
            input_hashes[source.name] = file_sha256(source.path)
            responses = _parse_responses(payload, cert_mode=self._cert_mode)
            entrants.append(self._score_agent(source, responses))
            entrant_roster.append(source.name)

        entrants.sort(key=lambda e: e.overall_average, reverse=True)
        winner = entrants[0].name if entrants else ""

        # Compute entrant roster hash for cohort lock.
        entrant_roster_hash = hashlib.sha256(":".join(sorted(entrant_roster)).encode()).hexdigest()
        
        comparative = {
            "generated_at": run_started,
            "scenario_count": len(SCENARIOS),
            "scoring_mode": "external-aligned-cert-v1" if self._cert_mode else "internal-sim-v1",
            "entrants": [asdict(e) for e in entrants],
            "ranking": [
                {
                    "rank": idx + 1,
                    "name": entrant.name,
                    "overall_average": round(entrant.overall_average, 4),
                    "pass_rate": round(entrant.pass_rate, 4),
                }
                for idx, entrant in enumerate(entrants)
            ],
            "winner": winner,
            "input_hashes": input_hashes,
            "scenarios_sha256": scenario_suite_sha256(),
            "script_sha256": file_sha256(Path(__file__)),
            "entrant_roster": entrant_roster,
            "entrant_roster_hash": entrant_roster_hash,
            "cert_mode": self._cert_mode,
        }
        return comparative

    def _score_agent(self, source: ArtifactSource, responses: dict[str, ScenarioResponse]) -> AgentScorecard:
        cap_scores: list[Any] = []
        scenario_scores: list[dict[str, Any]] = []
        missing: list[str] = []
        osworld_completed_count = 0
        strict_passes = 0
        tool_required_total = 0
        tool_required_strict_success = 0

        for scenario in SCENARIOS:
            response = responses.get(scenario.name)
            if response is None:
                missing.append(scenario.name)
                response = ScenarioResponse(
                    scenario=scenario.name,
                    response="",
                    completed=False,
                    error="missing_scenario_response",
                    planner_output=None,
                    tools_executed=[],
                    memory_accessed=[],
                    raw_state={},
                )

            response_ok = bool(response.completed) and not bool(response.error)
            if response_ok:
                osworld_completed_count += 1

            normalized = NormalizedTrace.from_mock(
                scenario_name=scenario.name,
                category=scenario.category,
                input_text=scenario.input_text,
                final_response=response.response,
                completed=response.completed,
                error=response.error,
                planner_output=response.planner_output,
                tools_executed=response.tools_executed,
                memory_accessed=response.memory_accessed,
                raw_state=response.raw_state,
            )
            # Artifact-backed entries must be scored as real traces, not synthetic mock.
            normalized.execution_mode = "artifact"
            cap = self._scoring.score(normalized, scenario)
            cap_scores.append(cap)
            scenario_scores.append(cap.to_dict())

        category_profile = aggregate_capability_profile(cap_scores)
        overall_average = _mean([c.overall for c in cap_scores])

        passes = 0
        for cap in cap_scores:
            spec = self._scenario_map[cap.scenario_name].behavioral_spec
            threshold = float(spec.get("quality_threshold") or 0.0)
            if cap.overall >= threshold:
                passes += 1

        # Internal simulator metrics (diagnostic only).
        internal_sim_pass_rate = passes / float(len(cap_scores) or 1)

        # External-aligned certification metrics (primary in cert mode).
        # Strict scenario pass: threshold met + completed + no error + required tool constraints.
        for scenario, cap in zip(SCENARIOS, cap_scores):
            response = responses.get(scenario.name)
            response_ok = bool(response and response.completed and not response.error)
            has_response_text = bool(response and str(response.response or "").strip())
            spec = scenario.behavioral_spec or {}
            threshold = float(spec.get("quality_threshold") or 0.0)
            expects_tool = bool(spec.get("expected_tool_use"))
            min_tool_calls = int(spec.get("min_tool_calls") or (1 if expects_tool else 0))
            tool_count = len(list(response.tools_executed or [])) if response else 0
            tool_ok = tool_count >= max(0, min_tool_calls)

            strict_ok = bool(response_ok and has_response_text and cap.overall >= threshold and tool_ok)
            if strict_ok:
                strict_passes += 1

            if expects_tool:
                tool_required_total += 1
                if strict_ok:
                    tool_required_strict_success += 1

        swebench_pass_rate = strict_passes / float(len(SCENARIOS) or 1)
        osworld_completion_rate = osworld_completed_count / float(len(SCENARIOS) or 1)
        if tool_required_total > 0:
            bfcl_tool_success_rate = tool_required_strict_success / float(tool_required_total)
        else:
            bfcl_tool_success_rate = swebench_pass_rate

        external_metrics = {
            "swebench_pass_rate": round(swebench_pass_rate, 4),
            "bfcl_tool_success_rate": round(bfcl_tool_success_rate, 4),
            "osworld_completion_rate": round(osworld_completion_rate, 4),
        }
        external_overall = _mean([
            swebench_pass_rate,
            bfcl_tool_success_rate,
            osworld_completion_rate,
        ])

        if self._cert_mode:
            scorer_mode = "external-aligned-cert-v1"
            # Use strict binary pass rate as primary cert headline to avoid inflation.
            overall_average = swebench_pass_rate
            pass_rate = swebench_pass_rate
        else:
            scorer_mode = "internal-sim-v1"
            pass_rate = internal_sim_pass_rate

        return AgentScorecard(
            name=source.name,
            source_path=_relpath_or_abs(source.path),
            scenario_scores=scenario_scores,
            category_profile=category_profile,
            overall_average=round(overall_average, 4),
            pass_rate=round(pass_rate, 4),
            scorer_mode=scorer_mode,
            external_metrics=external_metrics,
            internal_sim_overall_average=round(_mean([c.overall for c in cap_scores]), 4),
            internal_sim_pass_rate=round(internal_sim_pass_rate, 4),
            scenario_count_expected=len(SCENARIOS),
            scenario_count_scored=len(cap_scores),
            missing_scenarios=missing,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run external comparative benchmark loop against the Dad-Bot scenario suite.",
    )
    parser.add_argument(
        "--entry",
        action="append",
        default=[],
        help="Entrant artifact in the form NAME=path/to/artifact.json (repeatable).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for benchmark artifacts (default: artifacts/external_benchmark/<timestamp>).",
    )
    parser.add_argument(
        "--write-markdown",
        action="store_true",
        help="Also write a markdown summary report.",
    )
    parser.add_argument(
        "--include-stack",
        action="append",
        default=[],
        help=(
            "Write lane-targeted export bundles. Repeatable. "
            "Values: swebench, bfcl, osworld, all"
        ),
    )
    parser.add_argument(
        "--cert-mode",
        action="store_true",
        help="Enable certification mode: strict artifact validation, reject offline stubs, enforce field presence.",
    )
    return parser.parse_args()


def parse_entries(raw_entries: list[str]) -> list[ArtifactSource]:
    entries: list[ArtifactSource] = []
    for raw in raw_entries:
        token = str(raw or "").strip()
        if not token:
            continue
        if "=" not in token:
            raise ExternalBenchmarkError(
                f"Invalid --entry format: {token!r}. Expected NAME=path/to/artifact.json",
            )
        name, path_str = token.split("=", 1)
        name = name.strip()
        if not name:
            raise ExternalBenchmarkError(f"Invalid --entry format: {token!r}. NAME cannot be empty.")
        path = Path(path_str.strip())
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise ExternalBenchmarkError(f"Artifact file not found for {name!r}: {path}")
        entries.append(ArtifactSource(name=name, path=path.resolve()))
    if not entries:
        raise ExternalBenchmarkError("At least one --entry NAME=path artifact is required.")
    return entries


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExternalBenchmarkError(f"Invalid JSON in {path}: {exc}") from exc


def _validate_artifact_schema(payload: dict[str, Any], cert_mode: bool = False) -> None:
    """Enforce strict artifact schema.
    
    In cert mode: reject any unknown fields or missing required fields.
    In normal mode: be lenient with missing fields.
    """
    if not isinstance(payload.get("responses"), list):
        raise ExternalBenchmarkError("Artifact must contain a top-level 'responses' list.")
    
    # Allowed top-level fields.
    allowed_fields = {"agent", "model", "generated_at", "mode", "responses", "git_commit_sha", "offline_llm_stub"}
    actual_fields = set(payload.keys())
    if cert_mode:
        unknown = actual_fields - allowed_fields
        if unknown:
            raise ExternalBenchmarkError(f"Unknown fields in cert artifact: {unknown}. Allowed: {allowed_fields}")
    
    # In cert mode, reject offline stub flag.
    if cert_mode and payload.get("offline_llm_stub"):
        raise ExternalBenchmarkError("Certification artifacts cannot be generated with offline LLM stub enabled.")
    
    # Validate response items.
    for idx, item in enumerate(payload.get("responses", [])):
        if not isinstance(item, dict):
            raise ExternalBenchmarkError(f"Response item {idx} is not a dict.")
        
        scenario = str(item.get("scenario") or "").strip()
        if not scenario:
            raise ExternalBenchmarkError(f"Response item {idx} has missing or empty 'scenario'.")
        
        # In cert mode, require these fields.
        if cert_mode:
            if "completed" not in item:
                raise ExternalBenchmarkError(f"Response {scenario} missing required 'completed' field (cert mode).")
            if "error" not in item:
                raise ExternalBenchmarkError(f"Response {scenario} missing required 'error' field (cert mode).")
            if "response" not in item and "final_response" not in item:
                raise ExternalBenchmarkError(f"Response {scenario} missing 'response' or 'final_response' (cert mode).")


def _parse_responses(payload: dict[str, Any], cert_mode: bool = False) -> dict[str, ScenarioResponse]:
    """Parse and validate response items from artifact."""
    _validate_artifact_schema(payload, cert_mode=cert_mode)
    raw = payload.get("responses", [])

    parsed: dict[str, ScenarioResponse] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        scenario = str(item.get("scenario") or "").strip()
        if not scenario:
            continue
        tools = item.get("tools_executed")
        if tools is None:
            tools = item.get("tool_calls")
        memory = item.get("memory_accessed")
        if memory is None:
            memory = item.get("memory_reads")
        raw_state = item.get("raw_state")

        # In cert mode, 'completed' must be explicitly present; no default.
        if cert_mode:
            if "completed" not in item:
                raise ExternalBenchmarkError(f"Response {scenario} missing 'completed' (required in cert mode).")
            completed = bool(item["completed"])
        else:
            completed = bool(item.get("completed", True))  # Normal mode: default to True for backward compat.

        parsed[scenario] = ScenarioResponse(
            scenario=scenario,
            response=str(item.get("response") or item.get("final_response") or ""),
            completed=completed,
            error=str(item.get("error")) if item.get("error") else None,
            planner_output=item.get("planner_output") if isinstance(item.get("planner_output"), dict) else None,
            tools_executed=[str(v) for v in list(tools or [])],
            memory_accessed=[str(v) for v in list(memory or [])],
            raw_state=dict(raw_state) if isinstance(raw_state, dict) else {},
        )
    return parsed


def scenario_suite_sha256() -> str:
    hasher = hashlib.sha256()
    for scenario in SCENARIOS:
        hasher.update(scenario.name.encode("utf-8"))
        hasher.update(scenario.category.encode("utf-8"))
        hasher.update(scenario.input_text.encode("utf-8"))
        hasher.update(json.dumps(scenario.behavioral_spec, sort_keys=True).encode("utf-8"))
    return hasher.hexdigest()


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 64), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _relpath_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def build_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# External Benchmark Scorecard")
    lines.append("")
    lines.append(f"- generated_at: {payload.get('generated_at', '')}")
    lines.append(f"- scenario_count: {payload.get('scenario_count', 0)}")
    lines.append(f"- scoring_mode: {payload.get('scoring_mode', 'unknown')}")
    lines.append(f"- scenarios_sha256: {payload.get('scenarios_sha256', '')}")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append("| Rank | Entrant | Overall | Pass Rate |")
    lines.append("|---:|---|---:|---:|")
    for row in list(payload.get("ranking") or []):
        lines.append(
            f"| {int(row.get('rank', 0))} | {row.get('name', '')} | "
            f"{float(row.get('overall_average', 0.0)):.4f} | {float(row.get('pass_rate', 0.0)):.4f} |"
        )
    lines.append("")
    lines.append("## Entrants")
    lines.append("")

    for entrant in list(payload.get("entrants") or []):
        lines.append(f"### {entrant.get('name', '')}")
        lines.append("")
        lines.append(f"- source: {entrant.get('source_path', '')}")
        lines.append(f"- overall_average: {float(entrant.get('overall_average', 0.0)):.4f}")
        lines.append(f"- pass_rate: {float(entrant.get('pass_rate', 0.0)):.4f}")
        lines.append(f"- scorer_mode: {entrant.get('scorer_mode', 'unknown')}")
        ext = dict(entrant.get("external_metrics") or {})
        if ext:
            lines.append(f"- swebench_pass_rate: {float(ext.get('swebench_pass_rate', 0.0)):.4f}")
            lines.append(f"- bfcl_tool_success_rate: {float(ext.get('bfcl_tool_success_rate', 0.0)):.4f}")
            lines.append(f"- osworld_completion_rate: {float(ext.get('osworld_completion_rate', 0.0)):.4f}")
        lines.append(
            f"- internal_sim_overall_average: {float(entrant.get('internal_sim_overall_average', 0.0)):.4f}",
        )
        lines.append(f"- internal_sim_pass_rate: {float(entrant.get('internal_sim_pass_rate', 0.0)):.4f}")
        lines.append(f"- missing_scenarios: {len(list(entrant.get('missing_scenarios') or []))}")

        profile = dict(entrant.get("category_profile") or {})
        if profile:
            lines.append("")
            lines.append("Category profile:")
            for key in ["planning", "tool", "memory", "ux", "robustness"]:
                value = float(profile.get(key, 0.0))
                lines.append(f"- {key}: {value:.4f}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def parse_include_stack(raw_values: list[str]) -> set[str]:
    allowed = {"swebench", "bfcl", "osworld"}
    selected: set[str] = set()
    for raw in raw_values:
        token = str(raw or "").strip().lower()
        if not token:
            continue
        if token == "all":
            selected.update(allowed)
            continue
        if token not in allowed:
            raise ExternalBenchmarkError(
                f"Invalid --include-stack value: {token!r}. "
                "Expected one of: swebench, bfcl, osworld, all",
            )
        selected.add(token)
    return selected


def _responses_from_sources(sources: list[ArtifactSource]) -> dict[str, list[dict[str, Any]]]:
    responses_by_entry: dict[str, list[dict[str, Any]]] = {}
    for source in sources:
        payload = _load_json(source.path)
        parsed = _parse_responses(payload)
        rows: list[dict[str, Any]] = []
        for response in parsed.values():
            rows.append(
                {
                    "scenario": response.scenario,
                    "response": response.response,
                    "completed": response.completed,
                    "error": response.error,
                    "planner_output": response.planner_output,
                    "tools_executed": list(response.tools_executed),
                    "memory_accessed": list(response.memory_accessed),
                }
            )
        rows.sort(key=lambda item: str(item.get("scenario") or ""))
        responses_by_entry[source.name] = rows
    return responses_by_entry


def build_stack_includes(
    comparative: dict[str, Any],
    responses_by_entry: dict[str, list[dict[str, Any]]],
    selected: set[str],
) -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    generated_at = comparative.get("generated_at")
    scenario_count = int(comparative.get("scenario_count") or 0)
    scenarios_sha256 = comparative.get("scenarios_sha256")
    ranking = list(comparative.get("ranking") or [])
    entrants = list(comparative.get("entrants") or [])

    if "swebench" in selected:
        swebench_runs: list[dict[str, Any]] = []
        for row in ranking:
            name = str(row.get("name") or "")
            swebench_runs.append(
                {
                    "run_id": f"external::{name}",
                    "model_name_or_path": name,
                    "score": float(row.get("overall_average") or 0.0),
                    "pass_rate": float(row.get("pass_rate") or 0.0),
                    "predictions": responses_by_entry.get(name, []),
                }
            )
        bundles["swebench_like_bundle.json"] = {
            "format": "swebench-like-v1",
            "generated_at": generated_at,
            "scenario_count": scenario_count,
            "scenarios_sha256": scenarios_sha256,
            "leaderboard": swebench_runs,
        }

    if "bfcl" in selected:
        tool_name_counts: dict[str, int] = defaultdict(int)
        agents: list[dict[str, Any]] = []
        for entrant in entrants:
            name = str(entrant.get("name") or "")
            rows = responses_by_entry.get(name, [])
            total_calls = 0
            calls_on_completed = 0
            for item in rows:
                tools = list(item.get("tools_executed") or [])
                total_calls += len(tools)
                if bool(item.get("completed", False)):
                    calls_on_completed += len(tools)
                for tool in tools:
                    tool_name_counts[str(tool)] += 1

            agents.append(
                {
                    "agent": name,
                    "overall_average": float(entrant.get("overall_average") or 0.0),
                    "pass_rate": float(entrant.get("pass_rate") or 0.0),
                    "tool_calls_total": total_calls,
                    "tool_calls_on_completed": calls_on_completed,
                    "responses": rows,
                }
            )

        bundles["bfcl_like_bundle.json"] = {
            "format": "bfcl-like-v1",
            "generated_at": generated_at,
            "scenario_count": scenario_count,
            "scenarios_sha256": scenarios_sha256,
            "agents": agents,
            "tool_name_histogram": dict(sorted(tool_name_counts.items())),
        }

    if "osworld" in selected:
        tasks: list[dict[str, Any]] = []
        for entrant in entrants:
            name = str(entrant.get("name") or "")
            scenario_scores = list(entrant.get("scenario_scores") or [])
            response_map = {
                str(item.get("scenario") or ""): item
                for item in responses_by_entry.get(name, [])
                if str(item.get("scenario") or "")
            }
            for score in scenario_scores:
                scenario_name = str(score.get("scenario") or "")
                response = response_map.get(scenario_name, {})
                tasks.append(
                    {
                        "agent": name,
                        "task_id": scenario_name,
                        "category": score.get("category"),
                        "completed": bool(response.get("completed", False)),
                        "error": response.get("error"),
                        "final_response": response.get("response", ""),
                        "tools_executed": list(response.get("tools_executed") or []),
                        "memory_accessed": list(response.get("memory_accessed") or []),
                        "overall": float(score.get("overall") or 0.0),
                    }
                )

        bundles["osworld_ready_manifest.json"] = {
            "format": "osworld-ready-v1",
            "generated_at": generated_at,
            "scenario_count": scenario_count,
            "scenarios_sha256": scenarios_sha256,
            "tasks": tasks,
        }

    return bundles


def resolve_output_dir(raw_output_dir: str) -> Path:
    token = str(raw_output_dir or "").strip()
    if token:
        out = Path(token)
        if not out.is_absolute():
            out = ROOT / out
        return out.resolve()

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return (DEFAULT_OUTPUT_ROOT / f"external_benchmark_{stamp}").resolve()


def main() -> int:
    args = parse_args()
    try:
        sources = parse_entries(args.entry)
        runner = ExternalBenchmarkRunner(cert_mode=bool(args.cert_mode))
        comparative = runner.run(sources)

        out_dir = resolve_output_dir(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        scorecard_json = out_dir / "comparative_scorecard.json"
        scorecard_json.write_text(json.dumps(comparative, indent=2, sort_keys=True), encoding="utf-8")

        selected_stacks = parse_include_stack(args.include_stack)
        responses_by_entry = _responses_from_sources(sources) if selected_stacks else {}

        manifest = {
            "generated_at": comparative.get("generated_at"),
            "script": "tools/external_benchmark_loop.py",
            "script_sha256": comparative.get("script_sha256"),
            "scenarios_sha256": comparative.get("scenarios_sha256"),
            "entrant_roster_hash": comparative.get("entrant_roster_hash"),
            "cert_mode": comparative.get("cert_mode", False),
            "entries": [
                {"name": s.name, "path": _relpath_or_abs(s.path), "sha256": comparative["input_hashes"][s.name]}
                for s in sources
            ],
            "outputs": {
                "comparative_scorecard": "comparative_scorecard.json",
            },
        }
        manifest_path = out_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        if bool(args.write_markdown):
            md_path = out_dir / "comparative_scorecard.md"
            md_path.write_text(build_markdown(comparative), encoding="utf-8")
            manifest["outputs"]["comparative_scorecard_markdown"] = "comparative_scorecard.md"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        if selected_stacks:
            stack_dir = out_dir / "stack_includes"
            stack_dir.mkdir(parents=True, exist_ok=True)
            bundles = build_stack_includes(comparative, responses_by_entry, selected_stacks)
            for filename, payload in bundles.items():
                path = stack_dir / filename
                path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                manifest["outputs"][f"stack_include_{filename.replace('.json', '')}"] = (
                    f"stack_includes/{filename}"
                )
            manifest["stack_includes"] = sorted(selected_stacks)
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        print(f"WROTE_SCORECARD={_relpath_or_abs(scorecard_json)}")
        print(f"WROTE_MANIFEST={_relpath_or_abs(manifest_path)}")
        if bool(args.write_markdown):
            print(f"WROTE_MARKDOWN={_relpath_or_abs(out_dir / 'comparative_scorecard.md')}")
        if selected_stacks:
            print(f"WROTE_STACK_INCLUDES={_relpath_or_abs(out_dir / 'stack_includes')}")
        return 0
    except ExternalBenchmarkError as exc:
        print(f"ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
