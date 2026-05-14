from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from tests.eval.adapters.langsmith import log_run as log_langsmith_run
from tests.eval.adapters.openai_evals import to_eval_case
from tests.eval.adapters.wandb import log as log_wandb
from tests.eval.datasets import EvalCase, tool_match
from tests.eval.harness import run_with_trace
from tests.eval.metrics import compute_metrics
from tests.eval.tracer import ToolCall, Trace


@dataclass
class RunnerConfig:
    mode: str = "production-safe"
    case_timeout_seconds: float = 0.0
    langsmith: bool = False
    wandb: bool = False
    openai_evals: bool = False
    langsmith_client: Any | None = None
    openai_evals_output_path: str | None = None
    progress_log_path: str | None = None

    @classmethod
    def from_env(cls) -> "RunnerConfig":
        def enabled(name: str) -> bool:
            return str(os.getenv(name, "0")).strip().lower() in {"1", "true", "yes", "on"}

        langsmith_on = enabled("DADBOT_EVAL_LANGSMITH")
        wandb_on = enabled("DADBOT_EVAL_WANDB")
        openai_on = enabled("DADBOT_EVAL_OPENAI_EVALS")
        openai_path = os.getenv("DADBOT_EVAL_OPENAI_EVALS_OUTPUT", "").strip() or None
        progress_log_path = os.getenv("DADBOT_EVAL_PROGRESS_LOG", "").strip() or None
        mode_raw = str(os.getenv("DADBOT_EVAL_MODE", "production-safe")).strip().lower()
        if mode_raw in {"strict", "evaluation-strict"}:
            mode = "evaluation-strict"
        else:
            mode = "production-safe"

        timeout_raw = os.getenv("DADBOT_EVAL_CASE_TIMEOUT_SECONDS", "0").strip()
        try:
            case_timeout_seconds = float(timeout_raw)
        except Exception:
            case_timeout_seconds = 0.0
        if case_timeout_seconds <= 0:
            case_timeout_seconds = 0.0

        langsmith_client = None
        if langsmith_on:
            try:
                from langsmith import Client  # type: ignore

                langsmith_client = Client()
            except Exception:
                # Keep runner resilient when optional dependency/config is missing.
                langsmith_on = False

        return cls(
            mode=mode,
            case_timeout_seconds=case_timeout_seconds,
            langsmith=langsmith_on,
            wandb=wandb_on,
            openai_evals=openai_on,
            langsmith_client=langsmith_client,
            openai_evals_output_path=openai_path,
            progress_log_path=progress_log_path,
        )


def _append_progress(cfg: RunnerConfig, payload: dict[str, Any]) -> None:
    path = str(cfg.progress_log_path or "").strip()
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")


def _trace_from_json(input_text: str, payload: dict[str, Any]) -> Trace:
    calls: list[ToolCall] = []
    for raw in list(payload.get("tool_calls", []) or []):
        if not isinstance(raw, dict):
            continue
        calls.append(
            ToolCall(
                name=str(raw.get("name") or ""),
                input=raw.get("input"),
                output=raw.get("output"),
            ),
        )

    return Trace(
        input=input_text,
        final_output=str(payload.get("final_output") or ""),
        tool_calls=calls,
        error=(str(payload.get("error")) if payload.get("error") is not None else None),
        steps=int(payload.get("steps") or 0),
        latency_ms=(int(payload.get("latency_ms")) if payload.get("latency_ms") is not None else None),
        robustness_suppressed=bool(payload.get("robustness_suppressed") or False),
        decision_outcome=str(payload.get("decision_outcome") or "no_tool_needed"),
        planner_status=str(payload.get("planner_status") or ""),
        planner_tool=str(payload.get("planner_tool") or ""),
        robustness_reason=(str(payload.get("robustness_reason")) if payload.get("robustness_reason") else None),
    )


def _run_case_isolated_with_timeout(*, input_text: str, timeout_seconds: float) -> Trace:
    script = (
        "import json, os, sys;"
        "from tests.eval.cli import production_bootstrap;"
        "from tests.eval.harness import run_with_trace;"
        f"_input = json.loads({json.dumps(json.dumps(input_text))});"
        "runtime, tool_registry, _ = production_bootstrap();"
        "trace = run_with_trace(_input, runtime, tool_registry);"
        "payload = {"
        "'final_output': trace.final_output,"
        "'error': trace.error,"
        "'steps': trace.steps,"
        "'latency_ms': trace.latency_ms,"
        "'tool_calls': [{'name': c.name, 'input': c.input, 'output': c.output} for c in trace.tool_calls],"
        "'robustness_suppressed': trace.robustness_suppressed,"
        "'decision_outcome': trace.decision_outcome,"
        "'planner_status': trace.planner_status,"
        "'planner_tool': trace.planner_tool,"
        "'robustness_reason': trace.robustness_reason,"
        "};"
        "sys.stdout.write('TRACE_JSON::' + json.dumps(payload, ensure_ascii=True, default=str) + '\\n');"
        "sys.stdout.flush();"
        "os._exit(0)"
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=max(1.0, float(timeout_seconds)),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return Trace(
            input=input_text,
            final_output="",
            tool_calls=[],
            error=f"case timeout after {timeout_seconds:.1f}s",
            steps=0,
            latency_ms=int(timeout_seconds * 1000),
            robustness_suppressed=False,
            decision_outcome="no_tool_needed",
            planner_status="",
            planner_tool="",
            robustness_reason=None,
        )

    marker = "TRACE_JSON::"
    for line in reversed((proc.stdout or "").splitlines()):
        if marker in line:
            raw = line.split(marker, 1)[1].strip()
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    return _trace_from_json(input_text, payload)
            except Exception:
                break

    stderr_tail = (proc.stderr or "").strip()[-300:]
    return Trace(
        input=input_text,
        final_output="",
        tool_calls=[],
        error=f"isolated case failed rc={proc.returncode}; stderr={stderr_tail}",
        steps=0,
        latency_ms=None,
        robustness_suppressed=False,
        decision_outcome="no_tool_needed",
        planner_status="",
        planner_tool="",
        robustness_reason=None,
    )


def _is_strict_mode(cfg: RunnerConfig) -> bool:
    return str(cfg.mode).strip().lower() == "evaluation-strict"


def _is_missing_service_failure(trace: Any) -> bool:
    haystacks = [
        str(getattr(trace, "error", "") or ""),
        str(getattr(trace, "final_output", "") or ""),
    ]
    for text in haystacks:
        lowered = text.lower()
        if "service '" in lowered and "not registered" in lowered:
            return True
        if "missing required service" in lowered:
            return True
    return False


def _assert_execution_path(trace: Any, *, require_tool_calls: bool) -> None:
    if int(getattr(trace, "steps", 0) or 0) <= 0:
        raise RuntimeError("Execution-path assertion failed: trace.steps must be > 0")
    if require_tool_calls:
        tool_calls = list(getattr(trace, "tool_calls", []) or [])
        if len(tool_calls) <= 0:
            raise RuntimeError("Execution-path assertion failed: trace.tool_calls must be non-empty")


def _collect_strict_coherence_violations(trace: Trace, case: EvalCase) -> list[str]:
    violations: list[str] = []
    planner_tool = str(getattr(trace, "planner_tool", "") or "").strip()
    planner_status = str(getattr(trace, "planner_status", "") or "").strip().lower()
    decision_outcome = str(getattr(trace, "decision_outcome", "") or "no_tool_needed").strip().lower()
    tool_calls = list(getattr(trace, "tool_calls", []) or [])

    # Rule 1 - tool selection coherence.
    if planner_tool and decision_outcome == "no_tool_needed":
        violations.append("rule1: planner_tool set but decision_outcome=no_tool_needed")

    # Rule 2 - execution consistency.
    if decision_outcome == "executed_tool" and len(tool_calls) <= 0:
        violations.append("rule2: decision_outcome=executed_tool but trace.tool_calls is empty")

    # Rule 3 - strict expected-tool dataset rule.
    if bool(getattr(case, "expects_tool", False)) and decision_outcome != "executed_tool":
        violations.append("rule3: expects_tool case did not end in decision_outcome=executed_tool")

    # Rule 4 - suppression transparency.
    robustness_reason = str(getattr(trace, "robustness_reason", "") or "").strip()
    if decision_outcome == "robustness_suppressed" and not robustness_reason:
        violations.append("rule4: robustness_suppressed without robustness_reason")

    # Additional certifying coherence: planner-selected path should resolve to tool_selected.
    if planner_tool and decision_outcome == "executed_tool" and planner_status and planner_status != "tool_selected":
        violations.append("planner_status mismatch: planner_tool set and executed_tool but planner_status!=tool_selected")

    return violations


def _has_planner_execution_mismatch(trace: Trace) -> bool:
    planner_tool = str(getattr(trace, "planner_tool", "") or "").strip()
    if not planner_tool:
        return False
    decision_outcome = str(getattr(trace, "decision_outcome", "") or "no_tool_needed").strip().lower()
    tool_calls = list(getattr(trace, "tool_calls", []) or [])
    if decision_outcome != "executed_tool" or len(tool_calls) <= 0:
        return False
    executed = str(getattr(tool_calls[0], "name", "") or "").strip()
    return bool(executed and executed != planner_tool)


def _resolve_runtime(runtime_or_factory: Any):
    if callable(runtime_or_factory):
        return runtime_or_factory()
    return runtime_or_factory


def run_eval_suite(
    cases: list[EvalCase],
    runtime_or_factory: Any,
    *,
    tool_registry: Any | None = None,
    config: RunnerConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RunnerConfig()
    traces: list[dict[str, Any]] = []
    openai_eval_rows: list[dict[str, Any]] = []
    strict_failures: list[dict[str, Any]] = []
    coherence_violations: list[dict[str, Any]] = []
    planner_execution_mismatches = 0

    for idx, case in enumerate(cases):
        _append_progress(
            cfg,
            {
                "event": "case_start",
                "index": idx,
                "input": case.input,
                "timeout_seconds": cfg.case_timeout_seconds,
            },
        )

        if float(cfg.case_timeout_seconds) > 0:
            trace = _run_case_isolated_with_timeout(
                input_text=case.input,
                timeout_seconds=float(cfg.case_timeout_seconds),
            )
        else:
            runtime = _resolve_runtime(runtime_or_factory)
            trace = run_with_trace(case.input, runtime, tool_registry)

        _append_progress(
            cfg,
            {
                "event": "case_complete",
                "index": idx,
                "error": trace.error,
                "tool_calls": len(list(getattr(trace, "tool_calls", []) or [])),
            },
        )

        strict_error: str | None = None
        case_coherence_violations: list[str] = []
        if _is_strict_mode(cfg):
            if _is_missing_service_failure(trace):
                strict_error = "Missing service detected during strict eval execution"
            else:
                try:
                    _assert_execution_path(trace, require_tool_calls=bool(case.expected_tools))
                except RuntimeError as exc:
                    strict_error = str(exc)

            case_coherence_violations = _collect_strict_coherence_violations(trace, case)

            if strict_error:
                strict_failures.append(
                    {
                        "index": idx,
                        "input": case.input,
                        "error": strict_error,
                    },
                )

            if len(case_coherence_violations) > 0:
                coherence_violations.append(
                    {
                        "index": idx,
                        "input": case.input,
                        "violations": list(case_coherence_violations),
                    },
                )
                strict_failures.append(
                    {
                        "index": idx,
                        "input": case.input,
                        "error": "; ".join(case_coherence_violations),
                    },
                )

        if _has_planner_execution_mismatch(trace):
            planner_execution_mismatches += 1

        metrics = compute_metrics(trace)
        match = tool_match(trace, case.expected_tools)

        row = {
            "input": case.input,
            "trace": trace,
            "metrics": metrics,
            "tool_match": match,
            "strict_error": strict_error,
            "coherence_violations": case_coherence_violations,
            "planner_execution_mismatch": _has_planner_execution_mismatch(trace),
        }
        traces.append(row)

        if cfg.langsmith and cfg.langsmith_client is not None:
            try:
                log_langsmith_run(cfg.langsmith_client, trace)
            except Exception:
                pass
        if cfg.wandb:
            try:
                log_wandb(trace, metrics)
            except Exception:
                pass
        if cfg.openai_evals:
            openai_eval_rows.append(to_eval_case(trace))

    total = max(1, len(traces))
    success_rate = sum(1 for r in traces if r["metrics"]["success"]) / total
    irrelevance_rate = sum(float(r["metrics"]["irrelevance"]) for r in traces) / total
    avg_tool_calls = sum(int(r["metrics"]["tool_calls"]) for r in traces) / total
    efficiency = sum(float(r["metrics"]["efficiency"]) for r in traces) / total
    tool_execution_rate = sum(float(r["metrics"]["tool_execution_rate"]) for r in traces) / total
    no_tool_rate = sum(float(r["metrics"]["no_tool_rate"]) for r in traces) / total
    robustness_suppression_rate = (
        sum(float(r["metrics"]["robustness_suppression_rate"]) for r in traces) / total
    )

    summary = {
        "success_rate": round(success_rate, 4),
        "irrelevance_rate": round(irrelevance_rate, 4),
        "avg_tool_calls": round(avg_tool_calls, 4),
        "efficiency": round(efficiency, 4),
        "tool_execution_rate": round(tool_execution_rate, 4),
        "no_tool_rate": round(no_tool_rate, 4),
        "robustness_suppression_rate": round(robustness_suppression_rate, 4),
        "runs": traces,
        "openai_evals": openai_eval_rows,
        "strict_failures": strict_failures,
        "coherence_violations": coherence_violations,
        "coherence_violations_count": len(coherence_violations),
        "planner_execution_mismatches": int(planner_execution_mismatches),
    }

    if cfg.openai_evals and cfg.openai_evals_output_path:
        output_path = Path(cfg.openai_evals_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in openai_eval_rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    return summary
