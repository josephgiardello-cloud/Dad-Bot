from __future__ import annotations

import os
from typing import Any


def _result_output_payload_impl(result: Any) -> dict[str, Any]:
    return {
        "response": str(result[0] if isinstance(result, tuple) and len(result) >= 1 else ""),
        "should_end": bool(result[1] if isinstance(result, tuple) and len(result) >= 2 else False),
    }


def _build_composition_payload_impl(
    control_plane: Any,
    *,
    job: Any,
    terminal_state: dict[str, Any],
    output_payload: dict[str, Any],
    trace_events: list[dict[str, Any]],
    state_before_hash: str,
    state_after_hash: str,
) -> dict[str, Any]:
    state_delta_hash = control_plane._stable_hash(
        {
            "before": str(state_before_hash or ""),
            "after": str(state_after_hash or ""),
        },
    )
    event_log_hash = control_plane._event_stream_digest(trace_events)
    return {
        "contract_version": "turn-composition-v1",
        "context_input_hash": control_plane._stable_hash(
            {
                "session_id": str(job.session_id or ""),
                "trace_id": str(job.trace_id or ""),
                "user_input": str(job.user_input or ""),
                "attachments": list(job.attachments or []),
                "metadata": dict(job.metadata or {}),
            },
        ),
        "execution_dag_hash": str(terminal_state.get("execution_dag_hash") or ""),
        "policy_hash": str(terminal_state.get("policy_hash") or ""),
        "state_delta_hash": state_delta_hash,
        "event_log_hash": event_log_hash,
        "output_hash": control_plane._stable_hash(output_payload),
        "mutation_effects_hash": str(terminal_state.get("post_commit_mutation_effects_hash") or ""),
        "determinism_closure_hash": str(terminal_state.get("determinism_closure_hash") or ""),
    }


def _build_confluence_payload_impl(
    control_plane: Any,
    *,
    job: Any,
    terminal_state: dict[str, Any],
    output_payload: dict[str, Any],
    trace_events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "contract_version": "turn-confluence-v1",
        "semantic_input_hash": control_plane._stable_hash(
            {
                "user_input": str(job.user_input or ""),
                "attachments": list(job.attachments or []),
                "semantic_eval_input_hash": str(dict(job.metadata or {}).get("semantic_eval_input_hash") or ""),
            },
        ),
        "execution_dag_hash": str(terminal_state.get("execution_dag_hash") or ""),
        "policy_hash": str(terminal_state.get("policy_hash") or ""),
        "output_hash": control_plane._stable_hash(output_payload),
        "mutation_effects_hash": str(terminal_state.get("post_commit_mutation_effects_hash") or ""),
        "determinism_closure_hash": str(terminal_state.get("determinism_closure_hash") or ""),
        "event_semantic_hash": control_plane._event_semantic_digest(trace_events),
    }


def _expected_hashes_from_metadata_impl(metadata: dict[str, Any]) -> tuple[str, str]:
    expected = str(dict(metadata or {}).get("expected_execution_composition_hash") or "").strip()
    expected_confluence = str(
        dict(metadata or {}).get("expected_execution_confluence_hash") or "",
    ).strip()
    return expected, expected_confluence


def _confluence_config_from_metadata_impl(metadata: dict[str, Any]) -> tuple[str, str]:
    confluence_key = str(dict(metadata or {}).get("_global_confluence_key") or "").strip()
    confluence_mode = str(dict(metadata or {}).get("_global_confluence_mode") or "off").strip().lower()
    return confluence_key, confluence_mode


def _validate_composition_expectations_impl(
    *,
    expected: str,
    expected_confluence: str,
    composition_hash: str,
    confluence_class_hash: str,
) -> None:
    if expected and expected != composition_hash:
        raise RuntimeError(
            f"Execution composition mismatch: expected={expected!r}, actual={composition_hash!r}",
        )
    if expected_confluence and expected_confluence != confluence_class_hash:
        raise RuntimeError(
            f"Execution confluence mismatch: expected={expected_confluence!r}, actual={confluence_class_hash!r}",
        )


def _enforce_global_confluence_law_impl(
    control_plane: Any,
    *,
    confluence_key: str,
    confluence_mode: str,
    confluence_class_hash: str,
    expected_confluence: str,
    logger: Any,
) -> dict[str, Any]:
    confluence_report = {
        "enforced": False,
        "mode": confluence_mode,
        "key": confluence_key,
        "observed_hash": confluence_class_hash,
        "expected_hash": expected_confluence,
        "contract_version": "turn-confluence-v1",
    }
    if not confluence_key or confluence_mode == "off":
        return confluence_report

    control_plane._confluence_metrics["attempted"] = int(control_plane._confluence_metrics.get("attempted", 0)) + 1
    known = str(control_plane._global_confluence_contracts.get(confluence_key) or "")
    if not known:
        control_plane._global_confluence_contracts[confluence_key] = confluence_class_hash
        confluence_report["enforced"] = True
        confluence_report["action"] = "bound_first_observation"
        control_plane._confluence_metrics["bound_first_observation"] = int(
            control_plane._confluence_metrics.get("bound_first_observation", 0),
        ) + 1
        return confluence_report

    if known != confluence_class_hash:
        confluence_report["enforced"] = True
        confluence_report["expected_hash"] = known
        confluence_report["action"] = "mismatch"
        control_plane._last_confluence_report = dict(confluence_report)
        control_plane._confluence_metrics["mismatch"] = int(control_plane._confluence_metrics.get("mismatch", 0)) + 1
        fail_mode = str(
            os.environ.get("DADBOT_CONFLUENCE_VIOLATION_MODE", "fail"),
        ).strip().lower()
        if fail_mode != "audit":
            control_plane._confluence_metrics["enforced_blocked"] = int(
                control_plane._confluence_metrics.get("enforced_blocked", 0),
            ) + 1
            raise RuntimeError(
                "Global confluence law violated for key="
                f"{confluence_key!r}: expected={known!r}, "
                f"actual={confluence_class_hash!r}",
            )
        logger.warning(
            "Global confluence law mismatch (audit override): key=%s expected=%s actual=%s",
            confluence_key,
            known,
            confluence_class_hash,
        )
        return confluence_report

    confluence_report["enforced"] = True
    confluence_report["expected_hash"] = known
    confluence_report["action"] = "matched"
    control_plane._confluence_metrics["matched"] = int(control_plane._confluence_metrics.get("matched", 0)) + 1
    return confluence_report


def _record_turn_composition_contract_impl(
    control_plane: Any,
    *,
    session: dict[str, Any],
    job: Any,
    result: Any,
    state_before_hash: str,
    logger: Any,
) -> dict[str, Any]:
    state = dict(session.get("state") or {})
    state_after_hash = control_plane._stable_hash(state)
    terminal_state = dict(state.get("last_terminal_state") or {})
    trace_events = control_plane._job_trace_events(job)
    if any(str(event.get("type") or "").strip() for event in list(trace_events or [])):
        control_plane._assert_lifecycle_order(trace_events)

    output_payload = _result_output_payload_impl(result)
    composition_payload = _build_composition_payload_impl(
        control_plane,
        job=job,
        terminal_state=terminal_state,
        output_payload=output_payload,
        trace_events=trace_events,
        state_before_hash=state_before_hash,
        state_after_hash=state_after_hash,
    )
    composition_hash = control_plane._stable_hash(composition_payload)
    confluence_payload = _build_confluence_payload_impl(
        control_plane,
        job=job,
        terminal_state=terminal_state,
        output_payload=output_payload,
        trace_events=trace_events,
    )
    confluence_class_hash = control_plane._stable_hash(confluence_payload)
    contract = dict(composition_payload)
    contract["composition_hash"] = composition_hash
    contract["confluence_class_hash"] = confluence_class_hash

    expected, expected_confluence = _expected_hashes_from_metadata_impl(dict(job.metadata or {}))
    _validate_composition_expectations_impl(
        expected=expected,
        expected_confluence=expected_confluence,
        composition_hash=composition_hash,
        confluence_class_hash=confluence_class_hash,
    )

    confluence_key, confluence_mode = _confluence_config_from_metadata_impl(dict(job.metadata or {}))
    confluence_report = _enforce_global_confluence_law_impl(
        control_plane,
        confluence_key=confluence_key,
        confluence_mode=confluence_mode,
        confluence_class_hash=confluence_class_hash,
        expected_confluence=expected_confluence,
        logger=logger,
    )
    control_plane._last_confluence_report = dict(confluence_report)

    state_mut = session.setdefault("state", {})
    if isinstance(state_mut, dict):
        state_mut["last_execution_composition_contract"] = dict(contract)
        state_mut["last_execution_confluence_report"] = dict(confluence_report)
    return contract