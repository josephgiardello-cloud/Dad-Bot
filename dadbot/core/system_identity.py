from __future__ import annotations

import hashlib
import json
from typing import Any

from dadbot.core.critic import MAX_LOOP_ITERATIONS, PASS_THRESHOLD
from dadbot.core.graph_types import (
    GoalMutationOp,
    LedgerMutationOp,
    MemoryMutationOp,
    MutationKind,
    RelationshipMutationOp,
)
from dadbot.core.planner import ComplexityLevel, IntentType, ReplyStrategy


def _stable_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


def turn_graph_structure(*, tool_system_v2_enabled: bool) -> dict[str, Any]:
    if tool_system_v2_enabled:
        nodes = [
            "temporal",
            "preflight",
            "planner",
            "tool_router",
            "tool_executor",
            "inference",
            "safety",
            "reflection",
            "save",
        ]
        edges = [
            ["temporal", "preflight"],
            ["preflight", "planner"],
            ["planner", "tool_router"],
            ["tool_router", "tool_executor"],
            ["tool_executor", "inference"],
            ["inference", "safety"],
            ["safety", "reflection"],
            ["reflection", "save"],
        ]
    else:
        nodes = [
            "temporal",
            "preflight",
            "planner",
            "inference",
            "safety",
            "reflection",
            "save",
        ]
        edges = [
            ["temporal", "preflight"],
            ["preflight", "planner"],
            ["planner", "inference"],
            ["inference", "safety"],
            ["safety", "reflection"],
            ["reflection", "save"],
        ]
    return {
        "version": "v0",
        "tool_system_v2_enabled": bool(tool_system_v2_enabled),
        "nodes": nodes,
        "edges": edges,
    }


def planner_schema_snapshot() -> dict[str, Any]:
    return {
        "intent_type": [v.value for v in IntentType],
        "complexity": [v.value for v in ComplexityLevel],
        "strategy": [v.value for v in ReplyStrategy],
    }


def critic_schema_snapshot() -> dict[str, Any]:
    return {
        "pass_threshold": float(PASS_THRESHOLD),
        "max_loop_iterations": int(MAX_LOOP_ITERATIONS),
        "hard_failure_rules": ["reply_empty", "fallback_detected"],
        "strategy_checks": {
            "empathy_first": ["missing_empathy"],
            "question": ["reply_misses_question"],
            "moderate_or_complex": ["reply_too_brief"],
        },
        "tool_awareness_checks": {
            "necessity": ["tool_omission_detected", "tool_unnecessary_usage"],
            "correctness": [
                "tool_execution_mismatch",
                "tool_result_mismatch",
                "tool_correctness_low",
            ],
        },
    }


def memory_schema_snapshot() -> dict[str, Any]:
    return {
        "mutation_kinds": [v.value for v in MutationKind],
        "memory_ops": [v.value for v in MemoryMutationOp],
        "relationship_ops": [v.value for v in RelationshipMutationOp],
        "ledger_ops": [v.value for v in LedgerMutationOp],
        "goal_ops": [v.value for v in GoalMutationOp],
        "turn_state_keys": [
            "memories",
            "rich_context",
            "session_goals",
            "new_goals",
            "tool_ir",
            "tool_results",
        ],
    }


def determinism_manifest_logic_snapshot() -> dict[str, Any]:
    return {
        "manifest_keys": [
            "python_version",
            "env_hash",
            "dependency_versions",
            "timezone",
        ],
        "lock_payload_keys": [
            "user_input",
            "attachments",
            "llm_provider",
            "llm_model",
            "state_machine",
            "agent_blackboard_seed",
            "agent_blackboard_seed_fingerprint",
            "agent_blackboard",
            "agent_blackboard_fingerprint",
            "memory_fingerprint",
            "determinism_manifest_hash",
            "tool_trace_hash",
        ],
        "post_execution_hash_keys": [
            "lock_hash",
            "tool_trace_hash",
            "tool_system_v2_enabled",
        ],
        "strict_temporal_mode": "TurnTemporalAxis.from_lock_hash",
    }


def build_execution_schema_snapshot(*, tool_system_v2_enabled: bool) -> dict[str, Any]:
    return {
        "turn_graph": turn_graph_structure(
            tool_system_v2_enabled=tool_system_v2_enabled,
        ),
        "planner": planner_schema_snapshot(),
        "critic": critic_schema_snapshot(),
        "memory": memory_schema_snapshot(),
        "determinism_manifest_logic": determinism_manifest_logic_snapshot(),
    }


def compute_component_hashes(*, tool_system_v2_enabled: bool) -> dict[str, str]:
    graph = turn_graph_structure(tool_system_v2_enabled=tool_system_v2_enabled)
    planner = planner_schema_snapshot()
    critic = critic_schema_snapshot()
    memory = memory_schema_snapshot()
    return {
        "graph_hash": _stable_sha256(graph),
        "planner_hash": _stable_sha256(planner),
        "critic_hash": _stable_sha256(critic),
        "memory_schema_hash": _stable_sha256(memory),
    }


def compute_system_snapshot_v0_hash(*, tool_system_v2_enabled: bool) -> str:
    hashes = compute_component_hashes(tool_system_v2_enabled=tool_system_v2_enabled)
    return _stable_sha256(hashes)


SYSTEM_SNAPSHOT_V0_HASH: str = compute_system_snapshot_v0_hash(
    tool_system_v2_enabled=False,
)
