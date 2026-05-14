from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Dad import DadBot
from dadbot.core.system_identity import (  # noqa: E402
    SYSTEM_SNAPSHOT_V0_HASH,
    build_execution_schema_snapshot,
    compute_component_hashes,
)

SNAPSHOT_DIR = ROOT / "snapshots" / "system_snapshot_v0"

CANONICAL_PROMPTS: list[str] = [
    "Hey Dad, good morning.",
    "I slept okay but still feel tired.",
    "Can you help me plan my work day?",
    "I feel anxious before meetings.",
    "Give me one short focus tip.",
    "What did I say about my mood yesterday?",
    "I want to save money this month.",
    "Can you remind me to call mom tonight?",
    "I had a rough day at work.",
    "Please help me prioritize these tasks.",
    "I am trying to walk every evening.",
    "What should I do first after lunch?",
    "I feel better than last week.",
    "Can we keep this week simple?",
    "I am worried about deadlines.",
    "How can I stay calm during stress?",
    "Give me a practical next step.",
    "I appreciate your support.",
    "Can you summarize my current goals?",
    "Thanks Dad.",
]


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _stable_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _git_commit_hash() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return str(completed.stdout or "").strip()
    except Exception:
        return "unknown"


def _dependency_lock() -> str:
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return str(completed.stdout or "").strip()
    except Exception as exc:
        return f"# failed to capture pip freeze\n# {exc}"


def _runtime_config_dump() -> dict[str, Any]:
    env_vars = {key: value for key, value in os.environ.items() if key.startswith("DADBOT_")}
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "cwd": str(ROOT),
        "env": env_vars,
        "tool_system_v2_enabled": False,
        "checkpoint_every_node": False,
        "strict_mode": False,
    }


async def _capture_runtime_behavior() -> dict[str, Any]:
    bot = DadBot()
    try:
        orchestrator = bot.turn_orchestrator
        service = orchestrator.registry.get("agent_service")

        async def _deterministic_agent(context, _rich):
            return (f"SNAPSHOT_V0::{str(context.user_input or '').strip()}", False)

        service.run_agent = _deterministic_agent

        # Disable non-essential side effects for clean, tool-free baseline traces.
        mc = getattr(bot, "memory_coordinator", None)
        if mc is not None:
            mc.consolidate_memories = lambda **_: None
            mc.apply_controlled_forgetting = lambda **_: None
        rm = getattr(bot, "relationship_manager", None)
        if rm is not None:
            rm.materialize_projection = lambda **_: None
        mm = getattr(bot, "memory_manager", None)
        gm = getattr(mm, "graph_manager", None) if mm is not None else None
        if gm is not None:
            gm.sync_graph_store = lambda **_: None
        if hasattr(bot, "validate_reply"):
            setattr(bot, "validate_reply", lambda _u, r: r)

        traces: list[dict[str, Any]] = []
        session_id = "system-snapshot-v0"
        for index, prompt in enumerate(CANONICAL_PROMPTS, start=1):
            result = await orchestrator.handle_turn(prompt, session_id=session_id)
            context = getattr(orchestrator, "_last_turn_context", None)
            traces.append(
                {
                    "index": index,
                    "prompt": prompt,
                    "expected_output": result[0],
                    "should_end": bool(result[1]),
                    "tool_free": True,
                    "stage_traces": [
                        {
                            "stage": str(getattr(trace, "stage", "")),
                            "duration_ms": float(getattr(trace, "duration_ms", 0.0) or 0.0),
                            "error": getattr(trace, "error", None),
                        }
                        for trace in list(getattr(context, "stage_traces", []) or [])
                    ],
                    "execution_stages": sorted(list(context.state.get("_graph_executed_stages") or []))
                    if context
                    else [],
                    "determinism": dict(context.metadata.get("determinism") or {}) if context else {},
                    "tool_ir": dict(context.state.get("tool_ir") or {}) if context else {},
                    "tool_results": list(context.state.get("tool_results") or []) if context else [],
                }
            )

        return {
            "canonical_prompt_count": len(CANONICAL_PROMPTS),
            "session_id": session_id,
            "gold_traces": traces,
        }
    finally:
        try:
            bot.shutdown()
        except Exception:
            pass


def main() -> int:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    execution_schema = build_execution_schema_snapshot(tool_system_v2_enabled=False)
    component_hashes = compute_component_hashes(tool_system_v2_enabled=False)

    metadata = {
        "snapshot_name": "SYSTEM_SNAPSHOT_V0",
        "snapshot_hash": SYSTEM_SNAPSHOT_V0_HASH,
        "code_snapshot": {
            "git_commit_hash": _git_commit_hash(),
            "dependency_lockfile": "dependency_lock.txt",
            "runtime_config_dump": "runtime_config_dump.json",
        },
        "component_hashes": component_hashes,
    }

    _stable_json_dump(SNAPSHOT_DIR / "metadata.json", metadata)
    _stable_json_dump(SNAPSHOT_DIR / "execution_schema_snapshot.json", execution_schema)
    _stable_json_dump(SNAPSHOT_DIR / "runtime_config_dump.json", _runtime_config_dump())

    dependency_lock_text = _dependency_lock().strip() + "\n"
    (SNAPSHOT_DIR / "dependency_lock.txt").write_text(dependency_lock_text, encoding="utf-8")

    runtime_behavior = asyncio.run(_capture_runtime_behavior())
    _stable_json_dump(SNAPSHOT_DIR / "runtime_behavior_snapshot.json", runtime_behavior)

    (SNAPSHOT_DIR / "SYSTEM_SNAPSHOT_V0_HASH.txt").write_text(SYSTEM_SNAPSHOT_V0_HASH + "\n", encoding="utf-8")

    print(f"Wrote snapshot artifacts to {SNAPSHOT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
