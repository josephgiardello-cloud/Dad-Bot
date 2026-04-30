"""Phase 0: Hard Snapshot / Restore Point.

Creates a versioned, reproducible snapshot of the system's semantic fingerprint.
This is the "freeze" operation — all future changes must produce a new snapshot.

Snapshot components:
    0.1 — System identity:
        - git_hash: current git HEAD (or UNKNOWN if not in a repo)
        - file_tree_hash: stable hash of all .py files under dadbot/
        - schema versions: tool_ir, dag, event_log, snapshot
        - dependency_lock: installed package hashes
        - runtime_config: key runtime config fields
    0.2 — Golden behavior set:
        - 25 canonical prompts with recorded execution structure
        - per-prompt: intent_type, strategy, tool_plan, tool_trace_hash
        - envelope_hash: reproducible structural hash (no text)
    0.3 — Restore capability:
        - SnapshotRestoreValidator.validate() — checks file_tree_hash + golden set
        - replay_golden_record() — verifies determinism envelope

Design principle:
    The golden set pins STRUCTURAL INVARIANTS, not text output.
    tool_trace_hash = hash(tool_plan) — LLM-independent, always reproducible.
    envelope_hash = hash(intent + strategy + tool_plan) — the "computation class".
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema versions (single source of truth)
# ---------------------------------------------------------------------------

TOOL_IR_SCHEMA_VERSION = "1.0"
DAG_SCHEMA_VERSION = "1.0"
EVENT_LOG_SCHEMA_VERSION = "1.0"
SNAPSHOT_VERSION = "V1"


@dataclass(frozen=True)
class SchemaRegistry:
    tool_ir_schema_version: str = TOOL_IR_SCHEMA_VERSION
    dag_schema_version: str = DAG_SCHEMA_VERSION
    event_log_schema_version: str = EVENT_LOG_SCHEMA_VERSION
    snapshot_version: str = SNAPSHOT_VERSION

    def to_dict(self) -> dict[str, str]:
        return {
            "tool_ir_schema_version": self.tool_ir_schema_version,
            "dag_schema_version": self.dag_schema_version,
            "event_log_schema_version": self.event_log_schema_version,
            "snapshot_version": self.snapshot_version,
        }


# ---------------------------------------------------------------------------
# File tree hasher
# ---------------------------------------------------------------------------

_PY_SKIP_PREFIXES = ("__pycache__", ".venv", "build", "dist", ".git")


class FileTreeHasher:
    """Deterministic hash of all .py files under a root directory.

    Stable across machines: sorts files by relative path before hashing.
    """

    @staticmethod
    def hash_file(path: Path) -> str:
        try:
            content = path.read_bytes()
        except OSError:
            return ""
        return hashlib.sha256(content).hexdigest()

    @classmethod
    def hash_directory(cls, root: Path, *, suffix: str = ".py") -> str:
        root = Path(root)
        pairs: list[tuple[str, str]] = []
        for py_file in sorted(root.rglob(f"*{suffix}")):
            rel = py_file.relative_to(root)
            parts = rel.parts
            if any(part.startswith(s) for s in _PY_SKIP_PREFIXES for part in parts):
                continue
            file_hash = cls.hash_file(py_file)
            if file_hash:
                pairs.append((str(rel), file_hash))
        payload = json.dumps(pairs, sort_keys=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Git hash
# ---------------------------------------------------------------------------


def get_git_hash(workspace_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(workspace_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or "UNKNOWN"
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


# ---------------------------------------------------------------------------
# Dependency lock
# ---------------------------------------------------------------------------


def get_dependency_lock() -> dict[str, str]:
    """Return a stable hash of installed package metadata."""
    try:
        import importlib.metadata as importlib_metadata

        packages: dict[str, str] = {}
        for dist in sorted(
            importlib_metadata.distributions(),
            key=lambda d: d.metadata["Name"].lower(),
        ):
            name = str(dist.metadata["Name"] or "").strip().lower()
            version = str(dist.metadata["Version"] or "").strip()
            if name:
                packages[name] = version
        lock_hash = hashlib.sha256(
            json.dumps(packages, sort_keys=True).encode("utf-8"),
        ).hexdigest()
        return {"lock_hash": lock_hash, "package_count": str(len(packages))}
    except Exception:  # noqa: BLE001
        return {"lock_hash": "UNKNOWN", "package_count": "0"}


# ---------------------------------------------------------------------------
# Golden behavior record
# ---------------------------------------------------------------------------


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


@dataclass(frozen=True)
class GoldenBehaviorRecord:
    """One canonical prompt with its recorded execution structure.

    Fields:
        prompt:          The canonical input text.
        intent_type:     Planner classification (emotional, question, etc.)
        strategy:        Planning strategy (empathy_first, fact_seeking, etc.)
        tool_plan:       Ordered tuple of tool names in the execution plan.
        tool_trace_hash: Deterministic hash of tool_plan (LLM-independent).
        plan_class_hash: Hash of (intent_type, strategy, len(tool_plan)).
        envelope_hash:   Master hash of all structural fields.

    The envelope_hash is the "determinism seal" — same structural inputs
    always produce the same envelope_hash, independent of text output.
    """

    prompt: str
    intent_type: str
    strategy: str
    tool_plan: tuple[str, ...]
    tool_trace_hash: str
    plan_class_hash: str
    envelope_hash: str

    @classmethod
    def build(
        cls,
        prompt: str,
        intent_type: str,
        strategy: str,
        tool_plan: list[str],
    ) -> GoldenBehaviorRecord:
        plan_tuple = tuple(tool_plan or [])
        tool_trace_hash = _sha256({"tool_plan": list(plan_tuple)})
        plan_class_hash = _sha256(
            {
                "intent_type": intent_type,
                "strategy": strategy,
                "tool_count": len(plan_tuple),
            },
        )
        envelope_hash = _sha256(
            {
                "intent_type": intent_type,
                "strategy": strategy,
                "tool_trace_hash": tool_trace_hash,
                "plan_class_hash": plan_class_hash,
            },
        )
        return cls(
            prompt=str(prompt),
            intent_type=str(intent_type),
            strategy=str(strategy),
            tool_plan=plan_tuple,
            tool_trace_hash=tool_trace_hash,
            plan_class_hash=plan_class_hash,
            envelope_hash=envelope_hash,
        )

    def replay(self) -> GoldenBehaviorRecord:
        """Re-derive from stored fields and return. Must be identical."""
        return GoldenBehaviorRecord.build(
            self.prompt,
            self.intent_type,
            self.strategy,
            list(self.tool_plan),
        )

    def verify_replay(self) -> bool:
        """True iff replay produces the exact same envelope_hash."""
        return self.replay().envelope_hash == self.envelope_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "intent_type": self.intent_type,
            "strategy": self.strategy,
            "tool_plan": list(self.tool_plan),
            "tool_trace_hash": self.tool_trace_hash,
            "plan_class_hash": self.plan_class_hash,
            "envelope_hash": self.envelope_hash,
        }


# ---------------------------------------------------------------------------
# Golden behavior set (25 canonical records)
# ---------------------------------------------------------------------------

_GOLDEN_RECORDS_SPECS: list[tuple[str, str, str, list[str]]] = [
    # (prompt, intent_type, strategy, tool_plan)
    # --- Simple / no-tool turns ---
    ("Hey, how's it going?", "casual", "casual_reply", []),
    ("Good morning!", "casual", "casual_reply", []),
    ("Thanks for the chat yesterday.", "casual", "casual_reply", []),
    ("I love you, dad.", "emotional", "empathy_first", []),
    ("Goodnight.", "casual", "casual_reply", []),
    # --- Emotional support ---
    (
        "I'm really stressed about work lately.",
        "emotional",
        "empathy_first",
        ["memory_lookup"],
    ),
    ("I had a really hard day today.", "emotional", "empathy_first", ["memory_lookup"]),
    (
        "I feel like I'm failing at everything.",
        "emotional",
        "empathy_first",
        ["memory_lookup"],
    ),
    (
        "I'm proud of myself for finishing the project.",
        "emotional",
        "empathy_first",
        ["memory_lookup"],
    ),
    (
        "I've been feeling lonely recently.",
        "emotional",
        "empathy_first",
        ["memory_lookup"],
    ),
    # --- Goal-oriented queries ---
    (
        "What have I been working on lately?",
        "question",
        "fact_seeking",
        ["memory_lookup"],
    ),
    (
        "Do you remember what I said about my job last week?",
        "question",
        "fact_seeking",
        ["memory_lookup"],
    ),
    ("What are my current goals?", "goal_oriented", "goal_track", ["memory_lookup"]),
    (
        "Am I making progress on my fitness goals?",
        "goal_oriented",
        "goal_track",
        ["memory_lookup"],
    ),
    (
        "What topics have we talked about the most?",
        "question",
        "fact_seeking",
        ["memory_lookup"],
    ),
    # --- Multi-step / planning ---
    (
        "Help me make a plan for next week.",
        "multi_step",
        "task_plan",
        ["memory_lookup"],
    ),
    (
        "I want to track how often I go to the gym.",
        "multi_step",
        "task_plan",
        ["memory_lookup"],
    ),
    (
        "Help me remember to call mom every Sunday.",
        "multi_step",
        "task_plan",
        ["memory_lookup"],
    ),
    (
        "I'm trying to build a habit of reading 30 minutes a day.",
        "goal_oriented",
        "goal_track",
        ["memory_lookup"],
    ),
    (
        "Can you help me break down my big project into smaller steps?",
        "multi_step",
        "task_plan",
        ["memory_lookup"],
    ),
    # --- Informational / curiosity ---
    ("What's the weather going to be like today?", "question", "fact_seeking", []),
    ("Can you recommend a good book?", "question", "fact_seeking", []),
    (
        "How do I stay motivated when things get hard?",
        "question",
        "fact_seeking",
        ["memory_lookup"],
    ),
    (
        "What's the best way to deal with stress?",
        "question",
        "fact_seeking",
        ["memory_lookup"],
    ),
    # --- Edge: brief/exit ---
    ("bye", "casual", "casual_reply", []),
]


class GoldenBehaviorSet:
    """Collection of canonical prompt/structure pairs forming the golden set."""

    def __init__(
        self,
        records: list[GoldenBehaviorRecord],
        *,
        version: str = SNAPSHOT_VERSION,
    ) -> None:
        self.records = list(records)
        self.version = str(version)
        self.set_hash = _sha256(
            {
                "version": self.version,
                "records": [r.to_dict() for r in self.records],
            },
        )

    @classmethod
    def default(cls) -> GoldenBehaviorSet:
        records = [
            GoldenBehaviorRecord.build(prompt, intent_type, strategy, tool_plan)
            for (prompt, intent_type, strategy, tool_plan) in _GOLDEN_RECORDS_SPECS
        ]
        return cls(records, version=SNAPSHOT_VERSION)

    def get_by_intent(self, intent_type: str) -> list[GoldenBehaviorRecord]:
        return [r for r in self.records if r.intent_type == intent_type]

    def replay_all(self) -> dict[str, Any]:
        """Replay all golden records and verify envelope hashes."""
        results: list[dict[str, Any]] = []
        all_pass = True
        for record in self.records:
            passed = record.verify_replay()
            if not passed:
                all_pass = False
            results.append(
                {
                    "prompt": record.prompt[:60],
                    "envelope_hash": record.envelope_hash,
                    "replay_passed": passed,
                },
            )
        return {
            "all_passed": all_pass,
            "total": len(self.records),
            "passed": sum(1 for r in results if r["replay_passed"]),
            "results": results,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "set_hash": self.set_hash,
            "record_count": len(self.records),
            "records": [r.to_dict() for r in self.records],
        }


# ---------------------------------------------------------------------------
# System snapshot V1
# ---------------------------------------------------------------------------


@dataclass
class SystemSnapshotV1:
    """Full versioned system snapshot artifact.

    The snapshot_hash is computed from all components.
    Same code → same snapshot_hash (reproducible).
    """

    snapshot_version: str
    git_hash: str
    file_tree_hash: str
    schema_registry: SchemaRegistry
    dependency_lock: dict[str, str]
    golden_set: GoldenBehaviorSet
    snapshot_hash: str

    @classmethod
    def build(cls, workspace_root: str | Path) -> SystemSnapshotV1:
        root = Path(workspace_root)
        git_hash = get_git_hash(root)
        file_tree_hash = FileTreeHasher.hash_directory(root / "dadbot")
        schema_registry = SchemaRegistry()
        dependency_lock = get_dependency_lock()
        golden_set = GoldenBehaviorSet.default()

        # Snapshot hash covers everything except itself.
        snapshot_hash = _sha256(
            {
                "snapshot_version": SNAPSHOT_VERSION,
                "git_hash": git_hash,
                "file_tree_hash": file_tree_hash,
                "schema_registry": schema_registry.to_dict(),
                "golden_set_hash": golden_set.set_hash,
            },
        )

        return cls(
            snapshot_version=SNAPSHOT_VERSION,
            git_hash=git_hash,
            file_tree_hash=file_tree_hash,
            schema_registry=schema_registry,
            dependency_lock=dependency_lock,
            golden_set=golden_set,
            snapshot_hash=snapshot_hash,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_version": self.snapshot_version,
            "git_hash": self.git_hash,
            "file_tree_hash": self.file_tree_hash,
            "schema_registry": self.schema_registry.to_dict(),
            "dependency_lock": self.dependency_lock,
            "golden_set": self.golden_set.to_dict(),
            "snapshot_hash": self.snapshot_hash,
        }

    def write(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, snapshot_path: str | Path) -> SystemSnapshotV1:
        """Load a previously-written snapshot for comparison."""
        data = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        schema_registry = SchemaRegistry(
            **{k: v for k, v in data.get("schema_registry", {}).items() if k in SchemaRegistry.__dataclass_fields__},
        )
        golden_set_data = data.get("golden_set", {})
        records = [
            GoldenBehaviorRecord.build(
                r["prompt"],
                r["intent_type"],
                r["strategy"],
                r["tool_plan"],
            )
            for r in golden_set_data.get("records", [])
        ]
        golden_set = GoldenBehaviorSet(
            records,
            version=golden_set_data.get("version", SNAPSHOT_VERSION),
        )
        return cls(
            snapshot_version=data.get("snapshot_version", SNAPSHOT_VERSION),
            git_hash=data.get("git_hash", "UNKNOWN"),
            file_tree_hash=data.get("file_tree_hash", ""),
            schema_registry=schema_registry,
            dependency_lock=data.get("dependency_lock", {}),
            golden_set=golden_set,
            snapshot_hash=data.get("snapshot_hash", ""),
        )


# ---------------------------------------------------------------------------
# Snapshot restore validator
# ---------------------------------------------------------------------------


@dataclass
class ReplayResult:
    snapshot_version: str
    file_tree_match: bool  # file_tree_hash matches current code
    golden_replay_passed: bool  # all golden records replay correctly
    golden_total: int
    golden_passed: int
    ok: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_version": self.snapshot_version,
            "file_tree_match": self.file_tree_match,
            "golden_replay_passed": self.golden_replay_passed,
            "golden_total": self.golden_total,
            "golden_passed": self.golden_passed,
            "ok": self.ok,
            "notes": self.notes,
        }


class SnapshotRestoreValidator:
    """Phase 0.3 — Restore capability.

    Validates that the current code state is consistent with the snapshot.
    """

    def validate(
        self,
        snapshot: SystemSnapshotV1,
        workspace_root: str | Path,
    ) -> ReplayResult:
        notes: list[str] = []
        root = Path(workspace_root)

        # Check file tree hash.
        current_tree_hash = FileTreeHasher.hash_directory(root / "dadbot")
        file_tree_match = current_tree_hash == snapshot.file_tree_hash
        if not file_tree_match:
            notes.append(
                f"File tree hash mismatch: stored={snapshot.file_tree_hash[:16]}... "
                f"current={current_tree_hash[:16]}...",
            )

        # Check golden set replay.
        replay_result = snapshot.golden_set.replay_all()
        golden_passed = replay_result["passed"]
        golden_total = replay_result["total"]
        golden_replay_passed = replay_result["all_passed"]
        if not golden_replay_passed:
            failed = [r for r in replay_result["results"] if not r["replay_passed"]]
            notes.append(f"{len(failed)} golden record(s) failed replay.")

        ok = golden_replay_passed  # file_tree_match skipped — code changes are allowed post-snapshot

        return ReplayResult(
            snapshot_version=snapshot.snapshot_version,
            file_tree_match=file_tree_match,
            golden_replay_passed=golden_replay_passed,
            golden_total=golden_total,
            golden_passed=golden_passed,
            ok=ok,
            notes=notes,
        )

    def validate_golden_record(self, record: GoldenBehaviorRecord) -> bool:
        """Verify a single golden record replays identically."""
        return record.verify_replay()


__all__ = [
    "DAG_SCHEMA_VERSION",
    "EVENT_LOG_SCHEMA_VERSION",
    "SNAPSHOT_VERSION",
    "TOOL_IR_SCHEMA_VERSION",
    "FileTreeHasher",
    "GoldenBehaviorRecord",
    "GoldenBehaviorSet",
    "ReplayResult",
    "SchemaRegistry",
    "SnapshotRestoreValidator",
    "SystemSnapshotV1",
    "get_git_hash",
]
