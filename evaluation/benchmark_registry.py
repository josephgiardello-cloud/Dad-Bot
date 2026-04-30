"""Benchmark Registry — Phase 4D Stability Layer.

Purpose:
  Freeze benchmark runs into immutable snapshots so that:
    1. Any run can be replayed or compared exactly
    2. Scoring function changes are detected (hash drift)
    3. Scenario definition changes are detected (version hash)
    4. Orchestrator version is tracked per snapshot

Snapshot storage:
  evaluation/snapshots/<snapshot_id>.json
  evaluation/snapshots/index.json  (fast lookup)

Snapshot ID format:
  bench-<YYYYMMDD-HHMMSS>-<content_hash[:8]>

Usage:
    from evaluation.benchmark_registry import BenchmarkRegistry

    registry = BenchmarkRegistry()
    snap_id = registry.save(scores=scores, metadata={"run_label": "phase4b"})
    snap = registry.load(snap_id)
    latest = registry.latest()
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VersionManifest:
    """Hashes that uniquely identify the evaluation stack at snapshot time."""

    scoring_engine_hash: str  # hash of tests/scoring_engine.py
    trace_schema_hash: str  # hash of tests/trace_schema.py
    scenario_suite_hash: str  # hash of tests/scenario_suite.py
    gold_set_hash: str  # hash of evaluation/gold_set.py
    orchestrator_hash: str  # hash of dadbot/core/orchestrator.py; "mock" if no orchestrator

    def matches(self, other: VersionManifest) -> bool:
        return (
            self.scoring_engine_hash == other.scoring_engine_hash
            and self.trace_schema_hash == other.trace_schema_hash
            and self.scenario_suite_hash == other.scenario_suite_hash
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "scoring_engine_hash": self.scoring_engine_hash,
            "trace_schema_hash": self.trace_schema_hash,
            "scenario_suite_hash": self.scenario_suite_hash,
            "gold_set_hash": self.gold_set_hash,
            "orchestrator_hash": self.orchestrator_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> VersionManifest:
        return cls(
            scoring_engine_hash=str(d.get("scoring_engine_hash") or ""),
            trace_schema_hash=str(d.get("trace_schema_hash") or ""),
            scenario_suite_hash=str(d.get("scenario_suite_hash") or ""),
            gold_set_hash=str(d.get("gold_set_hash") or ""),
            orchestrator_hash=str(d.get("orchestrator_hash") or ""),
        )


@dataclass
class BenchmarkSnapshot:
    """Immutable record of one complete benchmark run."""

    snapshot_id: str
    created_at: str  # ISO-8601 UTC
    run_label: str  # human-readable tag
    execution_mode: str  # "mock" | "orchestrator"

    version_manifest: VersionManifest

    # Serialized CapabilityScore.to_dict() for each scenario
    scores: list[dict[str, Any]] = field(default_factory=list)

    # Per-category aggregate scores
    category_aggregates: dict[str, float] = field(default_factory=dict)

    # Optional calibration state snapshot
    calibration_applied: bool = False
    calibration_run_count: int = 0

    # Metadata blob for extensibility
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def scenario_count(self) -> int:
        return len(self.scores)

    @property
    def pass_rate(self) -> float:
        if not self.scores:
            return 0.0
        passed = sum(
            1 for s in self.scores if (s.get("scoring") or {}).get("success", False) or s.get("overall", 0.0) >= 0.5
        )
        return passed / len(self.scores)

    def get_score(self, scenario_name: str) -> dict[str, Any] | None:
        for s in self.scores:
            if s.get("scenario") == scenario_name:
                return s
        return None

    def overall_average(self) -> float:
        overalls = [float(s.get("overall") or 0.0) for s in self.scores if s.get("overall") is not None]
        if not overalls:
            return 0.0
        return sum(overalls) / len(overalls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "run_label": self.run_label,
            "execution_mode": self.execution_mode,
            "version_manifest": self.version_manifest.to_dict(),
            "scores": self.scores,
            "category_aggregates": self.category_aggregates,
            "calibration_applied": self.calibration_applied,
            "calibration_run_count": self.calibration_run_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkSnapshot:
        return cls(
            snapshot_id=str(d.get("snapshot_id") or ""),
            created_at=str(d.get("created_at") or ""),
            run_label=str(d.get("run_label") or ""),
            execution_mode=str(d.get("execution_mode") or "mock"),
            version_manifest=VersionManifest.from_dict(d.get("version_manifest") or {}),
            scores=list(d.get("scores") or []),
            category_aggregates=dict(d.get("category_aggregates") or {}),
            calibration_applied=bool(d.get("calibration_applied", False)),
            calibration_run_count=int(d.get("calibration_run_count") or 0),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass
class SnapshotIndexEntry:
    """Lightweight index entry for fast lookup."""

    snapshot_id: str
    created_at: str
    run_label: str
    execution_mode: str
    scenario_count: int
    overall_average: float
    scoring_engine_hash: str


# ---------------------------------------------------------------------------
# Version hashing
# ---------------------------------------------------------------------------

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def _file_hash(rel_path: str) -> str:
    """SHA-256 of a workspace file (first 12 hex chars)."""
    path = _WORKSPACE_ROOT / rel_path
    if not path.exists():
        return "absent"
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:12]


def build_version_manifest(execution_mode: str = "mock") -> VersionManifest:
    """Hash all evaluation-stack source files to detect drift."""
    orchestrator_hash = _file_hash("dadbot/core/orchestrator.py") if execution_mode == "orchestrator" else "mock"
    return VersionManifest(
        scoring_engine_hash=_file_hash("tests/scoring_engine.py"),
        trace_schema_hash=_file_hash("tests/trace_schema.py"),
        scenario_suite_hash=_file_hash("tests/scenario_suite.py"),
        gold_set_hash=_file_hash("evaluation/gold_set.py"),
        orchestrator_hash=orchestrator_hash,
    )


# ---------------------------------------------------------------------------
# Snapshot ID generation
# ---------------------------------------------------------------------------


def _make_snapshot_id(scores: list[dict]) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    payload = json.dumps(scores, sort_keys=True, default=str)
    content_hash = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"bench-{ts}-{content_hash}"


# ---------------------------------------------------------------------------
# Benchmark Registry
# ---------------------------------------------------------------------------


class BenchmarkRegistry:
    """Stores and retrieves benchmark snapshots from disk.

    Thread-safety: single-writer assumed; concurrent reads are fine.
    """

    def __init__(self, snapshots_dir: Path | None = None):
        self._dir = snapshots_dir or _WORKSPACE_ROOT / "evaluation" / "snapshots"
        self._index_path = self._dir / "index.json"

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        scores: list[dict[str, Any]],
        execution_mode: str = "mock",
        run_label: str = "",
        calibration_applied: bool = False,
        calibration_run_count: int = 0,
        metadata: dict | None = None,
    ) -> str:
        """Persist a benchmark run. Returns snapshot_id."""
        self._ensure_dir()

        snap_id = _make_snapshot_id(scores)
        manifest = build_version_manifest(execution_mode)

        # Aggregate per-category
        cat_scores: dict[str, list[float]] = {}
        for s in scores:
            cat = str(s.get("category") or "")
            sub = s.get("capability_score") or {}
            if cat and sub.get(cat) and sub[cat].get("score") is not None:
                cat_scores.setdefault(cat, []).append(float(sub[cat]["score"]))
        category_aggregates = {cat: round(sum(vals) / len(vals), 4) for cat, vals in cat_scores.items()}

        snapshot = BenchmarkSnapshot(
            snapshot_id=snap_id,
            created_at=datetime.now(UTC).isoformat(),
            run_label=run_label or snap_id,
            execution_mode=execution_mode,
            version_manifest=manifest,
            scores=scores,
            category_aggregates=category_aggregates,
            calibration_applied=calibration_applied,
            calibration_run_count=calibration_run_count,
            metadata=metadata or {},
        )

        snap_path = self._dir / f"{snap_id}.json"
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2)

        # Update index
        self._update_index(
            SnapshotIndexEntry(
                snapshot_id=snap_id,
                created_at=snapshot.created_at,
                run_label=snapshot.run_label,
                execution_mode=execution_mode,
                scenario_count=len(scores),
                overall_average=snapshot.overall_average(),
                scoring_engine_hash=manifest.scoring_engine_hash,
            )
        )

        return snap_id

    def load(self, snapshot_id: str) -> BenchmarkSnapshot:
        """Load a snapshot by ID. Raises FileNotFoundError if absent."""
        path = self._dir / f"{snapshot_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot_id}")
        with open(path, encoding="utf-8") as f:
            return BenchmarkSnapshot.from_dict(json.load(f))

    def latest(self, execution_mode: str | None = None) -> BenchmarkSnapshot | None:
        """Return the most recently saved snapshot."""
        index = self._load_index()
        entries = list(index.values())
        if execution_mode:
            entries = [e for e in entries if e.get("execution_mode") == execution_mode]
        if not entries:
            return None
        entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        return self.load(entries[0]["snapshot_id"])

    def list_snapshots(self) -> list[SnapshotIndexEntry]:
        """Return all index entries, newest first."""
        index = self._load_index()
        entries = [SnapshotIndexEntry(**e) for e in index.values()]
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries

    def version_changed_since(self, snapshot: BenchmarkSnapshot) -> list[str]:
        """Return list of files whose hash has changed since this snapshot was taken."""
        current = build_version_manifest(snapshot.execution_mode)
        snap_manifest = snapshot.version_manifest
        changed = []
        checks = {
            "tests/scoring_engine.py": (snap_manifest.scoring_engine_hash, current.scoring_engine_hash),
            "tests/trace_schema.py": (snap_manifest.trace_schema_hash, current.trace_schema_hash),
            "tests/scenario_suite.py": (snap_manifest.scenario_suite_hash, current.scenario_suite_hash),
            "evaluation/gold_set.py": (snap_manifest.gold_set_hash, current.gold_set_hash),
        }
        if snapshot.execution_mode == "orchestrator":
            checks["dadbot/core/orchestrator.py"] = (snap_manifest.orchestrator_hash, current.orchestrator_hash)
        for file, (old, new) in checks.items():
            if old != new:
                changed.append(file)
        return changed

    # -----------------------------------------------------------------------
    # Internal index management
    # -----------------------------------------------------------------------

    def _load_index(self) -> dict[str, dict]:
        if not self._index_path.exists():
            return {}
        with open(self._index_path, encoding="utf-8") as f:
            return json.load(f)

    def _update_index(self, entry: SnapshotIndexEntry) -> None:
        index = self._load_index()
        index[entry.snapshot_id] = {
            "snapshot_id": entry.snapshot_id,
            "created_at": entry.created_at,
            "run_label": entry.run_label,
            "execution_mode": entry.execution_mode,
            "scenario_count": entry.scenario_count,
            "overall_average": entry.overall_average,
            "scoring_engine_hash": entry.scoring_engine_hash,
        }
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
