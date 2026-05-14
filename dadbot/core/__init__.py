"""DadBot core package for graph-based orchestration primitives.

The package surface stays stable, but the exports are resolved lazily so that
lightweight submodule imports do not pull in the entire orchestration stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS: dict[str, list[str]] = {
    ".authorization": [
        "AuthorizationError",
        "Capability",
        "CapabilitySet",
        "CapabilityToken",
        "SessionAuthorizationPolicy",
        "TenantBoundary",
        "authorize_write",
    ],
    ".compaction": [
        "ArchiveTier",
        "CompactionPolicy",
        "EventCompactor",
    ],
    ".control_plane": [
        "ExecutionControlPlane",
        "ExecutionJob",
        "Scheduler",
        "SessionRegistry",
    ],
    ".durable_checkpoint": [
        "DurableCheckpoint",
        "CheckpointIntegrityError",
    ],
    ".durability": [
        "AtomicWriteUnit",
        "CRC32LineCodec",
        "FileLockMutex",
        "UNIT_BEGIN_TYPE",
        "UNIT_COMMIT_TYPE",
    ],
    ".event_schema": [
        "CURRENT_SCHEMA_VERSION",
        "EventSchemaMigration",
        "EventSchemaMigrator",
        "get_migrator",
        "migrate_event",
        "stamp_schema_version",
    ],
    ".execution_lease": [
        "ExecutionLease",
        "LeaseConflictError",
        "WorkerIdentity",
    ],
    ".execution_ledger": [
        "ExecutionLedger",
        "WriteBoundaryGuard",
        "WriteBoundaryViolationError",
    ],
    ".fault_injection": [
        "CompensatingActionRequired",
        "ErrorClassification",
        "ErrorClassifier",
        "FaultBoundary",
        "FaultInjector",
        "RetryableError",
        "RetryPolicy",
        "TerminalError",
        "classify_error",
        "get_fault_injector",
        "register_error_class",
    ],
    ".graph": [
        "GraphNode",
        "TurnContext",
        "TurnGraph",
    ],
    ".idempotency_boundary": [
        "IdempotencyBoundary",
    ],
    ".invariant_gate": [
        "InvariantGate",
        "InvariantViolationError",
    ],
    ".ledger_backend": [
        "BatchWriteBackend",
        "CRCFileWALLedgerBackend",
        "EventualConsistencyBackend",
        "FileWALLedgerBackend",
        "InMemoryLedgerBackend",
        "LedgerBackend",
        "SequenceValidator",
        "StrongConsistencyBackend",
    ],
    ".nodes": [
        "HealthNode",
        "InferenceNode",
        "MemoryNode",
        "ReflectionNode",
        "SafetyNode",
        "SaveNode",
    ],
    ".observability": [
        "CorrelationContext",
        "EventStreamExporter",
        "MetricsSink",
        "ReplayDebugger",
        "StructuredLogger",
        "TracingContext",
        "configure_exporter",
        "get_exporter",
        "get_metrics",
        "get_tracer",
    ],
    ".orchestrator": [
        "DadBotOrchestrator",
    ],
    ".recovery_manager": [
        "RecoveryManager",
        "StartupReconciliationError",
    ],
    ".snapshot_engine": [
        "SnapshotEngine",
    ],
}

_EXPORT_MAP = {symbol: module_name for module_name, symbols in _EXPORT_GROUPS.items() for symbol in symbols}

__all__ = [
    # authorization
    "AuthorizationError",
    "Capability",
    "CapabilitySet",
    "CapabilityToken",
    "SessionAuthorizationPolicy",
    "TenantBoundary",
    "authorize_write",
    # compaction
    "ArchiveTier",
    "CompactionPolicy",
    "EventCompactor",
    # control plane
    "GraphNode",
    "SessionRegistry",
    "ReflectionNode",
    "ExecutionJob",
    "Scheduler",
    "ExecutionControlPlane",
    # checkpoint
    "DurableCheckpoint",
    "CheckpointIntegrityError",
    # durability
    "AtomicWriteUnit",
    "CRC32LineCodec",
    "FileLockMutex",
    "UNIT_BEGIN_TYPE",
    "UNIT_COMMIT_TYPE",
    # event schema
    "CURRENT_SCHEMA_VERSION",
    "EventSchemaMigration",
    "EventSchemaMigrator",
    "get_migrator",
    "migrate_event",
    "stamp_schema_version",
    # lease
    "ExecutionLease",
    "LeaseConflictError",
    "WorkerIdentity",
    # ledger
    "ExecutionLedger",
    "WriteBoundaryGuard",
    "WriteBoundaryViolationError",
    # fault injection
    "CompensatingActionRequired",
    "ErrorClassification",
    "ErrorClassifier",
    "FaultBoundary",
    "FaultInjector",
    "RetryableError",
    "RetryPolicy",
    "TerminalError",
    "classify_error",
    "get_fault_injector",
    "register_error_class",
    # idempotency
    "IdempotencyBoundary",
    # invariant gate
    "InvariantGate",
    "InvariantViolationError",
    # ledger backend
    "BatchWriteBackend",
    "CRCFileWALLedgerBackend",
    "EventualConsistencyBackend",
    "FileWALLedgerBackend",
    "InMemoryLedgerBackend",
    "LedgerBackend",
    "SequenceValidator",
    "StrongConsistencyBackend",
    # nodes
    "HealthNode",
    "MemoryNode",
    "InferenceNode",
    "SafetyNode",
    "SaveNode",
    # observability
    "CorrelationContext",
    "EventStreamExporter",
    "MetricsSink",
    "ReplayDebugger",
    "StructuredLogger",
    "TracingContext",
    "configure_exporter",
    "get_exporter",
    "get_metrics",
    "get_tracer",
    # orchestrator
    "DadBotOrchestrator",
    # recovery
    "RecoveryManager",
    "StartupReconciliationError",
    # snapshot
    "SnapshotEngine",
    # graph
    "TurnContext",
    "TurnGraph",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MAP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
