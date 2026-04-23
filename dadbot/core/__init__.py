"""DadBot core package for graph-based orchestration primitives."""

from .authorization import (
    AuthorizationError,
    Capability,
    CapabilitySet,
    CapabilityToken,
    SessionAuthorizationPolicy,
    TenantBoundary,
    authorize_write,
)
from .compaction import ArchiveTier, CompactionPolicy, EventCompactor
from .control_plane import ExecutionControlPlane, ExecutionJob, Scheduler, SessionRegistry
from .durable_checkpoint import DurableCheckpoint, CheckpointIntegrityError
from .durability import (
    AtomicWriteUnit,
    CRC32LineCodec,
    FileLockMutex,
    UNIT_BEGIN_TYPE,
    UNIT_COMMIT_TYPE,
)
from .event_schema import (
    CURRENT_SCHEMA_VERSION,
    EventSchemaMigration,
    EventSchemaMigrator,
    get_migrator,
    migrate_event,
    stamp_schema_version,
)
from .execution_lease import ExecutionLease, LeaseConflictError, WorkerIdentity
from .execution_ledger import ExecutionLedger, WriteBoundaryGuard, WriteBoundaryViolationError
from .fault_injection import (
    CompensatingActionRequired,
    ErrorClassification,
    ErrorClassifier,
    FaultBoundary,
    FaultInjector,
    RetryableError,
    RetryPolicy,
    TerminalError,
    classify_error,
    get_fault_injector,
    register_error_class,
)
from .graph import GraphNode, TurnContext, TurnGraph
from .idempotency_boundary import IdempotencyBoundary
from .invariant_gate import InvariantGate, InvariantViolationError
from .ledger_backend import (
    BatchWriteBackend,
    CRCFileWALLedgerBackend,
    EventualConsistencyBackend,
    FileWALLedgerBackend,
    InMemoryLedgerBackend,
    LedgerBackend,
    SequenceValidator,
    StrongConsistencyBackend,
)
from .nodes import HealthNode, InferenceNode, MemoryNode, SafetyNode, SaveNode
from .observability import (
    CorrelationContext,
    EventStreamExporter,
    MetricsSink,
    ReplayDebugger,
    StructuredLogger,
    TracingContext,
    configure_exporter,
    get_exporter,
    get_metrics,
    get_tracer,
)
from .orchestrator import DadBotOrchestrator
from .recovery_manager import RecoveryManager, StartupReconciliationError
from .snapshot_engine import SnapshotEngine

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
