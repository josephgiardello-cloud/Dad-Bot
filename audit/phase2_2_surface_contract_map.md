# Phase 2.2 Surface Contract Map (True Baseline)

Generated for safe-mode pruning before any removals.

## 1) Public Exports Snapshot

### 1.1 dadbot.core.graph
Notes:
- This module does not define an explicit __all__.
- Practical public surface is defined by top-level bindings, especially explicit re-export import blocks and test imports.

Re-export blocks (explicitly public by module-level binding):
- Temporal/types: TurnPhase, TurnTemporalAxis, VirtualClock, _PHASE_ORDER
- Trace/op types: ExecutionTraceEvent, GoalMutationOp, LedgerMutationOp, MemoryMutationOp, MutationKind, MutationTransactionRecord, MutationTransactionStatus, NodeType, RelationshipMutationOp, StageTrace, _json_safe
- Mutation primitives: SAVE_NODE_COMMIT_CONTRACT, MutationGuard, MutationIntent, MutationQueue
- Context/state: TurnContext, TurnFidelity
- Pipeline node stubs: ContextBuilderNode, GraphNode, HealthNode, InferenceNode, MemoryNode, ReflectionNode, SafetyNode, SaveNode, TemporalNode
- UX: TurnHealthState, TurnUxProjector

Primary class/function surface in file:
- TurnGraph
- FatalTurnError, TurnFailureSeverity, KernelRejectionSemantics, PersistenceServiceContract, ExecutionPolicyEngine, StagePhaseMappingPolicy (re-exported from execution_policy)

### 1.2 dadbot.core.control_plane
Top-level class surface:
- ExecutionJob
- SchedulerOptions
- ControlPlaneOptions
- SessionRegistry
- Scheduler
- ExecutionControlPlane

Compat-relevant imported symbols (module-level availability):
- InMemoryExecutionLedger
- LedgerReader
- LedgerWriter

### 1.3 dadbot.core.contracts.__init__
Explicit __all__ contract:
- CANONICAL_MUTATION_VOCAB
- ExecutionContextCarrier
- ensure_valid_mutation_op
- require_turn_context

Resolution mechanism:
- Lazy module-level __getattr__ dispatches these names.

## 2) Node Entrypoints Snapshot

### 2.1 Canonical graph pipeline node stubs (dadbot.core.graph_pipeline_nodes)
Entrypoints:
- GraphNode.run(registry, ctx)
- GraphNode.execute(registry, turn_context)
- HealthNode.execute(registry, turn_context)
- ContextBuilderNode.execute(registry, turn_context)
- InferenceNode.execute(registry, turn_context)
- SafetyNode.execute(registry, turn_context)
- SaveNode.execute(registry, turn_context)
- TemporalNode.execute(registry, turn_context)
- ReflectionNode.execute(registry, turn_context)

Alias:
- MemoryNode = ContextBuilderNode

### 2.2 Production node implementations (dadbot.core.nodes)
Observed entrypoints in canonical path and adjacent runtime path:
- HealthNode.run(context)
- ContextBuilderNode.run(context)
- MemoryNode.run(context)
- ReflectionNode.run(context)
- SaveNode.run(context)

## 3) Ledger Entrypoints Snapshot

### 3.1 Writer (dadbot.core.ledger_writer.LedgerWriter)
Public write/append interface:
- write_event(...)
- append_session_bound(...)
- append_job_submitted(...)
- append_job_queued(...)
- append_job_started(...)
- append_job_completed(...)
- append_job_failed(...)
- append_runtime_witness(...)

### 3.2 Reader (dadbot.core.ledger_reader.LedgerReader)
Public read/query interface:
- events()
- events_for_job(job_id)
- is_terminal(job_id)

### 3.3 Control-plane ledger touchpoints
- ExecutionControlPlane.ledger_events()
- ExecutionControlPlane.boot_reconcile()
- Scheduler uses LedgerWriter + LedgerReader for queue lifecycle

## 4) Test-Imported Symbols (Authoritative Keep Set)

Source: AST scan of tests importing from dadbot.core.graph and dadbot.core.control_plane.

### 4.1 Imported from dadbot.core.graph
- ContextBuilderNode
- FatalTurnError
- HealthNode
- InferenceNode
- KernelRejectionSemantics
- LedgerMutationOp
- MutationGuard
- MutationIntent
- MutationKind
- MutationQueue
- ReflectionNode
- SafetyNode
- SaveNode
- TemporalNode
- TurnContext
- TurnFidelity
- TurnGraph
- TurnPhase
- TurnTemporalAxis
- VirtualClock
- _json_safe

### 4.2 Imported from dadbot.core.control_plane
- ExecutionControlPlane
- ExecutionJob
- InMemoryExecutionLedger
- LedgerReader
- LedgerWriter
- Scheduler
- SessionRegistry

### 4.3 Imported from dadbot.core.contracts
- No direct test imports found.

## 5) Stabilization Bridge Freeze (Non-Permanent API)

The following are explicitly classified as temporary stabilization bridge layer and must not be promoted as permanent public surface:
- Dual node-call signature handling in TurnGraph node execution path ((ctx) vs (registry, ctx))
- Lazy __getattr__ export shim in dadbot.core.contracts.__init__
- Legacy compatibility import exposure of LedgerWriter in dadbot.core.control_plane
- Graph re-export blocks preserving legacy import paths during module split

## 6) Phase 2.2 Pruning Order Guardrail

Required order:
1. Leaf utilities (safe)
2. Unused helpers (vulture-confirmed)
3. Internal shims (not test-facing)
4. Re-export cleanup (last)

Current lock decision:
- Keep all symbols listed in Sections 4.1 and 4.2 through early pruning batches.
- Treat Section 5 items as bridge-only and eligible for later reduction once tests and imports are migrated.
