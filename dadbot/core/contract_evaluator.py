"""
Unified contract evaluator (RULE AUTHORITY).

Consolidates:
  - execution_contract: TurnRequest/Response/AgentState models
  - capability_contracts: Behavioral capability registry
  - invariance_contract: Evaluation-time invariance gates
  - contract_propagation: DAG-based contract validation
  - runtime_adapter: Runtime contract checks

This is the SOLE rule evaluator for the system.
All contract decisions flow through this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Literal, Protocol, TypeAlias

from pydantic import BaseModel, Field

# ============================================================================
# TYPE ALIASES (from execution_contract.py)
# ============================================================================

ChunkCallback: TypeAlias = Callable[[str], Any]
TurnResult: TypeAlias = tuple[str | None, bool]

# ============================================================================
# SECTION 1: EXECUTION CONTRACT (from execution_contract.py)
# ============================================================================


class ExecutionMode(str, Enum):
    """How the orchestrator executes the turn."""
    LIVE = "live"
    REPLAY = "replay"
    RECOVERY = "recovery"


class TurnDelivery(str, Enum):
    """How the turn result is delivered to the caller."""
    SYNC = "sync"
    ASYNC = "async"
    STREAM = "stream"
    STREAM_ASYNC = "stream_async"


class UserInput(BaseModel):
    """User-facing input to the system."""
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageRecord(BaseModel):
    """A single message in turn history."""
    role: str  # "user" | "assistant"
    content: str
    timestamp: str | None = None


class NodeRecord(BaseModel):
    """A node executed during this turn."""
    node_id: str
    status: str  # "pending" | "running" | "completed" | "failed"
    output: Any | None = None
    error: str | None = None


class ToolRecord(BaseModel):
    """A tool invoked during this turn."""
    tool_name: str
    args: dict[str, Any]
    result: Any | None = None
    error: str | None = None


class TurnRequest(BaseModel):
    """Request to execute a turn."""
    user_input: UserInput
    execution_mode: ExecutionMode = ExecutionMode.LIVE
    turn_delivery: TurnDelivery = TurnDelivery.SYNC
    request_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnResponse(BaseModel):
    """Response from turn execution."""
    response_text: str
    request_id: str | None = None
    execution_mode: ExecutionMode
    messages: list[MessageRecord] = Field(default_factory=list)
    nodes_executed: list[NodeRecord] = Field(default_factory=list)
    tools_invoked: list[ToolRecord] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def as_result(self) -> dict[str, Any]:
        """Materialize response for external consumption."""
        return {
            "response": self.response_text,
            "request_id": self.request_id,
            "execution_mode": self.execution_mode.value,
            "metadata": self.metadata,
        }


class AgentState(BaseModel):
    """State snapshot of the agent during execution."""
    turn_id: str
    execution_mode: ExecutionMode
    nodes_visited: list[str] = Field(default_factory=list)
    tools_invoked: list[str] = Field(default_factory=list)
    messages: list[MessageRecord] = Field(default_factory=list)
    invariance_hash: str | None = None

    def recompute_invariance_hash(self) -> str:
        """Compute hash of current state for determinism verification."""
        import hashlib
        state_str = f"{self.turn_id}:{','.join(self.nodes_visited)}:{','.join(self.tools_invoked)}"
        return hashlib.sha256(state_str.encode()).hexdigest()


class TurnExecutor(Protocol):
    """Protocol for executing a turn."""
    def __call__(self, request: TurnRequest) -> TurnResponse:
        ...


class TurnRuntimeContract(Protocol):
    """Protocol for turn runtime contract enforcement."""
    def validate(self, request: TurnRequest) -> list[str]:
        """Return list of violations, empty if valid."""
        ...


def live_turn_request(user_text: str, metadata: dict[str, Any] | None = None) -> TurnRequest:
    """Factory for creating a standard live turn request."""
    return TurnRequest(
        user_input=UserInput(text=user_text, metadata=metadata or {}),
        execution_mode=ExecutionMode.LIVE,
        turn_delivery=TurnDelivery.SYNC,
    )


# ============================================================================
# SECTION 2: CAPABILITY CONTRACTS (from capability_contracts.py)
# ============================================================================

CAPABILITY_CONTRACTS: dict[str, dict[str, Any]] = {
    "temporal_ordering": {
        "required_stages": ["plan", "execute", "recover"],
        "runtime_enforcement": True,
        "description": "Turns must follow temporal ordering (plan → execute → recover)",
    },
    "mutation_safety": {
        "rule": "Only recovery phase mutates graph state",
        "runtime_enforcement": True,
        "description": "Graph mutations only allowed in RECOVERY execution mode",
    },
    "deterministic_replay": {
        "runtime_enforcement": False,
        "test_enforcement": True,
        "description": "Replay execution must produce identical output to original",
    },
    "save_node_single_execution": {
        "runtime_enforcement": True,
        "description": "Save nodes must execute exactly once per turn",
    },
    "capability_audit_emission": {
        "runtime_enforcement": True,
        "description": "All capability checks must emit audit events",
    },
}


def capability_contracts() -> dict[str, dict[str, Any]]:
    """Get the capability contract registry."""
    return CAPABILITY_CONTRACTS.copy()


# ============================================================================
# SECTION 3: INVARIANCE CONTRACT (from invariance_contract.py)
# ============================================================================

BoundaryName = Literal["boot", "registry", "orchestrator"]


@dataclass
class EvaluationContract:
    """Contract for evaluation-time invariance gates."""
    behavioral_invariance: dict[str, Any]
    envelope_invariance: dict[str, Any]
    boundaries: dict[BoundaryName, dict[str, Any]]


@dataclass
class BoundaryComplianceDeclaration:
    """Declaration of compliance for a system boundary."""
    boundary: BoundaryName
    compliant: bool
    notes: list[str]


def get_evaluation_contract() -> EvaluationContract:
    """Get the pre-defined evaluation contract."""
    return EvaluationContract(
        behavioral_invariance={
            "gate": "CORE",
            "correctness_critical": True,
            "description": "Behavioral invariance enforcement at evaluation time",
            "signals": ["replay_hash", "tool_trace_hash"],
            "must_not_depend_on": [
                "determinism_manifest_hash",
                "lock_hash",
                "PYTEST_CURRENT_TEST",
            ],
        },
        envelope_invariance={
            "gate": "SECONDARY",
            "correctness_critical": False,
            "description": "Envelope (request/response) invariance",
        },
        boundaries={
            "boot": {
                "description": "Kernel bootstrap phase",
                "invariance_gates_required": ["boot_completeness"],
            },
            "registry": {
                "description": "Capability registry phase",
                "invariance_gates_required": ["registry_consistency"],
            },
            "orchestrator": {
                "description": "Orchestrator execution phase",
                "invariance_gates_required": ["execution_correctness"],
            },
        },
    )


def evaluation_contract_payload() -> dict[str, Any]:
    """Get contract payload for serialization."""
    contract = get_evaluation_contract()
    return {
        "behavioral_invariance": contract.behavioral_invariance,
        "envelope_invariance": contract.envelope_invariance,
        "boundaries": contract.boundaries,
    }


def evaluation_contract_hash() -> str:
    """Get hash of evaluation contract for verification."""
    import hashlib
    payload = str(evaluation_contract_payload())
    return hashlib.sha256(payload.encode()).hexdigest()


def build_boundary_compliance(
    boundary: BoundaryName, *, compliant: bool, notes: list[str]
) -> BoundaryComplianceDeclaration:
    """Build a boundary compliance declaration."""
    return BoundaryComplianceDeclaration(boundary=boundary, compliant=compliant, notes=notes)


def resolve_boundary_declaration(
    boundary: BoundaryName, declaration: BoundaryComplianceDeclaration
) -> BoundaryComplianceDeclaration:
    """Resolve (validate + return) a boundary compliance declaration."""
    if declaration.boundary != boundary:
        raise ValueError(f"Boundary mismatch: {declaration.boundary} != {boundary}")
    return declaration


# ============================================================================
# SECTION 4: CONTRACT PROPAGATION (from contract_propagation.py)
# ============================================================================

@dataclass
class ContractNode:
    """Node in contract validation DAG."""
    contract_id: str
    version: str
    upstream_dependencies: list[str] = field(default_factory=list)
    downstream_consumers: list[str] = field(default_factory=list)
    validator_fn: Callable[[], list[str]] | None = None
    description: str = ""


@dataclass
class ContractValidationResult:
    """Result of validating a contract."""
    contract_id: str
    version: str
    violations: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Whether this contract passed validation."""
        return len(self.violations) == 0


class ContractPropagationMap:
    """DAG-based contract validation and propagation."""

    def __init__(self):
        self._contracts: dict[str, ContractNode] = {}
        self._validation_cache: dict[str, ContractValidationResult] = {}

    def register(self, node: ContractNode) -> None:
        """Register a contract node."""
        self._contracts[node.contract_id] = node

    def get(self, contract_id: str) -> ContractNode | None:
        return self._contracts.get(str(contract_id or ""))

    def all_ids(self) -> list[str]:
        return list(self._contracts.keys())

    def mark_changed(self, contract_id: str) -> list[ContractValidationResult]:
        """Mark contract as changed, revalidate downstream."""
        start = str(contract_id or "")
        if start not in self._contracts:
            return []
        results: list[ContractValidationResult] = []
        queue: list[str] = [start]
        visited: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            node = self._contracts.get(current)
            if node is None:
                continue
            results.append(self._validate_contract(node))
            for downstream in list(node.downstream_consumers or []):
                if downstream not in visited:
                    queue.append(str(downstream))
        return results

    def revalidate_all(self) -> list[ContractValidationResult]:
        """Revalidate all contracts."""
        self._validation_cache.clear()
        results = []

        def _depth(contract_id: str, seen: set[str] | None = None) -> int:
            seen = set(seen or set())
            if contract_id in seen:
                return 0
            seen.add(contract_id)
            node = self._contracts.get(contract_id)
            if node is None:
                return 0
            upstream = [str(dep) for dep in list(node.upstream_dependencies or []) if str(dep)]
            if not upstream:
                return 0
            return 1 + max(_depth(dep, seen.copy()) for dep in upstream)

        for contract_id in sorted(self._contracts.keys(), key=lambda cid: (_depth(cid), cid)):
            node = self._contracts[contract_id]
            result = self._validate_contract(node)
            results.append(result)
        return results

    def _validate_contract(self, node: ContractNode) -> ContractValidationResult:
        """Validate a single contract node."""
        if node.contract_id in self._validation_cache:
            return self._validation_cache[node.contract_id]

        violations = []
        if node.validator_fn:
            try:
                violations = node.validator_fn()
            except Exception as e:
                violations = [f"Validator error: {str(e)}"]

        result = ContractValidationResult(
            contract_id=node.contract_id,
            version=node.version,
            violations=violations,
        )
        self._validation_cache[node.contract_id] = result
        return result


def _validate_runtime_contract() -> list[str]:
    """Validator for runtime contract."""
    return []  # Placeholder


def _validate_persistence_contract() -> list[str]:
    """Validator for persistence contract."""
    return []  # Placeholder


def _validate_capability(
    contracts: dict[str, dict[str, Any]], required_keys: list[str]
) -> list[str]:
    """Validator for capability contracts."""
    violations = []
    for key in required_keys:
        if key not in contracts:
            violations.append(f"Required capability '{key}' not found")
    return violations


def build_dadbot_contract_map() -> ContractPropagationMap:
    """Build pre-wired DAG of DadBot contracts."""
    contract_map = ContractPropagationMap()

    # Register runtime contract
    contract_map.register(
        ContractNode(
            contract_id="runtime_contract",
            version="1.0",
            downstream_consumers=["persistence_contract"],
            validator_fn=_validate_runtime_contract,
            description="Core turn execution contract",
        )
    )

    # Register persistence contract
    contract_map.register(
        ContractNode(
            contract_id="persistence_contract",
            version="1.0",
            upstream_dependencies=["runtime_contract"],
            downstream_consumers=["graph_integrity_contract"],
            validator_fn=_validate_persistence_contract,
            description="Persistence layer contract",
        )
    )

    # Register graph integrity contract
    contract_map.register(
        ContractNode(
            contract_id="graph_integrity_contract",
            version="1.0",
            upstream_dependencies=["persistence_contract"],
            downstream_consumers=["determinism_boundary_contract"],
            validator_fn=lambda: _validate_capability(
                capability_contracts(),
                ["save_node_single_execution", "temporal_ordering"],
            ),
            description="Graph structural and stage integrity contract",
        )
    )

    # Register determinism boundary contract
    contract_map.register(
        ContractNode(
            contract_id="determinism_boundary_contract",
            version="1.0",
            upstream_dependencies=["graph_integrity_contract"],
            validator_fn=lambda: [],
            description="Determinism/replay boundary contract",
        )
    )

    return contract_map


# ============================================================================
# SECTION 5: RUNTIME ADAPTER (from runtime_adapter.py)
# ============================================================================

class AppRuntimeContract(Protocol):
    """Protocol for app runtime contract enforcement."""
    def __call__(self) -> bool:
        """Return True if runtime contract satisfied."""
        ...


def runtime_contract_errors() -> list[str]:
    """Check runtime contract and return list of violations."""
    errors = []
    # Add runtime checks here
    return errors


# ============================================================================
# PUBLIC API (this module is the RULE AUTHORITY)
# ============================================================================

__all__ = [
    # Type aliases
    "ChunkCallback",
    "TurnResult",
    # Execution contract
    "ExecutionMode",
    "TurnDelivery",
    "UserInput",
    "MessageRecord",
    "NodeRecord",
    "ToolRecord",
    "TurnRequest",
    "TurnResponse",
    "AgentState",
    "TurnExecutor",
    "TurnRuntimeContract",
    "live_turn_request",
    # Capability contracts
    "CAPABILITY_CONTRACTS",
    "capability_contracts",
    # Invariance contract
    "BoundaryName",
    "EvaluationContract",
    "BoundaryComplianceDeclaration",
    "get_evaluation_contract",
    "evaluation_contract_payload",
    "evaluation_contract_hash",
    "build_boundary_compliance",
    "resolve_boundary_declaration",
    # Contract propagation
    "ContractNode",
    "ContractValidationResult",
    "ContractPropagationMap",
    "build_dadbot_contract_map",
    # Runtime adapter
    "AppRuntimeContract",
    "runtime_contract_errors",
]
