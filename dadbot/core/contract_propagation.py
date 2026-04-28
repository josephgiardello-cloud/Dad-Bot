"""Contract propagation system (Phase 4.2).

When a contract changes, all dependent (downstream) contracts must be
revalidated.  This module provides:

- ``ContractNode``       — a registered contract with upstream/downstream edges
                           and an optional validator callable.
- ``ContractPropagationMap``  — the directed dependency graph; call
                                ``mark_changed`` to trigger cascaded revalidation
                                of every downstream consumer.

The pre-wired DAG for DadBot's Phase 4 contracts is exposed as a ready-made
singleton via ``build_dadbot_contract_map()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ContractNode:
    """A single registered contract in the propagation graph."""

    contract_id: str
    """Unique identifier (e.g. 'runtime_contract', 'persistence_contract')."""

    version: str
    """Semantic version string for the contract definition."""

    upstream_dependencies: List[str] = field(default_factory=list)
    """IDs of contracts this node depends on (must be valid before this one)."""

    downstream_consumers: List[str] = field(default_factory=list)
    """IDs of contracts that consume / depend on this node."""

    validator_fn: Optional[Callable[[], List[str]]] = field(default=None, repr=False)
    """Callable returning a list of violation strings (empty → valid)."""

    description: str = ""


@dataclass
class ContractValidationResult:
    """Outcome of a single contract revalidation pass."""

    contract_id: str
    version: str
    violations: List[str]

    @property
    def valid(self) -> bool:
        return len(self.violations) == 0


# ---------------------------------------------------------------------------
# ContractPropagationMap
# ---------------------------------------------------------------------------

class ContractPropagationMap:
    """Directed dependency graph of Phase 4 contracts.

    Usage
    -----
    ::

        cmap = ContractPropagationMap()
        cmap.register(ContractNode(
            contract_id="runtime_contract",
            version="1.0.0",
            downstream_consumers=["persistence_contract"],
            validator_fn=my_validator,
        ))
        # When a contract changes:
        results = cmap.mark_changed("runtime_contract")
        # results is a list of ContractValidationResult for every affected node
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, ContractNode] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, node: ContractNode) -> None:
        """Add a contract node to the propagation graph."""
        if node.contract_id in self._nodes:
            existing = self._nodes[node.contract_id]
            logger.warning(
                "[Phase 4.2] Overwriting contract '%s' (was v%s, now v%s)",
                node.contract_id, existing.version, node.version,
            )
        self._nodes[node.contract_id] = node

    def get(self, contract_id: str) -> Optional[ContractNode]:
        return self._nodes.get(contract_id)

    def all_ids(self) -> List[str]:
        return list(self._nodes.keys())

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def mark_changed(self, contract_id: str) -> List[ContractValidationResult]:
        """Trigger revalidation of *contract_id* and all downstream consumers.

        Returns a list of ``ContractValidationResult`` for every node visited
        (breadth-first, including the originating node itself).
        """
        if contract_id not in self._nodes:
            logger.warning(
                "[Phase 4.2] mark_changed called for unknown contract '%s'",
                contract_id,
            )
            return []

        visited: list[str] = []
        queue: list[str] = [contract_id]
        seen: set[str] = set()

        while queue:
            current_id = queue.pop(0)
            if current_id in seen:
                continue
            seen.add(current_id)
            visited.append(current_id)

            node = self._nodes.get(current_id)
            if node is None:
                continue
            for consumer_id in node.downstream_consumers:
                if consumer_id not in seen:
                    queue.append(consumer_id)

        results: list[ContractValidationResult] = []
        for cid in visited:
            results.append(self._revalidate(cid))

        return results

    def revalidate_all(self) -> List[ContractValidationResult]:
        """Revalidate every registered contract in topological order."""
        order = self._topo_sort()
        return [self._revalidate(cid) for cid in order]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _revalidate(self, contract_id: str) -> ContractValidationResult:
        node = self._nodes.get(contract_id)
        if node is None:
            return ContractValidationResult(
                contract_id=contract_id,
                version="unknown",
                violations=[f"Contract '{contract_id}' not registered"],
            )
        violations: list[str] = []
        if node.validator_fn is not None:
            try:
                violations = list(node.validator_fn() or [])
            except Exception as exc:
                violations = [f"Validator raised: {exc}"]

        if violations:
            logger.warning(
                "[Phase 4.2] Contract '%s' v%s revalidated — %d violation(s): %s",
                contract_id, node.version, len(violations), violations,
            )
        else:
            logger.debug(
                "[Phase 4.2] Contract '%s' v%s revalidated — OK", contract_id, node.version
            )

        return ContractValidationResult(
            contract_id=contract_id,
            version=node.version,
            violations=violations,
        )

    def _topo_sort(self) -> List[str]:
        """Return contract IDs in dependency order (upstream before downstream)."""
        visited: set[str] = set()
        order: list[str] = []

        def _visit(cid: str) -> None:
            if cid in visited:
                return
            visited.add(cid)
            node = self._nodes.get(cid)
            if node:
                for dep in node.upstream_dependencies:
                    _visit(dep)
            order.append(cid)

        for cid in self._nodes:
            _visit(cid)
        return order


# ---------------------------------------------------------------------------
# Pre-wired DadBot contract DAG
# ---------------------------------------------------------------------------

def build_dadbot_contract_map() -> ContractPropagationMap:
    """Build the canonical Phase 4 contract propagation map.

    DAG shape
    ---------
    runtime_contract
        └─► persistence_contract
                └─► graph_integrity_contract
                        └─► determinism_boundary_contract
    """
    from dadbot.core.capability_contracts import CAPABILITY_CONTRACTS

    cmap = ContractPropagationMap()

    cmap.register(ContractNode(
        contract_id="runtime_contract",
        version="1.0.0",
        upstream_dependencies=[],
        downstream_consumers=["persistence_contract"],
        description="AppRuntimeContract — descriptor/parameter/init validation",
        validator_fn=_validate_runtime_contract,
    ))
    cmap.register(ContractNode(
        contract_id="persistence_contract",
        version="1.0.0",
        upstream_dependencies=["runtime_contract"],
        downstream_consumers=["graph_integrity_contract"],
        description="PersistenceServiceContract — method arity validation",
        validator_fn=_validate_persistence_contract,
    ))
    cmap.register(ContractNode(
        contract_id="graph_integrity_contract",
        version="1.0.0",
        upstream_dependencies=["persistence_contract"],
        downstream_consumers=["determinism_boundary_contract"],
        description=(
            "Graph integrity — save_node_single_execution + mutation_safety "
            "from CAPABILITY_CONTRACTS"
        ),
        validator_fn=lambda: _validate_capability(
            CAPABILITY_CONTRACTS, ["save_node_single_execution", "mutation_safety"]
        ),
    ))
    cmap.register(ContractNode(
        contract_id="determinism_boundary_contract",
        version="1.0.0",
        upstream_dependencies=["graph_integrity_contract"],
        downstream_consumers=[],
        description="Determinism boundary — deterministic_replay + temporal_ordering",
        validator_fn=lambda: _validate_capability(
            CAPABILITY_CONTRACTS, ["deterministic_replay", "temporal_ordering"]
        ),
    ))

    return cmap


# ---------------------------------------------------------------------------
# Built-in validators
# ---------------------------------------------------------------------------

def _validate_runtime_contract() -> List[str]:
    """Check that the module-level runtime adapter interface is importable and well-formed."""
    try:
        import importlib

        _m = importlib.import_module("dadbot.runtime_adapter")
        # Verify the expected public API is present and callable.
        if not callable(getattr(_m, "runtime_contract_errors", None)):
            return ["runtime_adapter.runtime_contract_errors not callable"]
        if not callable(getattr(_m, "AppRuntimeContract", None)) and not hasattr(_m, "AppRuntimeContract"):
            return ["runtime_adapter.AppRuntimeContract not found"]
        return []
    except Exception as exc:
        return [f"runtime_adapter import error: {exc}"]


def _validate_persistence_contract() -> List[str]:
    """Check that ExecutionPolicyEngine and PersistenceServiceContract are importable."""
    try:
        from dadbot.core.execution_policy import ExecutionPolicyEngine, PersistenceServiceContract
        if not callable(ExecutionPolicyEngine):
            return ["ExecutionPolicyEngine not callable"]
        if not callable(PersistenceServiceContract):
            return ["PersistenceServiceContract not callable"]
        return []
    except Exception as exc:
        return [f"persistence contract validation error: {exc}"]


def _validate_capability(
    contracts: Dict[str, Any], required_keys: List[str]
) -> List[str]:
    issues: list[str] = []
    for key in required_keys:
        if key not in contracts:
            issues.append(f"Missing CAPABILITY_CONTRACTS entry: '{key}'")
        else:
            entry = contracts[key]
            if not entry.get("runtime_enforcement") and not entry.get("test_enforcement"):
                issues.append(
                    f"Capability contract '{key}' has neither "
                    "runtime_enforcement nor test_enforcement enabled"
                )
    return issues
