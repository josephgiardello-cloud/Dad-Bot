"""Semantic graph validation layer (Phase 4.3).

Each pipeline node declares an *intent contract* describing:

- ``input_intent_schema``    — dict of expected keys in TurnContext.state/metadata
                               that must be present before this node runs.
- ``output_expectation_class``  — string label of what this node is expected to
                                   produce (used in edge compatibility checks).
- ``mutation_constraints``   — list of strings naming mutation rules enforced by
                               this node (e.g. "MutationQueue.drain_only").

``GraphSemanticValidator`` checks node-to-node semantic compatibility before
execution and returns structured violation reports.  It does NOT execute the
graph — it validates the static contract declarations.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NodeIntentContract
# ---------------------------------------------------------------------------


@dataclass
class NodeIntentContract:
    """Semantic execution contract for a single pipeline node."""

    node_name: str
    """Unique name of the node as registered in the TurnGraph."""

    input_intent_schema: dict[str, str] = field(default_factory=dict)
    """Required keys in TurnContext.state/metadata before this node may run.

    Format: ``{key: description}`` — descriptions are human-readable intent
    notes (not type annotations).
    """

    output_expectation_class: str = ""
    """Label for what this node is expected to produce.

    Downstream consumers declare their required ``input_intent_schema`` entries
    which should match what the upstream ``output_expectation_class`` claims to
    provide via ``NodeIntentContract.output_provides``.
    """

    output_provides: list[str] = field(default_factory=list)
    """State/metadata keys this node writes, enabling downstream schema checks."""

    mutation_constraints: list[str] = field(default_factory=list)
    """Semantic mutation rules enforced by this node.

    Examples: ``"MutationQueue.drain_only"``, ``"no_direct_memory_write"``.
    """

    description: str = ""


# ---------------------------------------------------------------------------
# Violation type
# ---------------------------------------------------------------------------


@dataclass
class SemanticViolation:
    """A single semantic incompatibility found during graph validation."""

    upstream_node: str
    downstream_node: str
    missing_key: str | None
    violation_type: str
    detail: str


# ---------------------------------------------------------------------------
# GraphSemanticValidator
# ---------------------------------------------------------------------------


class GraphSemanticValidator:
    """Validates node-to-node semantic compatibility before graph execution.

    Usage
    -----
    ::

        validator = GraphSemanticValidator()
        validator.register_node_contract(NodeIntentContract(
            node_name="temporal",
            output_provides=["temporal_axis"],
            output_expectation_class="TurnTemporalAxis",
        ))
        validator.register_node_contract(NodeIntentContract(
            node_name="inference",
            input_intent_schema={"temporal_axis": "resolved temporal snapshot"},
            output_expectation_class="FinalizedTurnResult",
        ))

        violations = validator.validate_graph(["temporal", "inference"])
    """

    def __init__(self) -> None:
        self._contracts: dict[str, NodeIntentContract] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_node_contract(self, contract: NodeIntentContract) -> None:
        """Register a node intent contract."""
        self._contracts[contract.node_name] = contract

    def get_contract(self, node_name: str) -> NodeIntentContract | None:
        return self._contracts.get(node_name)

    def registered_nodes(self) -> list[str]:
        return list(self._contracts.keys())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_edge(
        self,
        upstream_name: str,
        downstream_name: str,
    ) -> list[SemanticViolation]:
        """Check that ``upstream_name`` satisfies the input requirements of ``downstream_name``.

        Returns an empty list when the edge is semantically compatible.
        """
        violations: list[SemanticViolation] = []

        upstream = self._contracts.get(upstream_name)
        downstream = self._contracts.get(downstream_name)

        if upstream is None:
            violations.append(
                SemanticViolation(
                    upstream_node=upstream_name,
                    downstream_node=downstream_name,
                    missing_key=None,
                    violation_type="missing_contract",
                    detail=f"No NodeIntentContract registered for upstream node '{upstream_name}'",
                ),
            )
            return violations

        if downstream is None:
            violations.append(
                SemanticViolation(
                    upstream_node=upstream_name,
                    downstream_node=downstream_name,
                    missing_key=None,
                    violation_type="missing_contract",
                    detail=f"No NodeIntentContract registered for downstream node '{downstream_name}'",
                ),
            )
            return violations

        provided: set[str] = set(upstream.output_provides)
        for required_key, intent in downstream.input_intent_schema.items():
            if required_key not in provided:
                # Check if any earlier node in the chain provides it.
                # (For edge-level check we only look at immediate predecessor.)
                violations.append(
                    SemanticViolation(
                        upstream_node=upstream_name,
                        downstream_node=downstream_name,
                        missing_key=required_key,
                        violation_type="missing_output",
                        detail=(
                            f"Downstream '{downstream_name}' requires '{required_key}' "
                            f"({intent}) but upstream '{upstream_name}' does not list it "
                            f"in output_provides (has: {sorted(provided) or 'none'})"
                        ),
                    ),
                )

        return violations

    def validate_graph(
        self,
        node_order: Sequence[str],
    ) -> list[SemanticViolation]:
        """Validate semantic compatibility for every consecutive edge in ``node_order``.

        For each downstream node, we check that the *cumulative* set of outputs
        provided by all previous nodes satisfies its ``input_intent_schema``.
        This mirrors how TurnContext.state accumulates across the pipeline.

        Returns a list of all violations found (empty → semantically valid).
        """
        cumulative_provides: set[str] = set()
        violations: list[SemanticViolation] = []

        for i, node_name in enumerate(node_order):
            contract = self._contracts.get(node_name)

            if contract is None:
                if i > 0:
                    violations.append(
                        SemanticViolation(
                            upstream_node=node_order[i - 1],
                            downstream_node=node_name,
                            missing_key=None,
                            violation_type="missing_contract",
                            detail=f"No NodeIntentContract registered for node '{node_name}'",
                        ),
                    )
                # Cannot accumulate outputs from an unknown node; skip.
                continue

            # Check that all required inputs are satisfied by cumulative outputs.
            for required_key, intent in contract.input_intent_schema.items():
                if required_key not in cumulative_provides:
                    upstream = node_order[i - 1] if i > 0 else "<pipeline_start>"
                    violations.append(
                        SemanticViolation(
                            upstream_node=upstream,
                            downstream_node=node_name,
                            missing_key=required_key,
                            violation_type="unsatisfied_input",
                            detail=(
                                f"Node '{node_name}' requires '{required_key}' ({intent}) "
                                f"but it has not been provided by any preceding node in "
                                f"[{', '.join(node_order[:i])}]"
                            ),
                        ),
                    )

            # Accumulate what this node outputs.
            cumulative_provides.update(contract.output_provides)

        if violations:
            logger.warning(
                "[Phase 4.3] Graph semantic validation found %d violation(s) in node order %s",
                len(violations),
                list(node_order),
            )
        else:
            logger.debug(
                "[Phase 4.3] Graph semantic validation passed for node order %s",
                list(node_order),
            )

        return violations


# ---------------------------------------------------------------------------
# Pre-wired DadBot pipeline node contracts
# ---------------------------------------------------------------------------


def build_dadbot_semantic_validator() -> GraphSemanticValidator:
    """Return a ``GraphSemanticValidator`` pre-populated with DadBot's pipeline contracts.

    Node order: temporal → preflight → planner → inference → safety → reflection → save
    """
    validator = GraphSemanticValidator()

    validator.register_node_contract(
        NodeIntentContract(
            node_name="temporal",
            input_intent_schema={},
            output_expectation_class="TurnTemporalAxis",
            output_provides=["temporal_axis", "temporal_snapshot"],
            mutation_constraints=[],
            description="Resolves wall-clock time; no fallback to system time permitted.",
        ),
    )
    validator.register_node_contract(
        NodeIntentContract(
            node_name="preflight",
            input_intent_schema={
                "temporal_axis": "resolved temporal snapshot from TemporalNode",
            },
            output_expectation_class="PreflightContext",
            output_provides=["health_state", "memory_context", "context_layers"],
            mutation_constraints=["no_direct_memory_write"],
            description="Parallel health check + memory context assembly.",
        ),
    )
    validator.register_node_contract(
        NodeIntentContract(
            node_name="planner",
            input_intent_schema={
                "memory_context": "assembled memory context from ContextBuilderNode",
            },
            output_expectation_class="TurnPlan",
            output_provides=["turn_plan", "detected_goals"],
            mutation_constraints=["no_direct_memory_write"],
            description="Decomposes user intent and detects new goals.",
        ),
    )
    validator.register_node_contract(
        NodeIntentContract(
            node_name="inference",
            input_intent_schema={
                "temporal_axis": "resolved temporal snapshot",
                "memory_context": "assembled memory context",
            },
            output_expectation_class="FinalizedTurnResult",
            output_provides=["inference_result", "reply_text"],
            mutation_constraints=["no_direct_memory_write"],
            description="LLM inference with critique engine.",
        ),
    )
    validator.register_node_contract(
        NodeIntentContract(
            node_name="safety",
            input_intent_schema={
                "inference_result": "finalized LLM reply from InferenceNode",
            },
            output_expectation_class="SafetyVerdict",
            output_provides=["safety_verdict", "filtered_reply"],
            mutation_constraints=["no_direct_memory_write"],
            description="Safety policy enforcement over the LLM reply.",
        ),
    )
    validator.register_node_contract(
        NodeIntentContract(
            node_name="reflection",
            input_intent_schema={
                "safety_verdict": "safety policy verdict from SafetyNode",
            },
            output_expectation_class="ReflectionReport",
            output_provides=["reflection_report"],
            mutation_constraints=["no_direct_memory_write"],
            description="Post-turn reflection and self-critique.",
        ),
    )
    validator.register_node_contract(
        NodeIntentContract(
            node_name="save",
            input_intent_schema={
                "inference_result": "finalized reply to persist",
                "temporal_axis": "temporal envelope for ledger stamping",
            },
            output_expectation_class="PersistenceAck",
            output_provides=["turn_saved", "checkpoint_hash"],
            mutation_constraints=[
                "MutationQueue.drain_only",
                "save_node_single_execution",
            ],
            description="Durable persistence; executes exactly once per completed turn.",
        ),
    )

    return validator
