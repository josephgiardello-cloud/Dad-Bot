"""Phase 2.3 — Tool Resource Model.

Provides cost modeling, budget tracking, and exhaustion behavior for tool execution.

Primitives:
    ToolCostEntry:       per-tool cost profile (cost_units, latency_ms estimate)
    TurnBudget:          per-turn limits with exhaustion policy
    BudgetExhaustionPolicy: DEGRADE | SKIP | COMPRESS
    ResourceModelValidator: validates a DAG or tool list against a budget
    BudgetReport:        result of validation (within_budget, exhausted_tools, action)

Design:
    Resource tracking is a FIRST-CLASS execution variable, not an afterthought.
    Budget enforcement happens at plan time (before execution), not at failure.
    Exhaustion policies degrade gracefully rather than hard-fail.
"""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dadbot.core.tool_dag import ToolDAG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()


# ---------------------------------------------------------------------------
# Exhaustion policy
# ---------------------------------------------------------------------------


class BudgetExhaustionPolicy(enum.Enum):
    """What to do when tool budget is exhausted mid-execution.

    DEGRADE:   Continue with reduced tool set (drop lowest-priority tools first).
    SKIP:      Skip all tools beyond the budget; reply from what's available.
    COMPRESS:  Run remaining tools with a compressed output budget.
    """

    DEGRADE = "degrade"
    SKIP = "skip"
    COMPRESS = "compress"


# ---------------------------------------------------------------------------
# Tool cost entry
# ---------------------------------------------------------------------------


# Default cost catalog for the allowed tool set.
_DEFAULT_COSTS: dict[str, tuple[float, int]] = {
    # tool_name: (cost_units, latency_ms_estimate)
    "memory_lookup": (1.0, 30),
    "goal_lookup": (0.5, 20),
    "session_memory_fetch": (0.5, 20),
    # Fallback for unknown tools:
    "_unknown_": (2.0, 100),
}


@dataclass(frozen=True)
class ToolCostEntry:
    """Cost profile for a single tool.

    Attributes:
        tool_name:          Tool identifier.
        cost_units:         Abstract cost (used against TurnBudget.max_cost_units).
        latency_ms_estimate: Estimated execution latency in milliseconds.

    """

    tool_name: str
    cost_units: float
    latency_ms_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "cost_units": self.cost_units,
            "latency_ms_estimate": self.latency_ms_estimate,
        }


class ToolCostCatalog:
    """Registry of per-tool cost entries."""

    def __init__(self, entries: dict[str, ToolCostEntry] | None = None) -> None:
        self._entries: dict[str, ToolCostEntry] = dict(entries or {})

    @classmethod
    def default(cls) -> ToolCostCatalog:
        entries: dict[str, ToolCostEntry] = {}
        for name, (cost, latency) in _DEFAULT_COSTS.items():
            entries[name] = ToolCostEntry(
                tool_name=name,
                cost_units=cost,
                latency_ms_estimate=latency,
            )
        return cls(entries)

    def get(self, tool_name: str) -> ToolCostEntry:
        return self._entries.get(
            tool_name,
            self._entries.get(
                "_unknown_",
                ToolCostEntry(
                    tool_name=tool_name,
                    cost_units=2.0,
                    latency_ms_estimate=100,
                ),
            ),
        )

    def register(self, entry: ToolCostEntry) -> None:
        self._entries[entry.tool_name] = entry

    def all_tools(self) -> list[str]:
        return [k for k in self._entries if k != "_unknown_"]


# ---------------------------------------------------------------------------
# Turn budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnBudget:
    """Per-turn execution budget.

    Attributes:
        max_cost_units:       Total cost units available for this turn.
        max_tools:            Maximum number of tool calls per turn.
        max_latency_ms:       Optional wall-clock time budget.
        exhaustion_policy:    What to do when budget is exhausted.

    """

    max_cost_units: float
    max_tools: int
    max_latency_ms: int
    exhaustion_policy: BudgetExhaustionPolicy

    @classmethod
    def default(cls) -> TurnBudget:
        """Standard budget: 5 cost units, 3 tools, 500ms, DEGRADE."""
        return cls(
            max_cost_units=5.0,
            max_tools=3,
            max_latency_ms=500,
            exhaustion_policy=BudgetExhaustionPolicy.DEGRADE,
        )

    @classmethod
    def strict(cls) -> TurnBudget:
        """Strict budget: 2 cost units, 1 tool, 200ms, SKIP."""
        return cls(
            max_cost_units=2.0,
            max_tools=1,
            max_latency_ms=200,
            exhaustion_policy=BudgetExhaustionPolicy.SKIP,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_cost_units": self.max_cost_units,
            "max_tools": self.max_tools,
            "max_latency_ms": self.max_latency_ms,
            "exhaustion_policy": self.exhaustion_policy.value,
        }


# ---------------------------------------------------------------------------
# Budget report
# ---------------------------------------------------------------------------


@dataclass
class BudgetReport:
    """Result of validating a tool plan against a TurnBudget.

    Attributes:
        total_cost_units:     Total cost of all tools in the plan.
        total_latency_ms:     Total estimated latency.
        tool_count:           Number of tools in the plan.
        within_budget:        True iff all limits are satisfied.
        exhausted_tools:      List of tools that exceed/push-over budget.
        approved_tools:       Tools approved for execution within budget.
        suggested_action:     Recommended action from exhaustion_policy.
        budget_hash:          Deterministic hash of the report (for audit).

    """

    total_cost_units: float
    total_latency_ms: int
    tool_count: int
    within_budget: bool
    exhausted_tools: list[str]
    approved_tools: list[str]
    suggested_action: str
    budget_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cost_units": self.total_cost_units,
            "total_latency_ms": self.total_latency_ms,
            "tool_count": self.tool_count,
            "within_budget": self.within_budget,
            "exhausted_tools": self.exhausted_tools,
            "approved_tools": self.approved_tools,
            "suggested_action": self.suggested_action,
            "budget_hash": self.budget_hash,
        }


# ---------------------------------------------------------------------------
# Resource model validator
# ---------------------------------------------------------------------------


class ResourceModelValidator:
    """Validates a list of tool names (or a ToolDAG) against a TurnBudget.

    Usage:
        catalog = ToolCostCatalog.default()
        budget  = TurnBudget.default()
        validator = ResourceModelValidator(catalog)

        report = validator.validate_tool_list(["memory_lookup", "goal_lookup"], budget)
        if not report.within_budget:
            # handle based on report.suggested_action
    """

    def __init__(self, catalog: ToolCostCatalog | None = None) -> None:
        self.catalog = catalog or ToolCostCatalog.default()

    def validate_tool_list(
        self,
        tool_names: list[str],
        budget: TurnBudget,
    ) -> BudgetReport:
        """Validate an ordered list of tool names against the budget.

        DEGRADE policy: drops lowest-priority (last) tools that push over budget.
        SKIP policy: rejects entire plan if over budget.
        COMPRESS policy: allows the plan but marks exhausted tools.
        """
        costs = [self.catalog.get(name) for name in (tool_names or [])]
        total_cost = sum(c.cost_units for c in costs)
        total_latency = sum(c.latency_ms_estimate for c in costs)
        tool_count = len(costs)

        within_cost = total_cost <= budget.max_cost_units
        within_count = tool_count <= budget.max_tools
        within_latency = total_latency <= budget.max_latency_ms
        within_budget = within_cost and within_count and within_latency

        approved_tools: list[str] = []
        exhausted_tools: list[str] = []

        if budget.exhaustion_policy == BudgetExhaustionPolicy.DEGRADE:
            # Greedily take tools in order until budget is exhausted.
            running_cost = 0.0
            running_latency = 0
            running_count = 0
            for name, entry in zip(tool_names or [], costs):
                if (
                    running_cost + entry.cost_units <= budget.max_cost_units
                    and running_latency + entry.latency_ms_estimate <= budget.max_latency_ms
                    and running_count < budget.max_tools
                ):
                    approved_tools.append(name)
                    running_cost += entry.cost_units
                    running_latency += entry.latency_ms_estimate
                    running_count += 1
                else:
                    exhausted_tools.append(name)

        elif budget.exhaustion_policy == BudgetExhaustionPolicy.SKIP:
            if within_budget:
                approved_tools = list(tool_names or [])
            else:
                exhausted_tools = list(tool_names or [])

        elif budget.exhaustion_policy == BudgetExhaustionPolicy.COMPRESS:
            # All tools allowed; flag the ones that exceed budget.
            approved_tools = list(tool_names or [])
            if not (within_cost and within_count and within_latency):
                exhausted_tools = list(tool_names or [])

        if budget.exhaustion_policy in (
            BudgetExhaustionPolicy.DEGRADE,
            BudgetExhaustionPolicy.SKIP,
        ):
            effective_within = (
                sum(self.catalog.get(t).cost_units for t in approved_tools) <= budget.max_cost_units
                and len(approved_tools) <= budget.max_tools
            )
        else:
            effective_within = within_budget

        suggested_action = "proceed" if not exhausted_tools else budget.exhaustion_policy.value

        budget_hash = _sha256(
            {
                "total_cost_units": total_cost,
                "total_latency_ms": total_latency,
                "tool_count": tool_count,
                "approved_tools": approved_tools,
                "policy": budget.exhaustion_policy.value,
            },
        )

        return BudgetReport(
            total_cost_units=total_cost,
            total_latency_ms=total_latency,
            tool_count=tool_count,
            within_budget=effective_within,
            exhausted_tools=exhausted_tools,
            approved_tools=approved_tools,
            suggested_action=suggested_action,
            budget_hash=budget_hash,
        )

    def validate_dag(self, dag: ToolDAG, budget: TurnBudget) -> BudgetReport:
        """Validate a ToolDAG against the budget."""
        tool_names = [node.tool_name for node in sorted(dag.nodes, key=lambda n: n.sequence)]
        return self.validate_tool_list(tool_names, budget)


__all__ = [
    "BudgetExhaustionPolicy",
    "BudgetReport",
    "ResourceModelValidator",
    "ToolCostCatalog",
    "ToolCostEntry",
    "TurnBudget",
]
