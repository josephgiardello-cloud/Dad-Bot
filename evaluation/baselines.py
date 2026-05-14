"""Baseline Model Registry — Phase 4F Industry Normalization Layer.

Purpose:
  Define a set of known comparison baselines so capability scores can be
  expressed as "better/worse than X" rather than just raw numbers.

Baseline agents defined here:
  heuristic_agent  — rule-based responder, no LLM, no tools, minimal memory
  naive_planner    — LLM without tool routing or goal tracking
  llm_only         — pure LLM (GPT-class), no tool system, no structured memory
  tool_disabled    — full orchestrator but tools disabled (control condition)

Derivation methodology:
  Scores are calibrated estimates based on:
    - Known architectural capabilities / limitations of each agent type
    - Published benchmark results for similar agent classes
    - Conservative estimates (biased toward underestimating baselines
      to avoid false "above baseline" claims)

  These are SEEDED VALUES — they are NOT measured on real runs yet.
  When real runs are available, update via update_baseline() to replace
  these estimates with measured values.

Usage:
    from evaluation.baselines import BASELINES, get_baseline, list_baselines

    llm_scores = get_baseline("llm_only")
    print(llm_scores.planning)  # 0.62
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Baseline data structure
# ---------------------------------------------------------------------------


@dataclass
class BaselineProfile:
    """Complete capability profile for a baseline agent type."""

    name: str
    description: str
    source: str  # "seeded" | "measured" | "published"

    # Per-subsystem average scores (0.0-1.0)
    planning: float
    tools: float
    memory: float
    ux: float
    robustness: float

    # Category-level scores (same subsystems but broken out for normalization)
    category_scores: dict[str, float] = field(default_factory=dict)

    # Overall weighted score (using standard weights)
    overall: float = 0.0

    def __post_init__(self) -> None:
        if not self.category_scores:
            self.category_scores = {
                "planning": self.planning,
                "tools": self.tools,
                "memory": self.memory,
                "ux": self.ux,
                "robustness": self.robustness,
            }
        if self.overall == 0.0:
            self.overall = round(
                self.planning * 0.25 + self.tools * 0.25 + self.memory * 0.20 + self.ux * 0.15 + self.robustness * 0.15,
                4,
            )

    def get(self, subsystem: str) -> float:
        return self.category_scores.get(subsystem, 0.0)


# ---------------------------------------------------------------------------
# Seeded baseline profiles
# ---------------------------------------------------------------------------

# Heuristic agent: hard-coded rule-based responder
# - No LLM, no tool system, no semantic memory
# - Can handle very simple queries with fixed rules
# - Fails on anything requiring context or tool use
HEURISTIC_AGENT = BaselineProfile(
    name="heuristic_agent",
    description=(
        "Rule-based agent with no LLM, no tools, and no semantic memory. Handles simple keyword-matched queries only."
    ),
    source="seeded",
    planning=0.28,  # simple rule matching, no decomposition
    tools=0.15,  # no real tool system
    memory=0.20,  # basic session key-value only
    ux=0.45,  # consistent but rigid
    robustness=0.60,  # no LLM = no hallucination, but no graceful degradation
)

# Naive planner: LLM with basic conversation but no structured planning
# - Has LLM inference, no goal tracker, no tool routing
# - Plans by generating prose, not structured steps
NAIVE_PLANNER = BaselineProfile(
    name="naive_planner",
    description=(
        "LLM agent with no structured planning, no tool routing, and no goal tracking. "
        "Responds conversationally but cannot execute tools or decompose tasks."
    ),
    source="seeded",
    planning=0.48,  # generates text plans but no formal decomposition
    tools=0.22,  # mentions tools but cannot invoke them reliably
    memory=0.35,  # conversational context only, no structured retrieval
    ux=0.62,  # better at natural language than heuristic
    robustness=0.58,  # can refuse adversarial input but not robustly
)

# LLM-only: production-grade LLM (GPT-4 class) with no orchestration layer
# - Full LLM capabilities: instruction following, chain-of-thought
# - No structured tool system, no goal memory, no formal plan graph
# - Based on conservative estimates from published LLM benchmarks
LLM_ONLY = BaselineProfile(
    name="llm_only",
    description=(
        "Production-grade LLM (GPT-4 class) without tool orchestration, "
        "goal tracking, or structured memory. Pure language model capabilities."
    ),
    source="seeded",
    planning=0.62,  # strong chain-of-thought, but no formal DAG
    tools=0.25,  # can describe tools but cannot reliably route/recover
    memory=0.48,  # long context window, but no cross-session persistence
    ux=0.72,  # excellent natural language, clarification behavior
    robustness=0.68,  # safety training present but no formal boundary enforcement
)

# Tool-disabled: full orchestrator pipeline with tools turned off (control)
# - All nodes active: planner, inference, memory, safety, reflection
# - Tools deliberately disabled — measures non-tool capability
TOOL_DISABLED = BaselineProfile(
    name="tool_disabled",
    description=(
        "Full Dad-Bot orchestrator with tool system disabled. "
        "Measures pure reasoning and memory capability without tool augmentation."
    ),
    source="seeded",
    planning=0.68,  # full planner active, no tool routing interference
    tools=0.08,  # tools disabled — only matters for tool scenarios
    memory=0.55,  # structured memory active
    ux=0.70,  # full inference + safety nodes
    robustness=0.72,  # safety node active, graceful degradation tested
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINES: dict[str, BaselineProfile] = {b.name: b for b in [HEURISTIC_AGENT, NAIVE_PLANNER, LLM_ONLY, TOOL_DISABLED]}


def get_baseline(name: str) -> BaselineProfile:
    """Get a baseline profile by name. Raises KeyError if not found."""
    if name not in BASELINES:
        raise KeyError(f"Baseline not found: {name!r}. Available: {list(BASELINES.keys())}")
    return BASELINES[name]


def list_baselines() -> list[str]:
    """Return names of all registered baselines."""
    return list(BASELINES.keys())


def baseline_distribution(subsystem: str) -> dict[str, float]:
    """Return {name: score} for all baselines on a given subsystem."""
    return {name: profile.get(subsystem) for name, profile in BASELINES.items()}


def update_baseline(name: str, **subsystem_scores: float) -> None:
    """Update a baseline with measured values (replaces seeded estimates).

    Args:
        name: baseline name
        **subsystem_scores: keyword args for each subsystem to update
            e.g. update_baseline("llm_only", planning=0.71, memory=0.53)
    """
    if name not in BASELINES:
        raise KeyError(f"Baseline not found: {name!r}")
    profile = BASELINES[name]
    for sub, val in subsystem_scores.items():
        if hasattr(profile, sub):
            setattr(profile, sub, float(val))
            profile.category_scores[sub] = float(val)
    # Recompute overall
    profile.overall = round(
        profile.planning * 0.25
        + profile.tools * 0.25
        + profile.memory * 0.20
        + profile.ux * 0.15
        + profile.robustness * 0.15,
        4,
    )
    profile.source = "measured"
