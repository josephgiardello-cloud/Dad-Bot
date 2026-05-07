"""Phase C: Policy Intermediate Representation.

Transforms policy evaluation from implicit control flow to explicit rules-as-data,
enabling:
- Policy rules with normalized conditions and effects
- Effect composition and precedence resolution
- Audit trails through PolicyEffect chains
- Separation of concerns: evaluation, synthesis, execution

Architecture:
  ToolResult (from registry)
       ↓
  PolicyEvaluator (rules engine)
       ↓
  PolicyDecision (matched rules + effects)
       ↓
  RecoveryStrategy selection (Phase D)
       ↓
  Execution / Audit Trail
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

from dadbot.core.runtime_types import (
    PolicyEffect,
    PolicyEffectType,
    ToolResult,
    ToolExecutionStatus,
)


class PolicyRuleCondition(str, Enum):
    """Condition types for policy rules."""

    TOOL_NAME_MATCH = "tool_name_match"  # tool_name in (names...)
    TOOL_DETERMINISM = "tool_determinism"  # spec.determinism == class
    TOOL_SIDE_EFFECTS = "tool_side_effects"  # spec.side_effect_class == class
    EXECUTION_STATUS = "execution_status"  # result.status in (statuses...)
    ERROR_PATTERN = "error_pattern"  # error message matches regex
    OUTPUT_SIZE_THRESHOLD = "output_size_threshold"  # payload size > threshold
    PERMISSION_REQUIRED = "permission_required"  # caller lacks permission
    USER_EXPLICIT = "user_explicit"  # explicit user approval/denial


@dataclass(frozen=True)
class PolicyCondition:
    """Single condition in a policy rule."""

    condition_type: PolicyRuleCondition
    params: dict[str, Any] = field(default_factory=dict)

    def matches(self, context: dict[str, Any]) -> bool:
        """Evaluate condition against context.
        
        Context typically contains:
          - tool_result: ToolResult
          - tool_spec: ToolSpec
          - caller_identity: ExecutionIdentity
          - user_context: dict
        """
        if self.condition_type == PolicyRuleCondition.TOOL_NAME_MATCH:
            names = self.params.get("names", [])
            tool_result = context.get("tool_result")
            return tool_result and tool_result.tool_name in names

        elif self.condition_type == PolicyRuleCondition.EXECUTION_STATUS:
            statuses = self.params.get("statuses", [])
            tool_result = context.get("tool_result")
            return tool_result and tool_result.status in statuses

        elif self.condition_type == PolicyRuleCondition.ERROR_PATTERN:
            import re

            pattern = self.params.get("pattern", "")
            tool_result = context.get("tool_result")
            if not tool_result or not tool_result.error:
                return False
            return bool(re.search(pattern, tool_result.error, re.IGNORECASE))

        elif self.condition_type == PolicyRuleCondition.TOOL_DETERMINISM:
            determinism = self.params.get("determinism")
            tool_spec = context.get("tool_spec")
            return tool_spec and tool_spec.determinism == determinism

        elif self.condition_type == PolicyRuleCondition.TOOL_SIDE_EFFECTS:
            side_effect = self.params.get("side_effect_class")
            tool_spec = context.get("tool_spec")
            return tool_spec and tool_spec.side_effect_class == side_effect

        elif self.condition_type == PolicyRuleCondition.OUTPUT_SIZE_THRESHOLD:
            threshold = self.params.get("threshold_bytes", 1000000)
            tool_result = context.get("tool_result")
            if not tool_result or not tool_result.payload:
                return False
            try:
                import json

                size = len(json.dumps(tool_result.payload.content).encode("utf-8"))
                return size > threshold
            except Exception:
                return False

        elif self.condition_type == PolicyRuleCondition.PERMISSION_REQUIRED:
            permission = self.params.get("permission", "")
            caller = context.get("caller_identity")
            user_context = context.get("user_context", {})
            granted = user_context.get("permissions", [])
            return permission and permission not in granted

        # Default: unrecognized condition always matches (fail-open)
        return True


@dataclass(frozen=True)
class PolicyRule:
    """Complete policy rule: conditions + effects."""

    rule_id: str
    rule_name: str
    description: str = ""
    # Conditions combined with AND (all must match to trigger)
    conditions: tuple[PolicyCondition, ...] = field(default_factory=tuple)
    # Effects to emit if rule matches
    effects: tuple[PolicyEffectType, ...] = field(default_factory=tuple)
    # Priority: higher number = higher priority (0-100)
    priority: int = 50

    def matches(self, context: dict[str, Any]) -> bool:
        """Rule matches if all conditions are true."""
        if not self.conditions:
            # No conditions = always matches
            return True
        return all(cond.matches(context) for cond in self.conditions)


class PolicyEvaluator:
    """Evaluates tool results against policy rules.
    
    Emits PolicyEffect chains that capture:
    - Which rules matched
    - What effects they triggered
    - Output mutations (before/after hashes)
    - Audit trail for recovery
    """

    def __init__(self, rules: list[PolicyRule] | None = None) -> None:
        """Initialize evaluator with optional rule set.
        
        Args:
            rules: List of PolicyRule to evaluate in order of priority
        """
        self.rules = sorted(rules or [], key=lambda r: r.priority, reverse=True)

    def evaluate(
        self,
        tool_result: ToolResult,
        context: dict[str, Any] | None = None,
    ) -> tuple[PolicyEffect, ...]:
        """Evaluate tool result against all rules.
        
        Args:
            tool_result: Result from tool execution
            context: Additional context (tool_spec, caller_identity, etc.)
        
        Returns:
            Tuple of PolicyEffect representing matched rules and their effects
        """
        context = context or {}
        context["tool_result"] = tool_result

        effects: list[PolicyEffect] = []

        # Evaluate each rule in priority order
        for rule in self.rules:
            if rule.matches(context):
                # Rule matches: emit effects
                for effect_type in rule.effects:
                    effect = PolicyEffect(
                        effect_type=effect_type,
                        source_rule=rule.rule_id,
                        before_hash=tool_result.payload.content_hash
                        if tool_result.payload
                        else "",
                        after_hash="",  # Will be computed by effect handler
                        reason=f"Rule {rule.rule_name} matched",
                    )
                    effects.append(effect)

        return tuple(effects)


@dataclass
class PolicyDecisionIR:
    """Policy decision with explicit effect chain (Phase C output).
    
    Distinct from legacy PolicyDecision (which is control-flow based).
    This is rules-as-data with effects as first-class objects.
    """

    tool_result: ToolResult
    matched_rules: tuple[str, ...] = field(default_factory=tuple)
    # Effects in evaluation order
    emitted_effects: tuple[PolicyEffect, ...] = field(default_factory=tuple)
    # Final output after applying all effects
    final_output: Any = None
    # Whether output was modified from original
    output_was_modified: bool = False

    def effect_chain_summary(self) -> str:
        """Summarize effect chain for audit."""
        if not self.emitted_effects:
            return "no_effects"
        effect_types = [e.effect_type.value for e in self.emitted_effects]
        return "|".join(effect_types)


class EffectSynthesizer(Protocol):
    """Protocol for synthesizing final output from policy effects."""

    def __call__(
        self,
        tool_result: ToolResult,
        effects: tuple[PolicyEffect, ...],
    ) -> tuple[Any, bool]:
        """Apply effects to tool result and return (final_output, was_modified).
        
        Args:
            tool_result: Original tool execution result
            effects: PolicyEffect chain to apply
        
        Returns:
            (final_output, output_was_modified) tuple
        """
        ...


def default_effect_synthesizer(
    tool_result: ToolResult,
    effects: tuple[PolicyEffect, ...],
) -> tuple[Any, bool]:
    """Default effect synthesizer: apply effects in order.
    
    Handles common policy effects:
    - DENY_TOOL: return error
    - REWRITE_OUTPUT: transform payload
    - STRIP_FACET: remove personality
    - FORCE_DEGRADATION: use fallback
    """
    output = tool_result.payload.content if tool_result.payload else None
    was_modified = False

    for effect in effects:
        if effect.effect_type == PolicyEffectType.DENY_TOOL:
            output = {"error": "Tool invocation denied by policy"}
            was_modified = True

        elif effect.effect_type == PolicyEffectType.REWRITE_OUTPUT:
            # Effect params contain rewrite function or mapping
            rewriter = effect.params.get("rewriter") if hasattr(effect, "params") else None
            if callable(rewriter):
                output = rewriter(output)
                was_modified = True

        elif effect.effect_type == PolicyEffectType.STRIP_FACET:
            # Remove sarcasm/personality/tone
            if isinstance(output, dict) and "sarcasm" in output:
                del output["sarcasm"]
                was_modified = True

        elif effect.effect_type == PolicyEffectType.FORCE_DEGRADATION:
            output = {"degraded": True, "message": "Tool output degraded by policy"}
            was_modified = True

    return output, was_modified


class PolicyCompilerIR:
    """Phase C policy compiler: rules-as-data, effects-as-audit-trail."""

    def __init__(
        self,
        rules: list[PolicyRule] | None = None,
        synthesizer: EffectSynthesizer | None = None,
    ) -> None:
        """Initialize compiler with rules and effect synthesizer.
        
        Args:
            rules: PolicyRule list to evaluate
            synthesizer: Function to apply effects to results
        """
        self.evaluator = PolicyEvaluator(rules)
        self.synthesizer = synthesizer or default_effect_synthesizer

    def evaluate_with_effects(
        self,
        tool_result: ToolResult,
        context: dict[str, Any] | None = None,
    ) -> PolicyDecisionIR:
        """Evaluate tool result and emit effect chain.
        
        Args:
            tool_result: Result from registry execution
            context: Tool spec, caller identity, user context, etc.
        
        Returns:
            PolicyDecisionIR with matched rules and effects
        """
        # Evaluate rules
        effects = self.evaluator.evaluate(tool_result, context)

        # Synthesize output
        final_output, was_modified = self.synthesizer(tool_result, effects)

        # Get matched rule IDs
        matched_rule_ids = []
        for rule in self.evaluator.rules:
            if rule.matches(context or {}):
                matched_rule_ids.append(rule.rule_id)

        return PolicyDecisionIR(
            tool_result=tool_result,
            matched_rules=tuple(matched_rule_ids),
            emitted_effects=effects,
            final_output=final_output,
            output_was_modified=was_modified,
        )


# Built-in policy rules

SAFETY_RULE_DENY_UNSAFE_TOOLS = PolicyRule(
    rule_id="safety_deny_unsafe",
    rule_name="Deny execution of unsafe tools",
    description="Block tools marked as potentially dangerous",
    conditions=(
        PolicyCondition(
            PolicyRuleCondition.TOOL_NAME_MATCH,
            params={"names": ["exec", "eval", "system_call", "delete_files"]},
        ),
    ),
    effects=(PolicyEffectType.DENY_TOOL,),
    priority=90,
)

AUDIT_RULE_LOG_ERRORS = PolicyRule(
    rule_id="audit_log_errors",
    rule_name="Log execution errors",
    description="Capture all tool errors for audit trail",
    conditions=(
        PolicyCondition(
            PolicyRuleCondition.EXECUTION_STATUS,
            params={"statuses": [ToolExecutionStatus.ERROR, ToolExecutionStatus.TIMEOUT]},
        ),
    ),
    effects=(PolicyEffectType.REQUIRE_APPROVAL,),
    priority=60,
)

AUDIT_RULE_LOG_LARGE_OUTPUT = PolicyRule(
    rule_id="audit_log_large",
    rule_name="Flag large output",
    description="Audit tools that return large payloads",
    conditions=(
        PolicyCondition(
            PolicyRuleCondition.OUTPUT_SIZE_THRESHOLD,
            params={"threshold_bytes": 1000000},
        ),
    ),
    effects=(PolicyEffectType.REQUIRE_APPROVAL,),
    priority=50,
)

__all__ = [
    "PolicyRuleCondition",
    "PolicyCondition",
    "PolicyRule",
    "PolicyEvaluator",
    "PolicyDecisionIR",
    "EffectSynthesizer",
    "default_effect_synthesizer",
    "PolicyCompilerIR",
    "SAFETY_RULE_DENY_UNSAFE_TOOLS",
    "AUDIT_RULE_LOG_ERRORS",
    "AUDIT_RULE_LOG_LARGE_OUTPUT",
]
