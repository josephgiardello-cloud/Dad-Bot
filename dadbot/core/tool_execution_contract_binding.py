"""Tool execution contract binding: connects formal contract to runtime execution.

This module implements pre-flight validation, failure classification via contract taxonomy,
contract-aware result normalization, and the Tool→Memory causal write loop.

After every execution the runtime optionally:
  1. Consults the FailurePolicyEngine to determine next-state action for failures.
  2. Builds a CausalMemoryEntry with full causal metadata.
  3. Emits the entry to the memory sink according to the configured CausalWritePolicy.
"""

from __future__ import annotations

from typing import Any

from dadbot.core.external_tool_runtime import (
    ExternalToolRuntime,
    NetworkFailureKind,
    ToolExecutionResult,
    ToolExecutionStatus,
)
from dadbot.core.tool_execution_contract import (
    FailureClass,
    ToolExecutionContract,
    taxonomy_entry_for,
)
from dadbot.core.tool_memory_causal_contract import (
    CausalWritePolicy,
    build_causal_entry,
    emit_causal_entry,
)


class ToolExecutionContractViolation(ValueError):
    """Raised when execution or result violates contract terms."""

    pass


def classify_runtime_failure_to_contract_failure(
    *,
    runtime_failure_kind: NetworkFailureKind,
    http_status: int | None = None,
    error_message: str = "",
) -> FailureClass:
    """Map runtime NetworkFailureKind + HTTP status to formal FailureClass."""
    if runtime_failure_kind == NetworkFailureKind.TIMEOUT:
        return FailureClass.NETWORK_TIMEOUT
    if runtime_failure_kind == NetworkFailureKind.RATE_LIMIT:
        return FailureClass.NETWORK_RATE_LIMIT
    if runtime_failure_kind in {
        NetworkFailureKind.DNS,
        NetworkFailureKind.CONNECTION,
        NetworkFailureKind.SERVER,
    }:
        return FailureClass.NETWORK_TEMPORARY
    if http_status is not None:
        if 400 <= http_status < 500:
            if http_status == 403:
                return FailureClass.PERMISSION_DENIED
            if http_status == 401:
                return FailureClass.PERMISSION_DENIED
            return FailureClass.CLIENT_ERROR
    if "permission" in error_message.lower() or "denied" in error_message.lower():
        return FailureClass.PERMISSION_DENIED
    if "schema" in error_message.lower() or "validation" in error_message.lower():
        return FailureClass.SCHEMA_VALIDATION
    if "not registered" in error_message.lower() or "incompatible" in error_message.lower():
        return FailureClass.RESOURCE_UNAVAILABLE
    if "timeout" in error_message.lower():
        return FailureClass.TIMEOUT_EXCEEDED
    if "isolation" in error_message.lower():
        return FailureClass.ISOLATION_VIOLATED
    return FailureClass.NETWORK_UNKNOWN


def tool_execution_status_to_failure_class(
    status: ToolExecutionStatus,
    error: str = "",
) -> FailureClass:
    """Map ToolExecutionStatus to FailureClass."""
    if status == ToolExecutionStatus.OK:
        raise ValueError("OK status is not a failure")
    if status == ToolExecutionStatus.PARTIAL:
        return FailureClass.PARTIAL_OUTPUT
    if status == ToolExecutionStatus.DEGRADED:
        return FailureClass.DEGRADED_CONFIDENCE
    if status == ToolExecutionStatus.TIMEOUT:
        return FailureClass.TIMEOUT_EXCEEDED
    if status == ToolExecutionStatus.ERROR:
        if "permission" in error.lower():
            return FailureClass.PERMISSION_DENIED
        if "schema" in error.lower():
            return FailureClass.SCHEMA_VALIDATION
        if "not_registered" in error or "incompatible_version" in error:
            return FailureClass.RESOURCE_UNAVAILABLE
        if "isolation" in error.lower():
            return FailureClass.ISOLATION_VIOLATED
        return FailureClass.TOOL_INTERNAL_ERROR
    return FailureClass.UNKNOWN


def validate_tool_execution_input(
    contract: ToolExecutionContract,
    payload: dict[str, Any],
) -> tuple[bool, str]:
    """Pre-flight validation: verify input satisfies contract.

    Returns (valid, error_message).
    """
    valid, msg = contract.validate_input(payload)
    if not valid:
        return False, f"Input contract violation: {msg}"
    return True, "ok"


def validate_tool_execution_output(
    contract: ToolExecutionContract,
    output: Any,
) -> tuple[bool, str]:
    """Post-flight validation: verify output satisfies contract.

    Returns (valid, error_message).
    """
    valid, msg = contract.validate_output(output)
    if not valid:
        return False, f"Output contract violation: {msg}"
    return True, "ok"


def normalize_result_failure_class(
    contract: ToolExecutionContract,
    result: ToolExecutionResult,
) -> ToolExecutionResult:
    """Enrich result metadata with formal FailureClass from contract taxonomy.

    If result is failure, classify it via contract and attach taxonomy entry.
    """
    if result.status == ToolExecutionStatus.OK:
        return result

    # Determine failure class
    if result.status == ToolExecutionStatus.PARTIAL:
        failure_class = FailureClass.PARTIAL_OUTPUT
    elif result.status == ToolExecutionStatus.DEGRADED:
        failure_class = FailureClass.DEGRADED_CONFIDENCE
    else:
        failure_class = tool_execution_status_to_failure_class(result.status, result.error or "")

    # Validate against contract's supported failures
    try:
        taxonomy_entry = contract.classify_failure(failure_class)
    except ValueError as exc:
        # If contract doesn't document this failure, log but don't fail
        result.metadata = dict(result.metadata or {})
        result.metadata["contract_violation"] = f"Undocumented failure: {exc}"
        return result

    # Attach taxonomy entry to result
    result.metadata = dict(result.metadata or {})
    result.metadata["failure_class"] = failure_class.value
    result.metadata["failure_severity"] = taxonomy_entry.severity.value
    result.metadata["failure_retryable"] = taxonomy_entry.retryable
    result.metadata["escalation_key"] = taxonomy_entry.escalation_key
    result.metadata["failure_description"] = taxonomy_entry.description

    return result


class ContractAwareToolRuntime:
    """Wrapper around ExternalToolRuntime that enforces contract validation, classification,
    and the Tool→Memory causal write loop.

    Memory wiring
    -------------
    Pass a ``memory_sink`` (any callable accepting a dict, e.g. ToolResultMemorySink) to
    enable the causal loop.  After each execution the runtime will:

      1. (If failure) consult the ``policy_engine`` (FailurePolicyEngine) for next-state action.
      2. Build a CausalMemoryEntry with full causal provenance.
      3. Emit it to ``memory_sink`` if ``causal_write_policy`` allows.
    """

    def __init__(
        self,
        runtime: ExternalToolRuntime,
        contract_registry: dict[str, ToolExecutionContract] | None = None,
        *,
        memory_sink: Any = None,
        policy_engine: Any = None,
        causal_write_policy: CausalWritePolicy = CausalWritePolicy.ALWAYS,
    ) -> None:
        self._runtime = runtime
        self._contract_registry = dict(contract_registry or {})
        self._memory_sink = memory_sink
        self._policy_engine = policy_engine
        self._causal_write_policy = causal_write_policy

    def register_contract(self, contract: ToolExecutionContract) -> None:
        """Register a contract for a tool."""
        key = f"{contract.tool_name}@{contract.version}".lower()
        self._contract_registry[key] = contract

    def get_contract(self, tool_name: str, version: str | None = None) -> ToolExecutionContract | None:
        """Retrieve contract for tool; if version not specified, returns latest."""
        if version is not None:
            key = f"{tool_name}@{version}".lower()
            return self._contract_registry.get(key)

        # Return highest version for tool_name
        prefix = f"{tool_name}@"
        matching = {
            k: v for k, v in self._contract_registry.items() if k.lower().startswith(prefix.lower())
        }
        if not matching:
            return None
        return list(matching.values())[-1]  # Assumes sorted insertion; crude but works for small registries

    def execute_with_contract(
        self,
        tool_name: str,
        payload: dict[str, Any],
        *,
        contract: ToolExecutionContract | None = None,
        required_version: str | None = None,
        validate_input: bool = True,
        validate_output: bool = True,
        current_attempt: int = 1,
    ) -> ToolExecutionResult:
        """Execute tool with contract validation, failure classification, and causal memory write.

        Args:
            tool_name: Tool identifier
            payload: Execution input
            contract: Contract to validate against; if None, attempts lookup via tool_name
            required_version: Specific version requirement
            validate_input: Whether to pre-flight validate input
            validate_output: Whether to post-flight validate output
            current_attempt: 1-based attempt counter; forwarded into the causal entry

        Returns:
            ToolExecutionResult with failure_class and taxonomy metadata enriched.

        Raises:
            ToolExecutionContractViolation: If contract validation fails (input, output, or undocumented failure).

        Side effects:
            When a memory_sink is configured, emits a CausalMemoryEntry to long-term memory
            carrying full causal provenance (tool, version, attempt, status, failure_class,
            policy_action).
        """
        # Resolve contract
        if contract is None:
            contract = self.get_contract(tool_name, version=required_version)

        # Pre-flight: input validation
        if validate_input and contract is not None:
            valid, msg = validate_tool_execution_input(contract, payload)
            if not valid:
                raise ToolExecutionContractViolation(f"{tool_name}: {msg}")

        # Execute via runtime
        result = self._runtime.execute(
            tool_name,
            payload,
            required_version=required_version,
        )

        # Post-flight: output validation
        if validate_output and contract is not None and result.status == ToolExecutionStatus.OK:
            valid, msg = validate_tool_execution_output(contract, result.output)
            if not valid:
                raise ToolExecutionContractViolation(f"{tool_name}: {msg}")

        # Enrich result with contract failure classification
        if contract is not None and result.status != ToolExecutionStatus.OK:
            result = normalize_result_failure_class(contract, result)

        # --- Tool→Memory causal write loop ---
        if self._memory_sink is not None and contract is not None:
            policy_action: str | None = None
            if self._policy_engine is not None and result.status != ToolExecutionStatus.OK:
                try:
                    decision = self._policy_engine.decide(result, contract, current_attempt)
                    policy_action = decision.action.value
                except Exception:
                    # Policy engine failure must never break the execution path
                    pass
            entry = build_causal_entry(result, contract, current_attempt, policy_action=policy_action)
            emit_causal_entry(self._memory_sink, entry, self._causal_write_policy)

        return result


__all__ = [
    "ContractAwareToolRuntime",
    "ToolExecutionContractViolation",
    "classify_runtime_failure_to_contract_failure",
    "normalize_result_failure_class",
    "tool_execution_status_to_failure_class",
    "validate_tool_execution_input",
    "validate_tool_execution_output",
    "CausalWritePolicy",
]
