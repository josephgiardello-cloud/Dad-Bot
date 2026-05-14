"""Tests for tool execution contract: schema validation, failure taxonomy, replay semantics."""

import pytest

from dadbot.core.tool_execution_contract import (
    DEFAULT_GENERIC_TOOL_CONTRACT,
    FailureClass,
    FailureSeverity,
    ReplaySemantics,
    ToolExecutionContract,
    ToolInputSchema,
    ToolOutputSchema,
    taxonomy_entry_for,
)
from dadbot.core.tool_execution_contract_binding import (
    ContractAwareToolRuntime,
    ToolExecutionContractViolation,
    classify_runtime_failure_to_contract_failure,
    normalize_result_failure_class,
    tool_execution_status_to_failure_class,
    validate_tool_execution_input,
    validate_tool_execution_output,
)
from dadbot.core.external_tool_runtime import (
    NetworkFailureKind,
    ToolExecutionResult,
    ToolExecutionStatus,
)


class TestFailureTaxonomy:
    """Test unified failure taxonomy and classification."""

    def test_taxonomy_entry_retrieval(self):
        """All failure classes have taxonomy entries."""
        for failure_class in FailureClass:
            entry = taxonomy_entry_for(failure_class)
            assert entry.failure_class == failure_class
            assert entry.severity in FailureSeverity
            assert isinstance(entry.retryable, bool)

    def test_transient_failures_are_retryable(self):
        """Transient failures have retryable=True."""
        transient_classes = {
            FailureClass.NETWORK_TIMEOUT,
            FailureClass.NETWORK_RATE_LIMIT,
            FailureClass.NETWORK_TEMPORARY,
            FailureClass.NETWORK_UNKNOWN,
        }
        for fc in transient_classes:
            entry = taxonomy_entry_for(fc)
            assert entry.severity == FailureSeverity.TRANSIENT
            assert entry.retryable is True

    def test_permanent_failures_are_not_retryable(self):
        """Permanent failures have retryable=False."""
        permanent_classes = {
            FailureClass.CLIENT_ERROR,
            FailureClass.PERMISSION_DENIED,
            FailureClass.RESOURCE_UNAVAILABLE,
            FailureClass.ISOLATION_VIOLATED,
            FailureClass.TOOL_INTERNAL_ERROR,
            FailureClass.SCHEMA_VALIDATION,
            FailureClass.TIMEOUT_EXCEEDED,
        }
        for fc in permanent_classes:
            entry = taxonomy_entry_for(fc)
            assert entry.severity == FailureSeverity.PERMANENT
            assert entry.retryable is False

    def test_partial_failures_are_not_retryable(self):
        """Partial/degraded failures have retryable=False but are usable."""
        partial_classes = {FailureClass.PARTIAL_OUTPUT, FailureClass.DEGRADED_CONFIDENCE}
        for fc in partial_classes:
            entry = taxonomy_entry_for(fc)
            assert entry.severity == FailureSeverity.PARTIAL
            assert entry.retryable is False

    def test_unknown_failure_defaults_transient(self):
        """Unknown failures default to transient for conservative retrying."""
        entry = taxonomy_entry_for(FailureClass.UNKNOWN)
        assert entry.severity == FailureSeverity.UNKNOWN
        assert entry.retryable is True  # Conservative: try again


class TestInputSchema:
    """Test input schema validation."""

    def test_valid_input_passes(self):
        """Valid input satisfies schema."""
        schema = ToolInputSchema(
            required_fields=frozenset({"action"}),
            optional_fields=frozenset({"metadata"}),
            field_types={"action": str, "metadata": dict},
        )
        payload = {"action": "get_data", "metadata": {"key": "value"}}
        valid, msg = schema.validate(payload)
        assert valid
        assert msg == "ok"

    def test_missing_required_field_fails(self):
        """Missing required field fails validation."""
        schema = ToolInputSchema(
            required_fields=frozenset({"action", "target"}),
            optional_fields=frozenset(),
            field_types={"action": str, "target": str},
        )
        payload = {"action": "get_data"}  # Missing "target"
        valid, msg = schema.validate(payload)
        assert not valid
        assert "missing required field" in msg

    def test_unexpected_field_fails(self):
        """Unexpected field fails validation."""
        schema = ToolInputSchema(
            required_fields=frozenset({"action"}),
            optional_fields=frozenset(),
            field_types={"action": str},
        )
        payload = {"action": "get_data", "unknown_field": "value"}
        valid, msg = schema.validate(payload)
        assert not valid
        assert "unexpected field" in msg

    def test_wrong_type_fails(self):
        """Wrong field type fails validation."""
        schema = ToolInputSchema(
            required_fields=frozenset({"action"}),
            optional_fields=frozenset(),
            field_types={"action": str},
        )
        payload = {"action": 123}  # Should be str
        valid, msg = schema.validate(payload)
        assert not valid
        assert "wrong type" in msg

    def test_non_dict_payload_fails(self):
        """Non-dict payload fails validation."""
        schema = ToolInputSchema(
            required_fields=frozenset({"action"}),
            optional_fields=frozenset(),
            field_types={"action": str},
        )
        valid, msg = schema.validate("not a dict")
        assert not valid
        assert "must be dict" in msg


class TestOutputSchema:
    """Test output schema validation."""

    def test_valid_output_passes(self):
        """Valid output satisfies schema."""
        schema = ToolOutputSchema(
            required_fields=frozenset({"result"}),
            optional_fields=frozenset({"metadata"}),
            field_types={"result": (str, dict), "metadata": dict},
        )
        output = {"result": "success", "metadata": {"status": "ok"}}
        valid, msg = schema.validate(output)
        assert valid
        assert msg == "ok"

    def test_missing_required_output_field_fails(self):
        """Missing required output field fails."""
        schema = ToolOutputSchema(
            required_fields=frozenset({"result"}),
            optional_fields=frozenset(),
            field_types={"result": str},
        )
        output = {"status": "ok"}  # Missing "result"
        valid, msg = schema.validate(output)
        assert not valid
        assert "missing required field in output" in msg


class TestReplaySemantics:
    """Test idempotency and cache validity."""

    def test_indefinite_cache_always_valid(self):
        """Cache with ttl=0 is always valid."""
        semantics = ReplaySemantics(
            idempotency_key_factors=frozenset({"action"}),
            policy_context_factors=frozenset({"permissions"}),
            cache_ttl_seconds=0,
            determinism_guarantee="strict",
        )
        assert semantics.is_replay_valid(cached_result_age_seconds=0)
        assert semantics.is_replay_valid(cached_result_age_seconds=1000)
        assert semantics.is_replay_valid(cached_result_age_seconds=1e10)

    def test_ttl_based_cache_expiry(self):
        """Cache expires after TTL."""
        semantics = ReplaySemantics(
            idempotency_key_factors=frozenset({"action"}),
            policy_context_factors=frozenset(),
            cache_ttl_seconds=10.0,
            determinism_guarantee="strict",
        )
        assert semantics.is_replay_valid(cached_result_age_seconds=5)
        assert not semantics.is_replay_valid(cached_result_age_seconds=11)
        assert not semantics.is_replay_valid(cached_result_age_seconds=10.1)

    def test_negative_ttl_disables_cache(self):
        """Negative TTL disables cache."""
        semantics = ReplaySemantics(
            idempotency_key_factors=frozenset({"action"}),
            policy_context_factors=frozenset(),
            cache_ttl_seconds=-1,
            determinism_guarantee="none",
        )
        assert not semantics.is_replay_valid(cached_result_age_seconds=0)
        assert not semantics.is_replay_valid(cached_result_age_seconds=1)


class TestToolExecutionContract:
    """Test formal tool execution contract."""

    def test_contract_validates_input(self):
        """Contract validates input against schema."""
        contract = ToolExecutionContract(
            tool_name="test_tool",
            version="1.0.0",
            input_schema=ToolInputSchema(
                required_fields=frozenset({"query"}),
                optional_fields=frozenset(),
                field_types={"query": str},
            ),
            output_schema=ToolOutputSchema(
                required_fields=frozenset({"result"}),
                optional_fields=frozenset(),
                field_types={"result": str},
            ),
            replay_semantics=ReplaySemantics(
                idempotency_key_factors=frozenset({"query"}),
                policy_context_factors=frozenset(),
                cache_ttl_seconds=0,
                determinism_guarantee="strict",
            ),
        )

        valid, msg = contract.validate_input({"query": "test"})
        assert valid
        assert msg == "ok"

        valid, msg = contract.validate_input({})  # Missing required "query"
        assert not valid

    def test_contract_validates_output(self):
        """Contract validates output against schema."""
        contract = ToolExecutionContract(
            tool_name="test_tool",
            version="1.0.0",
            input_schema=ToolInputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            output_schema=ToolOutputSchema(
                required_fields=frozenset({"result"}),
                optional_fields=frozenset(),
                field_types={"result": str},
            ),
            replay_semantics=ReplaySemantics(
                idempotency_key_factors=frozenset(),
                policy_context_factors=frozenset(),
                cache_ttl_seconds=0,
                determinism_guarantee="strict",
            ),
        )

        valid, msg = contract.validate_output({"result": "success"})
        assert valid

        valid, msg = contract.validate_output({})  # Missing required "result"
        assert not valid

    def test_contract_failure_classification(self):
        """Contract classifies failures against supported set."""
        contract = ToolExecutionContract(
            tool_name="test_tool",
            version="1.0.0",
            input_schema=ToolInputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            output_schema=ToolOutputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            replay_semantics=ReplaySemantics(
                idempotency_key_factors=frozenset(),
                policy_context_factors=frozenset(),
                cache_ttl_seconds=0,
                determinism_guarantee="strict",
            ),
            supported_failure_classes=frozenset(
                {FailureClass.NETWORK_TIMEOUT, FailureClass.PERMISSION_DENIED},
            ),
        )

        entry = contract.classify_failure(FailureClass.NETWORK_TIMEOUT)
        assert entry.failure_class == FailureClass.NETWORK_TIMEOUT

        with pytest.raises(ValueError, match="does not document support"):
            contract.classify_failure(FailureClass.CLIENT_ERROR)


class TestFailureClassification:
    """Test runtime failure mapping to contract failure classes."""

    def test_network_failure_classification(self):
        """Network failures map to contract failure classes."""
        assert (
            classify_runtime_failure_to_contract_failure(
                runtime_failure_kind=NetworkFailureKind.TIMEOUT,
            )
            == FailureClass.NETWORK_TIMEOUT
        )
        assert (
            classify_runtime_failure_to_contract_failure(
                runtime_failure_kind=NetworkFailureKind.RATE_LIMIT,
            )
            == FailureClass.NETWORK_RATE_LIMIT
        )
        assert (
            classify_runtime_failure_to_contract_failure(
                runtime_failure_kind=NetworkFailureKind.DNS,
            )
            == FailureClass.NETWORK_TEMPORARY
        )

    def test_http_status_classification(self):
        """HTTP status codes map to contract failure classes."""
        assert (
            classify_runtime_failure_to_contract_failure(
                runtime_failure_kind=NetworkFailureKind.CLIENT,
                http_status=403,
            )
            == FailureClass.PERMISSION_DENIED
        )
        assert (
            classify_runtime_failure_to_contract_failure(
                runtime_failure_kind=NetworkFailureKind.CLIENT,
                http_status=400,
            )
            == FailureClass.CLIENT_ERROR
        )

    def test_tool_status_classification(self):
        """Tool execution statuses map to failure classes."""
        assert (
            tool_execution_status_to_failure_class(ToolExecutionStatus.PARTIAL)
            == FailureClass.PARTIAL_OUTPUT
        )
        assert (
            tool_execution_status_to_failure_class(ToolExecutionStatus.DEGRADED)
            == FailureClass.DEGRADED_CONFIDENCE
        )
        assert (
            tool_execution_status_to_failure_class(ToolExecutionStatus.TIMEOUT)
            == FailureClass.TIMEOUT_EXCEEDED
        )


class TestResultNormalization:
    """Test result normalization with contract taxonomy."""

    def test_normalize_error_result_enriches_metadata(self):
        """Error result gets enriched with failure class metadata."""
        contract = ToolExecutionContract(
            tool_name="test_tool",
            version="1.0.0",
            input_schema=ToolInputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            output_schema=ToolOutputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            replay_semantics=ReplaySemantics(
                idempotency_key_factors=frozenset(),
                policy_context_factors=frozenset(),
                cache_ttl_seconds=0,
                determinism_guarantee="strict",
            ),
            supported_failure_classes=frozenset({FailureClass.TOOL_INTERNAL_ERROR}),
        )

        result = ToolExecutionResult(
            tool_name="test_tool",
            status=ToolExecutionStatus.ERROR,
            output=None,
            error="handler_failed",
            attempts=1,
            latency_ms=100.0,
            confidence=0.0,
        )

        normalized = normalize_result_failure_class(contract, result)
        assert normalized.metadata["failure_class"] == FailureClass.TOOL_INTERNAL_ERROR.value
        assert normalized.metadata["failure_severity"] == FailureSeverity.PERMANENT.value
        assert normalized.metadata["failure_retryable"] is False
        assert "escalation_key" in normalized.metadata

    def test_ok_result_unchanged(self):
        """OK result is not modified."""
        contract = DEFAULT_GENERIC_TOOL_CONTRACT

        result = ToolExecutionResult(
            tool_name="test_tool",
            status=ToolExecutionStatus.OK,
            output={"data": "result"},
            error="",
            attempts=1,
            latency_ms=50.0,
            confidence=1.0,
        )

        normalized = normalize_result_failure_class(contract, result)
        assert normalized.status == ToolExecutionStatus.OK
        assert "failure_class" not in normalized.metadata


class TestContractValidationFunctions:
    """Test validation helper functions."""

    def test_validate_input_success(self):
        """Input validation succeeds for valid payload."""
        schema = ToolInputSchema(
            required_fields=frozenset({"action"}),
            optional_fields=frozenset(),
            field_types={"action": str},
        )
        contract = ToolExecutionContract(
            tool_name="test",
            version="1.0.0",
            input_schema=schema,
            output_schema=ToolOutputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            replay_semantics=ReplaySemantics(
                idempotency_key_factors=frozenset(),
                policy_context_factors=frozenset(),
                cache_ttl_seconds=0,
                determinism_guarantee="strict",
            ),
        )

        valid, msg = validate_tool_execution_input(contract, {"action": "test"})
        assert valid
        assert msg == "ok"

        valid, msg = validate_tool_execution_input(contract, {})
        assert not valid
        assert "Contract violation" in msg

    def test_validate_output_success(self):
        """Output validation succeeds for valid output."""
        schema = ToolOutputSchema(
            required_fields=frozenset({"result"}),
            optional_fields=frozenset(),
            field_types={"result": str},
        )
        contract = ToolExecutionContract(
            tool_name="test",
            version="1.0.0",
            input_schema=ToolInputSchema(
                required_fields=frozenset(),
                optional_fields=frozenset(),
                field_types={},
            ),
            output_schema=schema,
            replay_semantics=ReplaySemantics(
                idempotency_key_factors=frozenset(),
                policy_context_factors=frozenset(),
                cache_ttl_seconds=0,
                determinism_guarantee="strict",
            ),
        )

        valid, msg = validate_tool_execution_output(contract, {"result": "success"})
        assert valid
        assert msg == "ok"

        valid, msg = validate_tool_execution_output(contract, {})
        assert not valid
        assert "Contract violation" in msg


class TestDefaultGenericContract:
    """Test default generic contract for testing and fallback."""

    def test_generic_contract_exists(self):
        """Default generic contract is available."""
        assert DEFAULT_GENERIC_TOOL_CONTRACT is not None
        assert DEFAULT_GENERIC_TOOL_CONTRACT.tool_name == "generic"

    def test_generic_contract_accepts_any_input(self):
        """Generic contract is permissive on input."""
        valid, msg = DEFAULT_GENERIC_TOOL_CONTRACT.validate_input(
            {"anything": "goes", "custom_field": 123},
        )
        assert valid

    def test_generic_contract_accepts_any_output(self):
        """Generic contract is permissive on output."""
        valid, msg = DEFAULT_GENERIC_TOOL_CONTRACT.validate_output(
            {"any": "output", "structure": {"nested": True}},
        )
        assert valid
