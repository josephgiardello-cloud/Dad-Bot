"""
Unit tests for execution_result_unified semantic invariants.

Invariants enforced:
  1. success ⇒ failure fields are all empty
  2. failure ⇒ outputs dict exists AND failure.class is non-empty
  3. timeout ⇒ failure.class == "timeout" AND timeout.timed_out == True (bidirectional)
  4. degradation is always present and well-typed (even if empty)

Freeze semantics enforced:
  F. Once status is ok/failed, any top-level key write raises ExecutionResultInvariantError.
"""
from __future__ import annotations

import copy
from typing import Any

import pytest

from dadbot.core.execution_result_unified import (
    ExecutionResultInvariantError,
    _FrozenExecutionResult,
    assert_execution_result_invariants,
    build_unified_execution_result,
    ensure_unified_execution_result,
    mark_unified_execution_failure,
    mark_unified_execution_success,
    set_unified_execution_result,
)

pytestmark = pytest.mark.unit


# ─── helpers ─────────────────────────────────────────────────────────────────

def _fresh() -> dict[str, Any]:
    return build_unified_execution_result()


def _ok(response: str = "hello") -> dict[str, Any]:
    return mark_unified_execution_success(_fresh(), response=response, should_end=False)


def _failed(failure_class: str = "runtime_exception") -> dict[str, Any]:
    return mark_unified_execution_failure(
        _fresh(),
        failure_class=failure_class,
        failure_source="execution",
        retryable=False,
        exception_type="RuntimeError",
        message="something went wrong",
    )


def _timeout() -> dict[str, Any]:
    return mark_unified_execution_failure(
        _fresh(),
        failure_class="timeout",
        failure_source="infrastructure",
        retryable=True,
        exception_type="TimeoutError",
        message="timed out",
    )


# ─── Invariant 1: success ⇒ failure fields empty ─────────────────────────────

class TestInvariant1SuccessImpliesNoFailure:
    def test_ok_with_clean_failure_fields_passes(self):
        er = _ok()
        assert_execution_result_invariants(er)  # must not raise

    def test_ok_with_dirty_class_raises(self):
        er = _ok()
        er["failure"]["class"] = "something"
        with pytest.raises(ExecutionResultInvariantError, match="Invariant 1"):
            assert_execution_result_invariants(er)

    def test_ok_with_dirty_message_raises(self):
        er = _ok()
        er["failure"]["message"] = "stale message"
        with pytest.raises(ExecutionResultInvariantError, match="Invariant 1"):
            assert_execution_result_invariants(er)

    def test_ok_with_dirty_source_raises(self):
        er = _ok()
        er["failure"]["source"] = "execution"
        with pytest.raises(ExecutionResultInvariantError, match="Invariant 1"):
            assert_execution_result_invariants(er)

    def test_mark_success_clears_failure_fields_automatically(self):
        # Build a mutable pending envelope that looks like it had failure state,
        # then mark success — failure fields must be cleared.
        er = _fresh()
        er["failure"]["class"] = "transient"   # sub-key write on inner dict (allowed)
        er["failure"]["message"] = "oops"
        result = mark_unified_execution_success(er, response="recovered", should_end=False)
        assert result["failure"]["class"] == ""
        assert result["failure"]["message"] == ""

    def test_pending_with_non_empty_failure_fields_passes_invariant1(self):
        # pending is not "ok", so invariant 1 does not apply
        er = _fresh()
        er["failure"]["class"] = "partial"
        assert_execution_result_invariants(er)  # must not raise (only "ok" triggers I-1)


# ─── Invariant 2: failure ⇒ outputs present + failure.class non-empty ────────

class TestInvariant2FailureImpliesOutputsAndClass:
    def test_failed_with_outputs_and_class_passes(self):
        er = _failed()
        assert_execution_result_invariants(er)  # must not raise

    def test_failed_with_missing_outputs_raises(self):
        er = dict(_failed())    # mutable copy
        del er["outputs"]
        with pytest.raises(ExecutionResultInvariantError, match="Invariant 2"):
            assert_execution_result_invariants(er)

    def test_failed_with_none_outputs_raises(self):
        er = dict(_failed())    # mutable copy
        er["outputs"] = None
        with pytest.raises(ExecutionResultInvariantError, match="Invariant 2"):
            assert_execution_result_invariants(er)

    def test_failed_with_empty_failure_class_raises(self):
        er = _failed()
        er["failure"]["class"] = ""
        with pytest.raises(ExecutionResultInvariantError, match="Invariant 2"):
            assert_execution_result_invariants(er)

    def test_failed_outputs_may_be_partial(self):
        er = _failed()
        # outputs may have empty response — that is explicitly allowed
        er["outputs"]["response"] = ""
        assert_execution_result_invariants(er)  # must not raise


# ─── Invariant 3: timeout consistency ────────────────────────────────────────

class TestInvariant3TimeoutConsistency:
    def test_timeout_failure_sets_timed_out_flag(self):
        er = _timeout()
        assert er["timeout"]["timed_out"] is True
        assert er["failure"]["class"] == "timeout"
        assert_execution_result_invariants(er)  # must not raise

    def test_timed_out_true_with_wrong_class_raises(self):
        er = _failed("runtime_exception")
        er["timeout"]["timed_out"] = True
        with pytest.raises(ExecutionResultInvariantError, match="Invariant 3"):
            assert_execution_result_invariants(er)

    def test_timeout_class_without_timed_out_flag_raises(self):
        er = _timeout()
        er["timeout"]["timed_out"] = False   # manually corrupt the flag
        with pytest.raises(ExecutionResultInvariantError, match="Invariant 3"):
            assert_execution_result_invariants(er)

    def test_timed_out_true_with_empty_class_passes(self):
        # pending state: timed_out might be set without a failure class
        er = _fresh()
        er["timeout"]["timed_out"] = True
        # empty failure.class → invariant 3 only rejects *wrong* non-empty class
        assert_execution_result_invariants(er)  # must not raise

    def test_non_timeout_failure_with_timed_out_false_passes(self):
        er = _failed("lease_conflict")
        assert er["timeout"]["timed_out"] is False
        assert_execution_result_invariants(er)  # must not raise


# ─── Invariant 4: degradation always present ─────────────────────────────────

class TestInvariant4DegradationAlwaysPresent:
    def test_fresh_envelope_has_degradation(self):
        er = _fresh()
        assert isinstance(er["degradation"], dict)
        assert isinstance(er["degradation"]["items"], list)
        assert_execution_result_invariants(er)

    def test_missing_degradation_raises(self):
        er = _fresh()
        del er["degradation"]
        with pytest.raises(ExecutionResultInvariantError, match="Invariant 4"):
            assert_execution_result_invariants(er)

    def test_degradation_not_dict_raises(self):
        er = _fresh()
        er["degradation"] = []
        with pytest.raises(ExecutionResultInvariantError, match="Invariant 4"):
            assert_execution_result_invariants(er)

    def test_degradation_missing_items_key_raises(self):
        er = _fresh()
        del er["degradation"]["items"]
        with pytest.raises(ExecutionResultInvariantError, match="Invariant 4"):
            assert_execution_result_invariants(er)

    def test_degradation_items_not_list_raises(self):
        er = _fresh()
        er["degradation"]["items"] = {"bad": "shape"}
        with pytest.raises(ExecutionResultInvariantError, match="Invariant 4"):
            assert_execution_result_invariants(er)

    def test_ensure_always_restores_degradation(self):
        raw = {"status": "ok", "failure": {}, "outputs": {}, "timeout": {}}
        er = ensure_unified_execution_result(raw)
        assert isinstance(er["degradation"], dict)
        assert isinstance(er["degradation"]["items"], list)


# ─── mark_unified_execution_failure: invalid class raises at production ───────

class TestMarkFailureRejectsEmptyClass:
    def test_empty_failure_class_raises_on_mark(self):
        with pytest.raises(ExecutionResultInvariantError, match="Invariant 2"):
            mark_unified_execution_failure(
                _fresh(),
                failure_class="",       # ← invalid
                failure_source="execution",
                retryable=False,
                exception_type="RuntimeError",
                message="boom",
            )


# ─── ensure_unified_execution_result: round-trip fidelity ────────────────────

class TestEnsureRoundTrip:
    def test_ok_roundtrip_is_invariant_safe(self):
        er = _ok("hi")
        assert_execution_result_invariants(ensure_unified_execution_result(er))

    def test_failed_roundtrip_is_invariant_safe(self):
        er = _failed()
        assert_execution_result_invariants(ensure_unified_execution_result(er))

    def test_timeout_roundtrip_is_invariant_safe(self):
        er = _timeout()
        assert_execution_result_invariants(ensure_unified_execution_result(er))

    def test_pending_roundtrip_is_invariant_safe(self):
        er = _fresh()
        assert_execution_result_invariants(ensure_unified_execution_result(er))


# ─── Freeze semantics ─────────────────────────────────────────────────────────

class TestFreezeSemantics:
    """Post-terminal envelopes must be immutable at the top-level key layer."""

    # ── type contract ──────────────────────────────────────────────────────────

    def test_mark_success_returns_frozen_instance(self):
        assert isinstance(_ok(), _FrozenExecutionResult)

    def test_mark_failure_returns_frozen_instance(self):
        assert isinstance(_failed(), _FrozenExecutionResult)

    def test_timeout_failure_returns_frozen_instance(self):
        assert isinstance(_timeout(), _FrozenExecutionResult)

    def test_frozen_is_still_a_dict(self):
        # isinstance check must pass for all existing callers
        assert isinstance(_ok(), dict)

    def test_pending_envelope_is_plain_dict(self):
        assert type(_fresh()) is dict

    # ── ensure does NOT re-freeze (it is a normalization utility) ───────────────

    def test_ensure_of_frozen_ok_returns_plain_dict(self):
        # ensure is a normalizer — callers may legitimately re-normalize terminal
        # dicts (e.g. when reading back from storage).
        stored = dict(_ok())   # plain dict copy as stored in session state
        re_read = ensure_unified_execution_result(stored)
        assert type(re_read) is dict  # plain, NOT frozen

    def test_ensure_of_frozen_failed_returns_plain_dict(self):
        stored = dict(_failed())
        re_read = ensure_unified_execution_result(stored)
        assert type(re_read) is dict

    def test_ensure_does_not_freeze_pending(self):
        er = ensure_unified_execution_result(None)
        assert type(er) is dict  # plain, mutable

    # ── write-protection on frozen top-level keys ──────────────────────────────

    def test_frozen_status_write_raises(self):
        er = _ok()
        with pytest.raises(ExecutionResultInvariantError, match="frozen"):
            er["status"] = "pending"

    def test_frozen_failure_write_raises(self):
        er = _ok()
        with pytest.raises(ExecutionResultInvariantError, match="frozen"):
            er["failure"] = {}

    def test_frozen_outputs_write_raises(self):
        er = _ok()
        with pytest.raises(ExecutionResultInvariantError, match="frozen"):
            er["outputs"] = {}

    def test_frozen_delete_raises(self):
        er = _ok()
        with pytest.raises(ExecutionResultInvariantError, match="frozen"):
            del er["status"]

    def test_frozen_arbitrary_key_write_raises(self):
        er = _failed()
        with pytest.raises(ExecutionResultInvariantError, match="frozen"):
            er["extra_field"] = "should fail"

    # ── re-marking a frozen result raises ─────────────────────────────────────

    def test_re_mark_success_on_frozen_raises(self):
        er = _ok()
        with pytest.raises(ExecutionResultInvariantError, match="terminal"):
            mark_unified_execution_success(er, response="new", should_end=False)

    def test_re_mark_failure_on_frozen_raises(self):
        er = _failed()
        with pytest.raises(ExecutionResultInvariantError, match="terminal"):
            mark_unified_execution_failure(
                er,
                failure_class="new_failure",
                failure_source="execution",
                retryable=False,
                exception_type="RuntimeError",
                message="re-marking frozen result",
            )

    def test_re_mark_failure_on_plain_dict_ok_raises(self):
        # Even a plain dict copy with status=ok must be refused.
        er = dict(_ok())
        with pytest.raises(ExecutionResultInvariantError, match="terminal"):
            mark_unified_execution_failure(
                er,
                failure_class="cascade",
                failure_source="execution",
                retryable=False,
                exception_type="RuntimeError",
                message="cascade failure",
            )

    # ── dict(frozen) produces mutable copy ────────────────────────────────────

    def test_dict_copy_of_frozen_is_mutable(self):
        er = _ok()
        mutable = dict(er)
        assert type(mutable) is dict
        mutable["status"] = "pending"  # must not raise

    def test_dict_copy_preserves_all_keys(self):
        er = _ok("the response")
        mutable = dict(er)
        assert mutable["status"] == "ok"
        assert mutable["outputs"]["response"] == "the response"

    # ── set_unified_execution_result strips freeze for storage ─────────────────

    def test_set_strips_freeze_for_attribute_storage(self):
        er = _ok("stored")

        class FakeTurnContext:
            execution_result = None
            metadata = {}
            state = {}

        ctx = FakeTurnContext()
        set_unified_execution_result(ctx, er)
        # Stored version must be plain dict (serialisable/patchable by callers)
        assert type(ctx.execution_result) is dict
        assert type(ctx.metadata["execution_result"]) is dict

    def test_get_after_set_returns_plain_dict(self):
        # set_unified_execution_result strips the freeze; get returns a fresh
        # plain dict for callers to work with (freeze is enforced by mark functions,
        # not by get).
        from dadbot.core.execution_result_unified import get_unified_execution_result

        er = _ok("round-trip")

        class FakeTurnContext:
            execution_result = None
            metadata: dict = {}
            state: dict = {}

        ctx = FakeTurnContext()
        set_unified_execution_result(ctx, er)
        re_read = get_unified_execution_result(ctx)
        assert isinstance(re_read, dict)
        assert re_read["status"] == "ok"
