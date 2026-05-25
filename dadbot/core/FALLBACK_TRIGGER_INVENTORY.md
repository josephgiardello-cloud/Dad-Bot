"""dadbot/core/FALLBACK_TRIGGER_INVENTORY.md

Enumeration of all fallback trigger points in Dad-Bot that could cause
thin-spine execution to diverge from legacy reference behavior.

CRITICALITY: **HIGH** — Each trigger is a potential contract violation point.
"""

# FALLBACK TRIGGER INVENTORY

## Summary
**Total Identified Triggers: 9 primary categories**
**Risk Level: CRITICAL for convergence validation**

---

## 1. PATH-LEVEL FALLBACK (turn_mixin.py, lines 110-135)

### Trigger Point
```python
# Line 110: PRIMARY BRANCHING DECISION
if thin_turn_handler_enabled() and callable(submit_turn):
    # THIN-SPINE PATH: TurnHandler wrapper
    handler = TurnHandler(submit_turn=submit_turn)
    return await handler.process_turn(...)
    
# Line 120: FALLBACK #1 - Legacy _submit_turn_via_control_plane
if callable(submit_turn):
    return await submit_turn(user_input, ...)
    
# Line 130: FALLBACK #2 - Direct control_plane.submit_turn
return await orchestrator.control_plane.submit_turn(...)
```

### Failure Conditions
1. `thin_turn_handler_enabled()` returns False
    - **Deprecated condition**: in the shipped runtime, thin-spine is canonical and this helper is effectively always enabled.
    - The historical environment variable `DADBOT_USE_THIN_TURN_HANDLER` is legacy-only and should not be relied on for routing control.
2. `TurnHandler` not callable
3. `_submit_turn_via_control_plane` not available on orchestrator

### Contract Risk
- **CRITICAL**: Entire execution path changes
- Different error handling
- Different middleware stacks
- Potential state mutation order changes

---

## 2. RESPONSE ENGINE FALLBACK (control_plane.py, lines 3485-3562)

### Trigger Point #1 (lines 3485-3487)
```python
ranked_response = self._response_engine.run(engine_context)
final_response = str(ranked_response or "")
if not final_response:
    fallback_response = getattr(self._response_engine, "_fallback_response", None)
    if callable(fallback_response):
        final_response = str(fallback_response(engine_context) or "")
```

### Trigger Point #2 (lines 3551-3553)
```python
# Duplicate fallback in finalization path
if not final_response:
    fallback_response = getattr(self._response_engine, "_fallback_response", None)
    if callable(fallback_response):
        final_response = str(fallback_response(engine_context) or "")
```

### Trigger Point #3 (lines 3559-3562)
```python
# Exception recovery fallback
except Exception as exc:
    logger.warning(f"ResponseEngine ranking failed: {exc}; using ResponseEngine fallback")
    fallback_response = getattr(self._response_engine, "_fallback_response", None)
    if callable(fallback_response):
        final_response = str(fallback_response(engine_context) or "")
    else:
        final_response = initial_response  # HARD FALLBACK to unranked response
```

### Failure Conditions
1. `_response_engine.run()` returns None/empty
2. `_response_engine.run()` raises Exception
3. ResponseEngine ranking produces no candidates

### Contract Risk
- **CRITICAL**: Changes semantic output
- `_fallback_response` may use different persona/tone rules
- Hard fallback to `initial_response` loses all response ranking

---

## 3. EXECUTION MODE ROUTING (turn_mixin.py, lines 419-441)

### Trigger Point
```python
# Line 419
if request.mode == ExecutionMode.LIVE:
    # PRIMARY: Execute via orchestrator
    return ...

# Line 425: FALLBACK - Replay mode
if request.mode == ExecutionMode.REPLAY:
    if state is None:
        # FALLBACK #1: No state available
        return ...
    if not callable(replay_handler):
        # FALLBACK #2: No replay handler
        return await orchestrator.handle_turn(...)

# Line 441: FALLBACK - Recovery mode
if request.mode == ExecutionMode.RECOVERY:
    # Different execution contract
    return ...
```

### Failure Conditions
1. ExecutionMode != LIVE
2. Replay without state snapshot
3. Recovery mode triggered

### Contract Risk
- **HIGH**: Different execution pipeline
- Replay may use cached state (diverges from live)
- Recovery mode uses fallback persona

---

## 4. RESPONSE AUTHORITY FALLBACK (control_plane.py, line 3584-3605)

### Trigger Point
```python
if current_execution_result.get("status") not in _TERMINAL_STATUS_VALUES:
    current_execution_result = mark_unified_execution_success(...)
    job.metadata["response_authority"] = "response_engine"
    telemetry_container = dict(job.metadata.get("response_engine_telemetry") or {})
    selected = dict(telemetry_container.get("selected") or {})
    
    if not selected:
        # FALLBACK: No response engine metadata
        selected = {
            "source": "response_engine_fallback",
            "score": 0.0,
            "reason": "No ranked candidate metadata available; fallback selected by response engine.",
        }
```

### Failure Conditions
1. ResponseEngine produces no telemetry
2. No "selected" key in telemetry
3. Status != terminal

### Contract Risk
- **MEDIUM**: Changes response attribution
- May skip tone/persona validation
- Fallback source attribution changes downstream analysis

---

## 5. DISTRIBUTED AUTHORITY FALLBACK (control_plane.py, lines 560-620)

### Trigger Point
```python
# Line 565
"is_fallback": "fallback" in selected_source,

# Line 614-615
if rates["fallback_rate"] >= 0.15:
    anomalies.append("fallback_frequency_high")
    # System may enter RECOVERY mode
```

### Failure Conditions
1. Fallback rate exceeds 15% in anomaly window
2. Authority lease expired (triggers _sync_distributed_authority)
3. Distributed correctness validation fails

### Contract Risk
- **HIGH**: May trigger recovery mode
- Changes execution authority
- May invoke persona fallback

---

## 6. EXCEPTION MAPPER FALLBACK (control_plane.py, lines 650-668)

### Trigger Point
```python
class SchedulerExceptionMapper:
    @staticmethod
    def from_exception(exc: BaseException) -> TurnTerminalState:
        # Maps any exception to terminal state
        # Falls back to generic error persona
```

### Failure Conditions
1. Any unhandled exception during turn execution
2. Scheduler worker failure
3. Asyncio task cancellation

### Contract Risk
- **CRITICAL**: Transforms output to error state
- Error persona != success persona
- May suppress original error details

---

## 7. LEGACY COMPATIBILITY FALLBACK (control_plane.py, line 1731)

### Trigger Point
```python
# Line 1731
# Legacy fallback for older callers; unified execution_result is authoritative.
```

### Failure Conditions
1. Caller using deprecated API
2. execution_result not properly unified
3. Old metadata schema used

### Contract Risk
- **MEDIUM**: May decode old formats
- Different metadata interpretation
- Compatibility layer overhead

---

## 8. RESPONSE SHAPING FALLBACK (control_plane.py, implied in continuity shaping)

### Trigger Point
```python
shaped_response, continuity_telemetry = self._apply_continuity_shaping(
    response_text=final_response,
    session_state=continuity_state,
)
final_response = str(shaped_response or final_response)  # FALLBACK
```

### Failure Conditions
1. Continuity shaping returns None/empty
2. Session state missing/corrupted
3. Tone modulation rules fail

### Contract Risk
- **MEDIUM**: Changes tone/voice
- May alter response semantics
- Persona rules applied inconsistently

---

## 9. NODE RETRYABILITY FALLBACK (nodes.py, line 934)

### Trigger Point
```python
# Line 934
except registration.retryable_exceptions as exc:
    # Retry tool invocation
    # May produce different results on retry
```

### Failure Conditions
1. Tool call raises retryable exception
2. Retry budget exhausted
3. Backoff strategy triggers

### Contract Risk
- **MEDIUM**: Tool results may differ
- Different state mutations
- Timing-dependent behavior

---

## CRITICAL INVARIANT VIOLATIONS AT EACH TRIGGER

For **strict convergence validation**, ALL triggers must be:

1. **Captured** at execution time
2. **Logged** with input hash
3. **Compared** between paths
4. **Asserted** as equivalent

### Per-Trigger Validation

| Trigger | Output Impact | State Impact | Tool Impact | Rank |
|---------|---------------|-------------|------------|------|
| Path-Level | YES | YES | YES | 1 |
| Response Engine | YES | NO | NO | 2 |
| Exec Mode | YES | YES | YES | 3 |
| Response Authority | YES | NO | NO | 4 |
| Distributed Authority | YES | YES | NO | 5 |
| Exception Mapper | YES | YES | NO | 6 |
| Legacy Compat | MAYBE | NO | NO | 7 |
| Response Shaping | YES | NO | NO | 8 |
| Node Retry | MAYBE | YES | YES | 9 |

---

## ENFORCEMENT STRATEGY

### Phase 1: Detection (CURRENT)
- ✅ Identify all triggers
- ✅ Create trigger inventory
- Document activation conditions

### Phase 2: Instrumentation (NEXT)
- Wrap each trigger with hook
- Record trigger activation
- Tag turn with trigger bits

### Phase 3: Comparison (VALIDATION)
- Execute both paths
- Compare trigger activation
- Assert identical trigger sequences

### Phase 4: Lockdown (PRODUCTION)
- Make triggers enumerated/deterministic
- Disable runtime trigger branching
- Use feature flags only at boot time

---

## DANGER ZONE: DO NOT ADD NEW FALLBACKS

Any new exception handler, conditional branch, or error recovery path
that executes DIFFERENT CODE between paths is a contract violation.

When adding new functionality:
1. Execute identically on both paths
2. Or add feature flag (decision at boot time only)
3. Or add to this inventory and re-validate contract

---

## REFERENCE
- **BehaviorContractLock**: dadbot/core/behavior_contract_lock.py
- **EquivalenceValidator**: dadbot/core/equivalence_validator.py
- **Turn Mixin Branching**: dadbot/core/turn_mixin.py lines 100-135
- **Response Engine**: dadbot/core/control_plane.py lines 3485-3605
