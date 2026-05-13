# Phase 1: Response Authority Collapse — Audit Matrix

## Authority Split Rule
- **Decision Authority**: ResponseEngine (candidate generation, scoring, ranking, selection)
- **Enforcement Authority**: Control-plane (gating, safety, final output shaping, persistence)
- **Hot Path**: Only ONE system produces final response = ResponseEngine selected answer
- **Everything else**: Telemetry-only observers

---

## Path Classification

| Path Location | Current Behavior | Classification | Action | Status |
|---|---|---|---|---|
| `control_plane._finalize_turn_with_response_engine()` L3144-3165 | Invokes ResponseEngine.run(), persists telemetry, commits response | **HOT PATH** | ✅ Preserve, enforce single decision source | ✅ COMPLETE |
| `response_engine.run()` L589-650 | Orchestrates generate→score→select, attaches telemetry | **HOT PATH - DECISION** | ✅ Preserve, this is the selection authority | ✅ COMPLETE |
| `reply_generation.generate_validated_reply()` / `generate_validated_reply_async()` | Fail-fast compatibility shim; records telemetry and raises authority-disabled error | **NON-AUTHORITY PATH** | ✅ Preserve for compatibility telemetry; block output production | ✅ COMPLETE |
| `reply_finalization.finalize()` L45-150 | Post-generation formatting: personality voice, moderation, audit, signoff | **ENFORCEMENT PATH** | ⏳ Keep callable, but enforce gating rule: only control-plane invokes | ⏳ IN PROGRESS |
| `reply_finalization.append_signoff()` L15-30 | Appends style signoff to reply text | **ENFORCEMENT LAYER** | ✅ Keep, but restrict to formatting-only calls from control-plane | ⏳ IN PROGRESS |
| `runtime_interface.append_signoff()` L794, 805, 857, 868 | Four fallback signoff calls in chat loop error handlers | **LEGACY PATH** | ⏳ Disable hot-path routing, preserve for compatibility, emit telemetry | ⏳ IN PROGRESS |
| `safety.crisis_support_reply()` + `append_signoff()` L242, 245 | Crisis intervention directly produces formatted response | **OBSERVER PATH** | ⏳ Convert to signal producer, feed crisis signal to control-plane gating | ⏳ IN PROGRESS |
| `agentic.py` line 717 | Calls `bot.reply_finalization.finalize()` | **OBSERVER PATH** | ⏳ Verify does not bypass control-plane in hot path | ⏳ PENDING |
| `app_runtime.py` line 547 | Calls `reply_finalization.append_signoff()` | **OBSERVER PATH** | ⏳ Verify scope and restrict if in hot path | ⏳ PENDING |

---

## Enforcement Rules (Phase 1)

**Rule 1: Single Response Source**
- ✅ Only `ResponseEngine.run()` may generate/rank response candidates
- ❌ No other system may independently influence response selection
- ✅ Control-plane gates the ResponseEngine result before delivery

**Rule 2: Telemetry-Only Compliance**
- All non-ResponseEngine response producers must:
  - Track their output in telemetry (not suppress it)
  - Not affect final_response value
  - Log their signals to observability system
  - Remain callable for compatibility

**Rule 3: Control-Plane as Single Enforcer**
- ✅ Only control-plane invokes ResponseEngine
- ✅ Only control-plane applies gating rules
- ✅ Only control-plane commits final response to persistence
- ❌ Never invoke ResponseEngine from reply_generation, runtime_interface, or safety paths

**Rule 4: Formatting vs. Selection**
- **Formatting** (reply_finalization methods): applies signoff, moderation, audit
  - Called from control-plane post-response selection
  - Non-binding (does not change which response is selected)
- **Selection** (ResponseEngine): generates candidates, scores, ranks, selects
  - Only allowed system to influence which response is chosen
  - Called by control-plane only

---

## Test Suite (Phase 1)

### Core Test: Single Response Authority Per Turn
- Assert: exactly ONE response path produces `final_response` per turn
- Assert: that path is ResponseEngine → control-plane → persistence
- Assert: all other paths are telemetry-only (do not set final_response)

### Control-Plane Integration Test
- Assert: control-plane successfully invokes ResponseEngine
- Assert: ResponseEngine result is used as final_response (not replaced by other systems)
- Assert: telemetry is attached to job metadata

### Determinism Test
- Run same input 3 times
- Assert: same response selected each time (ResponseEngine ranking deterministic)
- Assert: response source trace identical each time

### Fallback Path Isolation
- Assert: runtime_interface.append_signoff() calls do not affect turn execution path
- Assert: safety crisis paths emit signals but do not produce final_response in normal flow

---

## Patches Required

1. **reply_generation.py**: Convert finalize calls to emit telemetry only
2. **runtime_interface.py**: Disable hot-path routing of signoff calls
3. **safety.py**: Route crisis signals to control-plane gating (not direct response)
4. **reply_finalization.py**: Add assertion: finalize() only callable from control-plane
5. **control_plane.py**: Add telemetry tracking for non-ResponseEngine signal producers

---

## Success Criteria

Phase 1 is **COMPLETE** when:

- ✅ Exactly one response path per turn produces `final_response`
- ✅ That path is: ResponseEngine → control-plane → persistence
- ✅ All non-ResponseEngine systems are telemetry-only observers
- ✅ Core execution tests pass (fail-fast)
- ✅ Response selection tests pass
- ✅ Control-plane integration tests pass
- ✅ Determinism test passes
- ✅ All changes committed atomically as single "Phase 1: Response Authority Collapse" commit
