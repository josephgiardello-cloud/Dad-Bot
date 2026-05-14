# Priority 1: IO Determinism Sealing

**Status**: PLANNING PHASE  
**Goal**: Eliminate external IO nondeterminism by recording all tool IO at execution time and replaying from recordings.

## Overview

The system currently executes tools (reminders, web searches) in a way that **mutates external state and requests live data** every time a turn replays. This breaks determinism:

- **set_reminder**: Creates duplicate reminders on replay
- **web_search**: Makes new HTTP requests, potentially getting different results
- **Any IO-based tool**: Mutates external systems or retrieves nondeterministic data

**Solution**: Record all tool input/output at execution time, store in checkpoint, and during replay return **recorded outputs instead of executing**.

---

## Architecture: Four-Layer Model

```
┌─────────────────────────────────────────────────────────┐
│ Layer 1: TOOL EXECUTION RECORDING (NEW)                 │
│ - Capture tool input + output at execute_tool() time    │
│ - Build immutable ToolIORecord for each call            │
│ - Append to checkpoint under tool_io_ledger[]           │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 2: REPLAY MODE DETECTION (NEW)                    │
│ - Flag: turn_context.metadata["replay_mode"] = bool     │
│ - Set during checkpoint restore (before orchestrator)   │
│ - Available in execute_tool() for conditional routing   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 3: REPLAY INJECTION LAYER (NEW)                   │
│ - In execute_tool(): if replay_mode, look up recorded   │
│   output by (tool_name, input_hash) from tool_io_ledger │
│ - Return recorded ToolIORecord instead of executing     │
│ - Mark as "replayed" in status                          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│ Layer 4: SIDE-EFFECT GUARD (HARDENING)                  │
│ - Executor receives replay_context={replay_mode, ...}   │
│ - If replay_mode=True, executor MUST NOT mutate state   │
│ - If it tries, ToolIORecorder logs + counters violation │
│ - Post-turn assert: no side effects during replay       │
└─────────────────────────────────────────────────────────┘
```

---

## Core Components to Build

### 1. **ToolIORecord** (Immutable, serializable)

```python
@dataclass(frozen=True)
class ToolIORecord:
    """Complete recorded IO for one tool execution."""
    sequence: int
    tool_name: str
    input_hash: str                    # stable hash of {tool_name, parameters}
    input_payload: dict[str, Any]      # raw parameters
    output_payload: dict[str, Any]     # actual tool output
    output_hash: str
    status: str                        # "ok" | "error" | "partial" | "replayed"
    latency_ms: float
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def is_replay_hit(self) -> bool:
        return self.status == "replayed"
    
    def matches_request(self, tool_name: str, input_hash: str) -> bool:
        return self.tool_name == tool_name and self.input_hash == input_hash
```

### 2. **ToolIOLedger** (Sequential record store)

```python
@dataclass
class ToolIOLedger:
    """Sequential log of all tool IO in a turn."""
    records: list[ToolIORecord] = field(default_factory=list)
    _by_request_hash: dict[str, ToolIORecord] = field(default_factory=dict)
    
    def append(self, record: ToolIORecord) -> None:
        self.records.append(record)
        # Index by (tool_name, input_hash) for O(1) replay lookup
        key = f"{record.tool_name}:{record.input_hash}"
        self._by_request_hash[key] = record
    
    def lookup(self, tool_name: str, input_hash: str) -> ToolIORecord | None:
        """Fast lookup for replay injection."""
        key = f"{tool_name}:{input_hash}"
        return self._by_request_hash.get(key)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [asdict(r) for r in self.records],
            "sequence_count": len(self.records),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolIOLedger:
        ledger = cls()
        for rec_dict in data.get("records", []):
            record = ToolIORecord(**rec_dict)
            ledger.append(record)
        return ledger
```

### 3. **Modified execute_tool()** (Layer 1 + Layer 3)

```python
def execute_tool(
    *,
    tool_name: str,
    parameters: dict[str, Any] | None = None,
    executor: Callable[[], Any],
    compensating_action: Callable[[], None] | None = None,
    turn_context: Any | None = None,
) -> ToolExecutionRecord:
    """
    Execute a tool with IO recording + replay injection.
    
    If replay_mode=True:
      - Lookup recorded output from checkpoint's tool_io_ledger
      - Return recorded output WITHOUT executing
      - Mark as "replayed"
    
    If replay_mode=False (live):
      - Execute the tool normally
      - Record output to turn_context for checkpoint persistence
    """
    sandbox = _ToolSandbox()
    started = time.perf_counter()
    
    # LAYER 3: Check replay mode
    replay_mode = bool(
        getattr(turn_context, "metadata", {}).get("replay_mode", False)
    )
    input_hash = _stable_payload_hash({
        "tool_name": str(tool_name or ""),
        "parameters": dict(parameters or {}),
    })
    
    # Replay injection: bypass executor, return recorded output
    if replay_mode:
        tool_io_ledger = getattr(turn_context, "_tool_io_ledger", None)
        if isinstance(tool_io_ledger, ToolIOLedger):
            recorded = tool_io_ledger.lookup(tool_name, input_hash)
            if recorded is not None:
                # Return replayed record
                replay_record = ToolExecutionRecord(
                    tool_name=tool_name,
                    idempotency_key=_idempotency_key(tool_name, dict(parameters or {})),
                    status="replayed",
                    result=recorded.output_payload,
                    error="",
                    compensating_action=None,
                )
                _emit_tool_execution_event(
                    turn_context=turn_context,
                    tool_name=tool_name,
                    parameters=parameters,
                    record=replay_record,
                    latency_ms=recorded.latency_ms,
                )
                return replay_record
    
    # LAYER 1: Execute normally and record
    record = sandbox.execute(
        tool_name=tool_name,
        parameters=parameters,
        executor=executor,
        compensating_action=compensating_action,
    )
    
    latency_ms = (time.perf_counter() - started) * 1000.0
    
    # Record to tool IO ledger (for checkpoint)
    if isinstance(turn_context, dict) or hasattr(turn_context, "metadata"):
        tool_io_ledger = getattr(turn_context, "_tool_io_ledger", None)
        if tool_io_ledger is None:
            tool_io_ledger = ToolIOLedger()
            setattr(turn_context, "_tool_io_ledger", tool_io_ledger)
        
        io_record = ToolIORecord(
            sequence=len(tool_io_ledger.records) + 1,
            tool_name=tool_name,
            input_hash=input_hash,
            input_payload=dict(parameters or {}),
            output_payload=record.result if record.status == "succeeded" else {},
            output_hash=_stable_payload_hash(record.result if record.status == "succeeded" else {}),
            status=record.status,
            latency_ms=latency_ms,
            error=record.error,
        )
        tool_io_ledger.append(io_record)
    
    if turn_context is not None:
        _emit_tool_execution_event(
            turn_context=turn_context,
            tool_name=tool_name,
            parameters=parameters,
            record=record,
            latency_ms=latency_ms,
        )
    
    return record
```

### 4. **Checkpoint Integration** (Store + restore tool_io_ledger)

In `execution_checkpoint.py`:

```python
@dataclass
class ExecutionCheckpoint:
    """Includes tool IO ledger for deterministic replay."""
    # ... existing fields ...
    tool_io_ledger: ToolIOLedger = field(default_factory=ToolIOLedger)
    
    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["tool_io_ledger"] = self.tool_io_ledger.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionCheckpoint:
        ledger_data = data.pop("tool_io_ledger", {})
        checkpoint = cls(**data)
        checkpoint.tool_io_ledger = ToolIOLedger.from_dict(ledger_data)
        return checkpoint
```

In `PersistenceService`:

```python
def load_latest_graph_checkpoint(self, trace_token: str = "", **kwargs: Any) -> dict[str, Any] | None:
    checkpoint = self.persistence_manager.load_checkpoint(...)
    if checkpoint is None:
        return None
    
    # Extract tool_io_ledger from checkpoint
    ledger_dict = checkpoint.get("tool_io_ledger", {})
    tool_io_ledger = ToolIOLedger.from_dict(ledger_dict)
    checkpoint["_tool_io_ledger"] = tool_io_ledger
    
    return checkpoint
```

### 5. **Replay Mode Flag** (Layer 2)

In `TurnGraph.execute_turn()`:

```python
def execute_turn(self, turn_request: TurnRequest) -> TurnResponse:
    # Determine if replaying from checkpoint
    is_replay = self._is_restoring_from_checkpoint()
    
    turn_context.metadata["replay_mode"] = is_replay
    turn_context._tool_io_ledger = (
        checkpoint.get("_tool_io_ledger", ToolIOLedger())
        if checkpoint else ToolIOLedger()
    )
    
    # ... execute orchestrator ...
```

### 6. **Side-Effect Guard** (Layer 4)

In `_tool_sandbox.py`:

```python
class _ToolSandbox:
    def __init__(self, replay_mode: bool = False) -> None:
        self._replay_mode = replay_mode
        self._side_effect_violations: list[str] = []
    
    def execute(
        self,
        *,
        tool_name: str,
        parameters: dict[str, Any] | None = None,
        executor: Callable[[], Any],
        compensating_action: Callable[[], None] | None = None,
    ) -> ToolExecutionRecord:
        # Wrap executor to pass replay context
        replay_context = {
            "replay_mode": self._replay_mode,
            "tool_name": tool_name,
        }
        
        def wrapped_executor():
            # Executors that check replay_context can skip mutations
            return executor()
        
        # ... existing logic ...
```

---

## Rollout Plan

### Phase 1: Infrastructure (Week 1)
- [ ] Define `ToolIORecord` + `ToolIOLedger` dataclasses
- [ ] Add to `dadbot/core/tool_recording.py` (new file)
- [ ] Integrate with checkpoint struct
- [ ] Add serialization tests

### Phase 2: Record Layer (Week 1–2)
- [ ] Modify `execute_tool()` to build `ToolIORecord`
- [ ] Append to turn_context._tool_io_ledger
- [ ] Emit to events for observability
- [ ] Test: verify records are created for set_reminder, web_search

### Phase 3: Replay Mode Flag (Week 2)
- [ ] Add `metadata["replay_mode"]` to turn_context
- [ ] Set during checkpoint restore
- [ ] Test: verify flag is set correctly on replay paths

### Phase 4: Replay Injection (Week 2–3)
- [ ] Modify `execute_tool()` to check replay mode
- [ ] Lookup recorded output from ledger
- [ ] Return recorded output instead of executing
- [ ] Mark status as "replayed"
- [ ] Test: manual replay, verify no re-execution

### Phase 5: Side-Effect Guard (Week 3)
- [ ] Pass replay_context to executors
- [ ] Add guard in _ToolSandbox
- [ ] Test: catch violations in strict mode
- [ ] Assert no side effects on replay paths

### Phase 6: Validation (Week 3–4)
- [ ] Test determinism: identical inputs → identical outputs
- [ ] Test replay safety: no tool re-execution
- [ ] Test checkpoint round-trip
- [ ] Run soak tests with replay validation
- [ ] Run cert tests

---

## Key Invariants to Maintain

1. **Immutability**: ToolIORecords are frozen dataclasses
2. **Determinism**: Input hash uniquely identifies the recorded output
3. **Isolation**: Replay injection happens ONLY if replay_mode=True
4. **Ordering**: tool_io_ledger maintains sequence order
5. **No side effects on replay**: Executors MUST not mutate during replay
6. **Idempotency**: Multiple replays of same checkpoint produce identical behavior

---

## Testing Strategy

```python
# Determinism test: same input → same output (both live + replay)
def test_tool_execution_determinism():
    ctx1, output1 = execute_turn("remind me in 5 min", delivery=SYNC)
    ctx2, output2 = replay_turn(ctx1.checkpoint, delivery=SYNC)
    assert output1 == output2
    assert ctx2.metadata["replay_mode"] == True

# No re-execution test
def test_replay_does_not_execute_tool():
    ctx = execute_turn("remind me tomorrow", delivery=SYNC)
    reminder_count_before = count_reminders()
    
    replay_turn(ctx.checkpoint, delivery=SYNC)
    reminder_count_after = count_reminders()
    
    assert reminder_count_after == reminder_count_before  # No new reminder

# Replay injection test
def test_replay_injection_returns_recorded_output():
    live_ctx = execute_turn("search for coffee recipes", delivery=SYNC)
    live_output = live_ctx.response.reply
    
    replay_ctx = replay_turn(live_ctx.checkpoint, delivery=SYNC)
    replay_output = replay_ctx.response.reply
    
    assert live_output == replay_output
    assert replay_ctx.tool_io_ledger.records[0].status == "replayed"

# Checkpoint round-trip
def test_tool_io_ledger_checkpoint_round_trip():
    ctx = execute_turn("...", delivery=SYNC)
    checkpoint = ctx.checkpoint
    
    ledger_before = checkpoint["tool_io_ledger"]
    restored_ledger = ToolIOLedger.from_dict(ledger_before)
    
    assert len(restored_ledger.records) == len(ledger_before["records"])
    assert restored_ledger.lookup("set_reminder", hash) is not None
```

---

## Success Criteria

✅ **Level 1**: tool_io_ledger persists across checkpoints  
✅ **Level 2**: Replay mode flag correctly set  
✅ **Level 3**: Replay injection bypasses executor  
✅ **Level 4**: No side effects during replay (strict mode asserts)  
✅ **Level 5**: Soak test: 10k turns with replay validation pass  
✅ **Level 6**: Cert test: all regression tests pass  
✅ **Level 7**: Durability test: orchestrator restart + replay produces identical behavior  

---

## Related Issues

- **Execution context boundary**: Where does replay_mode get set? Answer: TurnGraph._restore_from_checkpoint()
- **Memory consistency**: Do we need memory search during replay? Answer: Yes, but it's read-only, so it's allowed
- **Deterministic latency**: Do we use recorded latency or live measurement? Answer: Use recorded latency (deterministic)
- **Error replay**: Do we replay errors or re-execute? Answer: Replay status=error, but don't execute compensating actions

---

## Notes for Future Phases

**Phase 2+ (after IO sealing):**
- Extend to external reminders (DB writes)
- Extend to external web calls (caching layer)
- Add cost tracking (metered tool usage)
- Add observability layer (tool execution dashboard)
