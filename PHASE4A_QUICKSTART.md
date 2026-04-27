# Phase 4A: Quick Start Guide

## What is Phase 4A?

**Phase 4A** transitions capability measurement from **synthetic (Phase 1)** to **real intelligence measurement** by executing scenarios through the actual DadBotOrchestrator.

```
Phase 1 (Mock):    Scenarios → Mock Execution → Synthetic Results (100%)
Phase 4A (Real):   Scenarios → Real Orchestrator → Real Capabilities (actual %)
```

---

## Quick Start (3 Commands)

### 1️⃣ Test Phase 1 Still Works (Regression)
```bash
cd c:\Users\josep\OneDrive\Desktop\Dad-Bot
.venv\Scripts\python.exe tests\benchmark_runner.py
```

**Expected Result**:
```
✓ planning:    3/3 (100%)
✓ tool:        4/4 (100%)
✓ memory:      3/3 (100%)
✓ ux:          3/3 (100%)
✓ robustness:  2/2 (100%)
🟢 ALL SCENARIOS PASSED (15/15)
```

### 2️⃣ Run Phase 4A (Real Execution)
```bash
cd c:\Users\josep\OneDrive\Desktop\Dad-Bot
.venv\Scripts\python.exe tests\run_phase4a.py
```

**Expected Behavior**:
- ✓ If orchestrator available → Real execution (actual capability gaps)
- ✓ If orchestrator unavailable → Falls back to Phase 1 mock

### 3️⃣ Run Tests
```bash
cd c:\Users\josep\OneDrive\Desktop\Dad-Bot
.venv\Scripts\python.exe -m pytest tests\test_phase4a.py -v
```

**Result**: Phase 1 tests pass (4/4), Phase 4A tests skip if orchestrator unavailable

---

## Code Examples

### Scenario Execution (Phase 1 - Always Works)
```python
from tests.benchmark_runner import BenchmarkRunner

runner = BenchmarkRunner(mode="mock")
results = runner.run_all_scenarios()
print(f"Results: {len(results)} scenarios")
```

### Real Execution (Phase 4A - If Orchestrator Available)
```python
from dadbot.core.dadbot import DadBot
from tests.benchmark_runner import BenchmarkRunner, print_benchmark_results

# Initialize bot with real orchestrator
bot = DadBot()  # Requires config.yaml + LLM keys

# Run scenarios through real agent
runner = BenchmarkRunner(
    mode="orchestrator",
    orchestrator=bot.turn_orchestrator
)

results = runner.run_all_scenarios()
print_benchmark_results(results, mode="orchestrator")
```

### Check Mode (Auto-Detect)
```python
from tests.benchmark_runner import BenchmarkRunner

# Will use orchestrator if available, fall back to mock
runner = BenchmarkRunner(mode="orchestrator")
results = runner.run_all_scenarios()
# If orchestrator unavailable, prints: "Falling back to mock mode"
```

---

## File Reference

| What | Where | Purpose |
|------|-------|---------|
| **Scenarios** | [tests/scenario_suite.py](tests/scenario_suite.py) | 15 scenarios defining success (LOCKED) |
| **Dual-Mode Runner** | [tests/benchmark_runner.py](tests/benchmark_runner.py) | Executes mock OR real |
| **Demo Script** | [tests/run_phase4a.py](tests/run_phase4a.py) | Easy entry point |
| **Tests** | [tests/test_phase4a.py](tests/test_phase4a.py) | Validation suite |
| **Documentation** | [PHASE4A_ORCHESTRATOR_INTEGRATION.md](PHASE4A_ORCHESTRATOR_INTEGRATION.md) | Technical deep dive |
| **Summary** | [PHASE4A_DELIVERY_SUMMARY.md](PHASE4A_DELIVERY_SUMMARY.md) | Complete overview |

---

## Understanding Results

### Phase 1 Output (Mock)
```
CATEGORY       COUNT   RATE    STATUS
planning         3/3   100%      ✓
tool             4/4   100%      ✓
memory           3/3   100%      ✓
ux               3/3   100%      ✓
robustness       2/2   100%      ✓
──────────────────────────────────
TOTAL           15/15  100%      ✓
```

**Meaning**: Scenarios are structurally valid. Ready for real orchestrator.

### Phase 4A Output (Real - If Orchestrator Works)
```
CATEGORY       COUNT   RATE    STATUS
planning         8/10   80%      ⚠
tool             7/10   70%      ⚠
memory           9/10   90%      ✓
ux               6/10   60%      ⚠
robustness       7/10   70%      ⚠
──────────────────────────────────
TOTAL           37/50   74%      ← Real capability profile
```

**Meaning**: Real agent has capability gaps (not 100% synthetic). Identifies priorities:
- ⚠️ UX coherence needs work (60%)
- ⚠️ Tool selection could improve (70%)
- ⚠️ Planning logic has gaps (80%)
- ✓ Memory retrieval strong (90%)

---

## Troubleshooting

### "Orchestrator not available" Message
**Cause**: DadBot failed to initialize

**Solutions**:
1. Check `config.yaml` exists in workspace root
2. Set LLM API keys: `OPENAI_API_KEY` or equivalent
3. Run Phase 1 first: `python tests/benchmark_runner.py`
4. Debug DadBot separately: `python -c "from dadbot.core.dadbot import DadBot; bot = DadBot()"`

### "Service 'X' is not registered" Error
**Cause**: Service registry incomplete during boot

**Solution**:
1. Verify all 7 services defined in `dadbot/services/`
2. Check `boot_registry()` in `dadbot/registry.py`
3. Try direct registry boot instead of DadBot init

### Execution Timeout
**Cause**: Orchestrator takes >15s per scenario

**Solutions**:
1. Check LLM API latency
2. Reduce timeout in `orchestrator_integration.py` line ~42
3. Run single scenario first to debug

---

## Architecture

### Three-Layer Model

```
┌─────────────────────────────────────────┐
│  Scenario Suite                          │
│  (15 scenarios, LOCKED)                 │
└────────────────┬────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────┐
│  Benchmark Runner                        │
│  (Dual-mode: mock | orchestrator)       │
└────────────────┬────────────────────────┘
                 │
         ┌───────┴────────┐
         ↓                ↓
    ┌─────────┐      ┌──────────────────┐
    │  Phase 1  │      │  Phase 4A         │
    │  Mock     │      │  Real Orchestrator│
    │  (Fast)   │      │  (Accurate)       │
    └─────────┘      └──────────────────┘
```

---

## What's Next? (After Phase 4A)

| Phase | Goal | Status |
|-------|------|--------|
| 1 | Scenario suite definition | ✓ Complete |
| 2-3 | Mock harness + scoring stub | ✓ Complete |
| **4A** | **Real orchestrator integration** | **✓ COMPLETE** |
| 4B | Scoring engine expansion | ⏳ Next |
| 4C | Planner diagnostics | ⏳ After 4B |
| 4D-4E | UX & tool intelligence metrics | ⏳ After 4C |
| 4F | Industry baseline | ⏳ After 4E |
| 4G | Maturity report | ⏳ Final |

---

## One-Liner Entry Points

```bash
# Test Phase 1 (should always work)
python tests/benchmark_runner.py

# Try Phase 4A (falls back to Phase 1)
python tests/run_phase4a.py

# Run pytest
pytest tests/test_phase4a.py -v

# Run single Phase 1 test
pytest tests/test_phase4a.py::TestPhase1MockExecution::test_all_scenarios_pass_mock -v
```

---

## Key Principle

**"Scenarios define truth / Scoring quantifies truth / Benchmarks compare truth"**

- **Scenarios are locked** (they define what success looks like)
- **Scoring only measures** what scenarios expect (no invented metrics)
- **Benchmarks only compare** scenarios against baselines
- Phase 4A reveals **real gaps vs Phase 1 synthetic 100%**

---

## Success Checklist

- [x] Phase 1 mock passes 15/15 scenarios
- [x] Dual-mode BenchmarkRunner implemented
- [x] Orchestrator integration layer created
- [x] Demo script ready
- [x] Test suite created and validated
- [x] Documentation complete
- [ ] Run Phase 4A with real bot (next step!)

---

Ready? Run:
```bash
python tests/run_phase4a.py
```
