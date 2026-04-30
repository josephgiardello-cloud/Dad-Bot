# Dad-Bot Code Quality Audit Report

**Date:** 2025-04-30  
**Suite:** 1568 passed / 0 failed (33 skipped, 2 xpassed) — FULL CERT passing  
**Scope:** `dadbot/` source tree (excludes `tests/`, `tools/`, `.tmp_*` temp files)

---

## 1. Linting — Ruff (`--select ALL`)

| Metric | Value |
|---|---|
| Total findings (dadbot/) | **17,529** |
| Auto-fixable (`--fix`) | 1,266 |
| Unsafe-fixable (`--unsafe-fixes`) | +729 |

### Top Categories by Volume

| Count | Code | Category |
|---|---|---|
| 5,838 | W191 | **Tab indentation** (highest priority — consistent style failure) |
| 3,212 | E501 | Line too long (>88 chars) |
| 1,540 | D102 | Undocumented public method |
| 1,243 | ANN001 | Missing type annotation (function arg) |
| 565 | ANN201 | Missing return type annotation |
| 468 | COM812 | Missing trailing comma |
| 432 | ANN401 | Bare `Any` type hint |
| 236 | SLF001 | Private member access |
| **222** | **BLE001** | **Blind `except Exception` catch** ← action item |
| 213 | D107 | Undocumented `__init__` |
| **211** | **TRY003** | **Raise with long inline message** ← action item |
| 199 | PLR2004 | Magic value comparison |
| **77** | **PLC0415** | **Import outside top level** ← deferred imports to audit |
| **75** | **C901** | **Complex structure** ← complexity flag |
| 62 | PLR0913 | Too many function arguments |
| 58 | F401 | Unused import |
| 38 | PLR0911 | Too many return statements |
| 37 | PLR0912 | Too many branches |

### Formatting
- `ruff format --check dadbot/`: **274 files would be reformatted**, 10 already formatted.
- Recommended: run `ruff format dadbot/` for a single clean pass.

---

## 2. Technical Debt Markers

| Pattern | Count |
|---|---|
| `TODO` | **0** |
| `FIXME` | **0** |
| `HACK` / `XXX` | **0** |
| `stub` (functional, e.g. `minimal_streamlit_stub_source`) | benign |
| `Phase 4.x` comments | informational/implemented |

**Zero open TODO/FIXME markers.** The codebase is clean of explicit debt flags.

---

## 3. Complexity — Radon CC

Average across analyzable files: **C (14.95)** in `dadbot/core/`, **D (27.9)** across all `dadbot/`.

### Grade F (Untestable — CC > 30)
| Function | File |
|---|---|
| `RelationshipManager._project_state_from_graph_state` | `dadbot/relationship.py` |
| `CritiqueEngine.critique` | `dadbot/core/critic.py` |
| `InferenceNode._handle_delegation` | `dadbot/core/nodes.py` |
| `DadBotOrchestrator._execute_job` | `dadbot/core/orchestrator.py` |
| `ContextService.build_context` | `dadbot/services/context_service.py` |

### Grade E (CC 21–30)
| Function | File |
|---|---|
| `app_runtime.stop_streamlit_app` | `dadbot/app_runtime.py` |
| `LongTermSignalsManager.deep_pattern_documents` | `dadbot/managers/long_term.py` |
| `LongTermSignalsManager.deep_pattern_matches` | `dadbot/managers/long_term.py` |
| `MemoryCoordinator.merge_consolidated_memories` | `dadbot/managers/memory_coordination.py` |
| `SemanticIndexManager.embed_texts` | `dadbot/memory/index_manager.py` |
| `MediaService.render_realtime_voice_call` | `dadbot/ui/media/service.py` |
| `MediaService.render_voice_controls` | `dadbot/ui/media/service.py` |
| `runtime.launcher.stop_streamlit_app` | `dadbot/runtime/launcher.py` |

### Grade D (CC 11–20) — notable entries
- `TurnGraph.execute` / `_run_stage` — `dadbot/core/graph.py`
- `InferenceNode.run` / `_run_subtask` — `dadbot/core/nodes.py`
- `ConversationPersistenceManager._fold_events` — `dadbot/managers/conversation_persistence.py`

---

## 4. Maintainability — Radon MI

Files rated **B** (satisfactory, but complex enough to watch):
- `dadbot/core/execution_trace_context.py`
- `dadbot/core/external_tool_runtime.py`

All other analyzable files passed. Previously-blocked files (167 BOM-infected) are now readable (see §6).

---

## 5. Dead Code — Vulture (80%+ confidence)

| Location | Issue |
|---|---|
| `dadbot/core/durability.py:362` | unused variable `exc_tb` |
| `dadbot/core/execution_ledger.py:30` | unused variable `tb` |
| `dadbot/core/execution_resource_budget.py:184` | **unsatisfiable ternary condition** |
| `dadbot/core/fault_injection.py:283` | unused variable `exc_tb` |
| `dadbot/core/nodes.py:18` | unused import `reduce_events_to_results` |
| `dadbot/core/observability.py:29` | unused import `Iterator` |
| `dadbot/core/tool_sandbox.py:29` | unused import `Iterator` |
| `dadbot/managers/maintenance.py:720` | **unreachable code after `return`** |
| `dadbot/mood.py:231` | **unsatisfiable ternary condition** |
| `dadbot/mood.py:237` | **unsatisfiable ternary condition** |
| `dadbot/ui/helpers.py:65` | unused variable `context_key` |

**Highest priority:** the 3 unsatisfiable conditions in `execution_resource_budget.py` and `mood.py` — these are likely logic bugs.

60% confidence: ~600 lines of potentially unused methods/classes across `contracts.py`, `causal_graph.py`, `authorization.py`, etc. — likely intentional extensibility APIs but worth a future pass.

---

## 6. Encoding Issues — UTF-8 BOM

**167 `.py` files** in `dadbot/` had UTF-8 BOM (`\xEF\xBB\xBF`) prepended, causing `radon` parse errors and `W191` tab-indentation false positives. **All 167 BOMs were stripped this session.** Tests still pass (409 unit / 1568 full cert).

---

## 7. Security — Ruff S-series

| Code | Count | Finding | Verdict |
|---|---|---|---|
| S608 | 3 | Hardcoded SQL expression | **False positive** — parameterized `?,?,?` placeholders built from counts, values passed as params |
| S301 | 2 | `pickle` usage | Review needed — used in causal_graph, conflict_resolution |
| S324 | 15 | Insecure hash (MD5/SHA1) | Likely for non-security hashing (deduplication keys); verify |
| S311 | 2 | Non-cryptographic random | OK if not security-sensitive |
| S310 | 6 | `urllib.urlopen` | Acceptable for internal HTTP; verify no user-controlled URLs |
| S607 | 8 | Partial path in subprocess | Review — ensure no PATH injection vectors |
| S112 | 9 | `try/except/continue` | Swallowed errors; convert to logged warnings |
| S101 | 1 | `assert` in production code | Remove or raise proper error |
| S314 | 1 | `xml.etree.ElementTree` | Low risk if parsing internal data only |

---

## 8. Import Topology

`pydeps` requires Graphviz (not installed). AST-based inspection shows no intra-`dadbot/core` circular imports — all cross-imports go through the top-level `dadbot` package namespace as expected.

---

## 9. Change Hotspots (6-month git churn)

| Commits | File |
|---|---|
| 19 | `dadbot/core/graph.py` |
| 14 | `dadbot/core/orchestrator.py` |
| 12 | `dadbot/core/nodes.py` |
| 11 | `dadbot/services/persistence.py` |
| 7 | `dadbot/core/dadbot.py`, `dadbot/services/turn_service.py` |
| 6 | `dadbot/relationship.py`, `dadbot/context.py` |

These files (especially `graph.py` + `orchestrator.py`) also appear in the worst-complexity list — **highest refactor ROI targets**.

---

## 10. Recommended Priority Actions

### P0 — Logic / Safety
1. **Investigate unsatisfiable conditions** in `dadbot/mood.py:231,237` and `dadbot/core/execution_resource_budget.py:184` — likely dead branches masking logic bugs.
2. **Unreachable code** at `dadbot/managers/maintenance.py:720` — remove or fix flow.
3. **Unused imports** (`reduce_events_to_results`, `Iterator` × 2) — clean up.

### P1 — Complexity
4. Extract sub-functions from **F-grade methods**: `_handle_delegation`, `_execute_job`, `build_context`, `critique`, `_project_state_from_graph_state`.
5. `TurnGraph.execute` / `graph.py` — highest churn + D-grade = top refactor candidate.

### P2 — Style / Auto-fixable
6. Run `ruff format dadbot/` to reformat 274 files in one pass.
7. Run `ruff check dadbot/ --fix` to auto-fix 1,266 fixable issues (unused imports, trailing commas, etc.).
8. Address **222 BLE001** bare `except Exception` catches — add specific exception types or at minimum log them.

### P3 — Documentation
9. 1,540 D102 / 182 D101 — add docstrings incrementally; consider enabling a docstring coverage gate.

---

*All tests passing: 1568 passed, 0 failed (33 skipped, 2 xpassed)*
