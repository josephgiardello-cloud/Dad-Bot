# Phase 4.1.B: The Enforcement Hardening

## Sprint Goal

Pivot to **Phase 4.1.B: The Enforcement Hardening** — execute the Phase 4 runtime foundation by hardening contract enforcement, permission gates, and configuration portability. This is a **release-blocking** phase focused on security, determinism, and robustness.

## Critical Objectives

### 1. Solidify the Contract Evaluator
- **What**: Replace schema-validation placeholders in `dadbot/core/contract_evaluator.py` with functional ledger-state transition checks.
- **Why**: Contracts are the guardrails; placeholders mean violations silently pass through.
- **How**:
  - Implement `SovereignLedgerState` and `SovereignLedgerTransition` Pydantic models.
  - Implement `SchemaContractValidator` with state-transition rules and causal ordering invariants.
  - Replace `_validate_runtime_contract()` and `_validate_persistence_contract()` with real schema checks.
  - Raise `ContractViolationError` on any violation; treat as critical failure path (no silent fallback).
- **Owner**: Enforcement layer (strict, no negotiation).

### 2. Enforce Tool Permissions
- **What**: Implement pre-execution permission checks in `dadbot/core/tool_registry.py` before tool invocation.
- **Why**: Tool registry is the execution boundary; permissions must be enforced _before_ the tool runs, not after.
- **How**:
  - Extract caller identity from `ToolInvocation.caller` (role-based defaults) and `invocation.arguments` (dynamic context).
  - Match required permissions from `ToolSpec.required_permissions` against caller permissions.
  - Return `ToolExecutionStatus.DENIED` (not `ERROR`) if permissions are insufficient.
  - Error message must include caller identity and specific missing permission names.
- **Owner**: Authorization layer (hard gate at execution entry).

### 3. Path Normalization (Portability)
- **What**: Remove hardcoded machine-specific paths from configuration and test artifacts.
- **Why**: Dev configs must be portable across machines and CI environments without manual edits.
- **How**:
  - Replace machine-specific Python paths in `.vscode/tasks.json` with `${command:python.interpreterPath}`.
  - Normalize command strings in `tests/phase4_baselines.json` to use `${WORKSPACE_ROOT}` tokens.
  - Move hardcoded dev secrets (e.g., `POSTGRES_PASSWORD: dadbot`) to `.env.template` with `${POSTGRES_PASSWORD}` references.
- **Owner**: Config layer (enable cross-machine development).

### 4. Secret Hygiene
- **What**: Remove inline default passwords from tracked configs; track `.env.template` for developer reference.
- **Why**: Secrets in git (even defaults) are a compliance violation; templates teach developers the pattern.
- **How**:
  - Move `POSTGRES_PASSWORD` from `docker-compose.yml` to `.env.template` + env-var reference.
  - Move `POSTGRES_PASSWORD` from `.github/workflows/semantic-pgvector.yml` to use secrets or CI ephemeral defaults.
  - Create `.env.template` with placeholder credentials and developer instructions.
  - Add `!.env.template` to `.gitignore` to track the template (not the actual secrets).
- **Owner**: Security/DevOps layer (compliance).

## Guardrails (Non-Negotiable)

1. **NO new features.** This phase is enforcement-only; no HUD visual work, no agent enhancements.
2. **NO modifications to `agent_driver_loop.py` logic.** The loop orchestration is stable; only hook into validation boundaries (`_enforce_global_turn_invariant_gate`, contract evaluation).
3. **NO silent fallbacks.** If a contract, permission, or portability check fails, raise and log; never degrade silently.
4. **Contracts are first-class.** All violations flow through `ContractViolationError` and are treated as critical system failures.

## Success Criteria

- [ ] All schema-validation placeholders replaced with functional contract checks.
- [ ] Tool permission enforcement returns `DENIED` status (not `ERROR`) when permissions are insufficient.
- [ ] All tests pass in DEV lane (unit tests), INTEGRATION lane, and CERT lane.
- [ ] No hardcoded paths in `.vscode/tasks.json` or `tests/phase4_baselines.json`.
- [ ] `.env.template` created, tracked in git; `.env` ignored as before.
- [ ] Inline secrets removed from `docker-compose.yml` and `.github/workflows/semantic-pgvector.yml`.
- [ ] No new agent-loop or HUD feature work is present in this phase.
- [ ] Zero suppressions in persistence: all `_persistence_mixins.py` `# type: ignore[attr-defined]` markers replaced by typed Protocol contracts.
- [ ] Schema-active contracts: a chaos payload injected into ledger mutation metadata triggers `ContractViolationError`.
- [ ] Wired boot boundary: `_model_port is not None` is verified before boot boundary compliance is declared.

## Timeline

- **Phase 4.1.B execution**: Enforcement hardening complete and green on all lanes.
- **Phase 4.2 (next)**: Real-time observability (tracing, metrics, dashboards).
- **Phase 5 onwards**: Feature work and HUD expansion (blocked until Phase 4.1.B passes).

---

**This phase gates all downstream work. Do not bypass or soft-check violations.**
