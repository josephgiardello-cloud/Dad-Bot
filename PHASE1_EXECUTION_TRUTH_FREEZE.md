# Phase 1 - Execution Truth Freeze

This document defines the single production execution path and hard boundaries for runtime behavior.

## Production execution kernel (hard rule)

- `dadbot/core/orchestrator.py` is the only production execution kernel.
- `DadBotOrchestrator.EXECUTION_ROLE` is `production_kernel`.
- All production turns must flow through `DadBotOrchestrator.handle_turn(...)`.

## Alternate execution roles

- `dadbot/runtime_adapter.py` is marked `EXECUTION_ROLE = "disabled"`.
- `dadbot/ux_overlay/runtime_entrypoint.py` is marked `EXECUTION_ROLE = "experimental"`.
- Experimental runtime entry is blocked unless `DADBOT_ENABLE_EXPERIMENTAL_RUNTIME=1` (or `true/yes/on`).
- Violations raise `RuntimeExecutionViolation` from `dadbot/core/execution_boundary.py`.

## Model gateway freeze

- `dadbot/managers/runtime_client.py` now enforces `ModelPort` caller identity.
- Any direct runtime model call without `caller="ModelPort"` raises `ModelGatewayViolation`.
- `dadbot/runtime/model/ollama_model_adapter.py` is the canonical bridge and tags calls with `caller="ModelPort"`.
- `dadbot/core/llm_mixin.py` compatibility wrappers route through `model_port.generate(...)` / `generate_async(...)`.

## Memory write freeze

- The only allowed memory mutation surface is `MemoryManager.mutate_memory_store(...)`.
- `dadbot/memory/storage.py` enforces `owner="MemoryManager"` at mutation entry.
- Any direct storage-level mutation bypass raises `MemoryWriteSurfaceViolation`.

## Determinism artifact

- A system-wide `ExecutionTraceContext` is generated at end of each orchestrated turn.
- Captured fields:
  - prompt
  - memory snapshot used
  - model call parameters
  - model output
  - memory retrieval set
  - tool outputs
  - normalized response
  - final hash
- Stored in:
  - `TurnContext.metadata["execution_trace_context"]`
  - `session_state["last_execution_trace_context"]`

## CI layering

- Layer 1: capability kernel trust gate (production blockers).
- Layer 2: runtime system validation.
- Layer 3: experimental systems (non-blocking).
