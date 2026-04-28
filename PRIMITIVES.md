# Execution Primitives

## Lock-Hash Execution Primitive

Primitive name: ExecutionCommitment

Definition:
- `ExecutionCommitment` is the canonical lock-hash envelope for turn execution.
- The lock payload includes user input, attachments, model identity, blackboard seed/fingerprint, memory fingerprint, tool-trace hash, and determinism-manifest hash.

Contract:
- Identical canonical payloads MUST produce identical `lock_hash` values.
- Any payload drift MUST produce a different `lock_hash`.
- Model adapters MUST receive a `determinism_context` carrying the lock hash and MUST normalize model output under that lock context before returning text.

Implementation anchors:
- `dadbot/core/execution_commitment.py`
- `dadbot/core/orchestrator.py` (`DadBotOrchestrator._build_turn_context`)
- `dadbot/runtime/model/model_call_port.py`
- `dadbot/runtime/model/ollama_model_adapter.py`

## SaveNode Transactional Boundary Primitive

Primitive name: SaveNode Commit Boundary (`NodeType.COMMIT`)

Definition:
- SaveNode is the only durable-commit boundary in the turn graph.
- Graph mutation guards enforce speculative execution for all non-commit nodes.

Contract:
- All durable mutations MUST go through SaveNode. The graph guarantees speculative execution until that boundary.

Implementation anchors:
- `dadbot/core/graph.py` (`NodeType`, `TurnGraph._is_commit_boundary_node`, `SAVE_NODE_COMMIT_CONTRACT`)
- `dadbot/core/nodes.py` (`SaveNode.node_type = NodeType.COMMIT`)

## Publishing Checklist Notes

Already addressed in this branch:
- Memory schema versioning and migration registry are implemented.
- Launcher entry points route through the canonical app runtime path.
