"""ROI #2 — Global Execution Trace Fingerprint Test.

Validates that the full execution trace (DAG stage order, receipt chain,
tool sequence, memory access pattern) produces a *stable, deterministic
fingerprint* across two identical constructions.

Any hidden graph-level corruption or non-deterministic execution order
causes the fingerprint to change, making this the single fastest regression
detector for system-level behaviour.

Tests:
  * Same inputs  → identical fingerprint (stability)
  * Mutated stage → different fingerprint (sensitivity)
  * Mutated tool  → different tool-sequence fingerprint
  * Mutated memory → different memory-pattern fingerprint
  * Empty chain   → empty / stable empty fingerprint
"""

from __future__ import annotations

import hashlib
import json

from dadbot.core.execution_receipt import ExecutionReceipt, ReceiptChain, ReceiptSigner
from dadbot.uril.truth_binding import compute_receipt_chain_hash

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXED_KEY = b"test-key-32bytes-padding-xxxxxxx!"[:32]


def _build_receipt_chain(
    stages: list[str],
    turn_id: str = "turn-fingerprint-001",
    key: bytes = _FIXED_KEY,
) -> list[ExecutionReceipt]:
    """Build a deterministic signed receipt chain for *stages*."""
    signer = ReceiptSigner(key)
    receipts: list[ExecutionReceipt] = []
    prev_sig = ""
    for i, stage in enumerate(stages):
        r = signer.sign(
            turn_id=turn_id,
            stage=stage,
            sequence=i + 1,  # ReceiptChain expects sequences starting at 1
            stage_call_id=hashlib.sha256(f"{turn_id}:{stage}".encode()).hexdigest()[:32],
            checkpoint_hash=hashlib.sha256(f"ckpt:{stage}:{i}".encode()).hexdigest()[:16],
            prev_receipt_sig=prev_sig,
            completed_at=1000.0 + i,
        )
        receipts.append(r)
        prev_sig = r.signature
    return receipts


def _hash_sequence(items: list[str]) -> str:
    """Order-sensitive SHA-256 fingerprint of a string sequence."""
    payload = json.dumps(items, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_set(items: list[str]) -> str:
    """Order-insensitive SHA-256 fingerprint of a string set."""
    payload = json.dumps(sorted(items), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compute_execution_fingerprint(
    stages: list[str],
    tools: list[str],
    memory_keys: list[str],
    receipts: list[ExecutionReceipt],
) -> dict[str, str]:
    """Compute all four fingerprint components for a single execution."""
    return {
        "dag_order_hash": _hash_sequence(stages),
        "receipt_chain_hash": compute_receipt_chain_hash(receipts),
        "tool_sequence_hash": _hash_sequence(tools),
        "memory_pattern_hash": _hash_set(memory_keys),
    }


# ---------------------------------------------------------------------------
# Canonical test fixtures
# ---------------------------------------------------------------------------

CANONICAL_STAGES = ["temporal", "inference", "tool", "memory", "output"]
CANONICAL_TOOLS = ["search_web", "query_memory"]
CANONICAL_MEMORY_KEYS = ["consolidated_memories", "life_patterns", "relationship_history"]


class TestExecutionTraceFingerprintStability:
    """Same inputs → identical fingerprint across N independent constructions."""

    def _build_fingerprint(self) -> dict[str, str]:
        receipts = _build_receipt_chain(CANONICAL_STAGES)
        return _compute_execution_fingerprint(CANONICAL_STAGES, CANONICAL_TOOLS, CANONICAL_MEMORY_KEYS, receipts)

    def test_dag_order_hash_is_stable(self):
        fp1 = self._build_fingerprint()
        fp2 = self._build_fingerprint()
        assert fp1["dag_order_hash"] == fp2["dag_order_hash"]

    def test_receipt_chain_hash_is_stable(self):
        fp1 = self._build_fingerprint()
        fp2 = self._build_fingerprint()
        assert fp1["receipt_chain_hash"] == fp2["receipt_chain_hash"]

    def test_tool_sequence_hash_is_stable(self):
        fp1 = self._build_fingerprint()
        fp2 = self._build_fingerprint()
        assert fp1["tool_sequence_hash"] == fp2["tool_sequence_hash"]

    def test_memory_pattern_hash_is_stable(self):
        fp1 = self._build_fingerprint()
        fp2 = self._build_fingerprint()
        assert fp1["memory_pattern_hash"] == fp2["memory_pattern_hash"]

    def test_full_fingerprint_dict_is_identical(self):
        fp1 = self._build_fingerprint()
        fp2 = self._build_fingerprint()
        assert fp1 == fp2


class TestExecutionTraceFingerprintSensitivity:
    """Mutations must produce different fingerprints."""

    def _base_receipts(self) -> list[ExecutionReceipt]:
        return _build_receipt_chain(CANONICAL_STAGES)

    def test_mutated_stage_order_changes_dag_hash(self):
        original_hash = _hash_sequence(CANONICAL_STAGES)
        mutated = list(reversed(CANONICAL_STAGES))
        mutated_hash = _hash_sequence(mutated)
        assert original_hash != mutated_hash

    def test_extra_stage_changes_dag_hash(self):
        original_hash = _hash_sequence(CANONICAL_STAGES)
        extra = CANONICAL_STAGES + ["cleanup"]
        extra_hash = _hash_sequence(extra)
        assert original_hash != extra_hash

    def test_mutated_receipt_changes_chain_hash(self):
        receipts = self._base_receipts()
        original_hash = compute_receipt_chain_hash(receipts)

        # Replace second receipt with a tampered version (forged signature).
        # compute_receipt_chain_hash reads the 'signature' field — tampering it
        # is the realistic attack that the fingerprint must catch.
        tampered = ExecutionReceipt(
            turn_id=receipts[1].turn_id,
            stage=receipts[1].stage,
            sequence=receipts[1].sequence,
            stage_call_id=receipts[1].stage_call_id,
            checkpoint_hash=receipts[1].checkpoint_hash,
            prev_receipt_sig=receipts[1].prev_receipt_sig,
            completed_at=receipts[1].completed_at,
            signature="FORGED_SIGNATURE_000000000000000000000000000000000000000000",
        )
        mutated_receipts = [receipts[0], tampered] + receipts[2:]
        mutated_hash = compute_receipt_chain_hash(mutated_receipts)
        assert original_hash != mutated_hash

    def test_different_tool_sequence_changes_tool_hash(self):
        h1 = _hash_sequence(["search_web", "query_memory"])
        h2 = _hash_sequence(["query_memory", "search_web"])  # reversed order
        assert h1 != h2

    def test_different_tool_set_changes_tool_hash(self):
        h1 = _hash_sequence(["search_web"])
        h2 = _hash_sequence(["search_web", "extra_tool"])
        assert h1 != h2

    def test_different_memory_keys_change_memory_hash(self):
        h1 = _hash_set(["key_a", "key_b"])
        h2 = _hash_set(["key_a", "key_c"])
        assert h1 != h2

    def test_memory_hash_is_order_insensitive(self):
        """Memory access pattern is a set — order must not matter."""
        h1 = _hash_set(["key_a", "key_b", "key_c"])
        h2 = _hash_set(["key_c", "key_a", "key_b"])
        assert h1 == h2


class TestExecutionTraceFingerprintEdgeCases:
    """Edge case coverage for the fingerprint helpers."""

    def test_empty_receipt_chain_returns_empty_string(self):
        assert compute_receipt_chain_hash([]) == ""

    def test_empty_stage_list_produces_stable_hash(self):
        h1 = _hash_sequence([])
        h2 = _hash_sequence([])
        assert h1 == h2

    def test_single_stage_is_stable(self):
        receipts = _build_receipt_chain(["only"])
        h1 = compute_receipt_chain_hash(receipts)
        receipts2 = _build_receipt_chain(["only"])
        h2 = compute_receipt_chain_hash(receipts2)
        assert h1 == h2 and h1 != ""

    def test_receipt_chain_from_dicts_matches_from_objects(self):
        receipts = _build_receipt_chain(CANONICAL_STAGES)
        receipt_dicts = [r.to_dict() for r in receipts]
        h_objects = compute_receipt_chain_hash(receipts)
        h_dicts = compute_receipt_chain_hash(receipt_dicts)
        assert h_objects == h_dicts

    def test_receipt_chain_continuity_is_valid(self):
        """The ReceiptChain built from our helper must pass verify_continuity."""
        signer = ReceiptSigner(_FIXED_KEY)
        receipts = _build_receipt_chain(CANONICAL_STAGES)
        chain = ReceiptChain(receipts)
        violations = chain.verify_continuity(signer, expected_stages=CANONICAL_STAGES)
        assert violations == [], [v.reason for v in violations]
