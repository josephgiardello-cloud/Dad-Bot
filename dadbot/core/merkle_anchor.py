from __future__ import annotations

import hashlib
import json
from typing import Any


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def stable_hash(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    return _sha256_bytes(serialized)


def make_leaf(payload: Any) -> str:
    return stable_hash({"leaf": stable_hash(payload)})


def _pair_hash(left: str, right: str) -> str:
    return _sha256_bytes(f"{left}:{right}".encode("utf-8"))


def merkle_root(leaves: list[str]) -> str:
    if not leaves:
        return _sha256_bytes(b"")
    level = list(leaves)
    while len(level) > 1:
        next_level: list[str] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            next_level.append(_pair_hash(left, right))
        level = next_level
    return level[0]


def build_inclusion_proof(leaves: list[str], index: int) -> list[dict[str, str]]:
    if not leaves:
        return []
    if index < 0 or index >= len(leaves):
        raise IndexError(f"Leaf index out of range: {index}")

    proof: list[dict[str, str]] = []
    level = list(leaves)
    current_index = int(index)

    while len(level) > 1:
        sibling_index = current_index - 1 if current_index % 2 == 1 else current_index + 1
        if sibling_index >= len(level):
            sibling_hash = level[current_index]
        else:
            sibling_hash = level[sibling_index]
        proof.append({
            "position": "left" if sibling_index < current_index else "right",
            "hash": sibling_hash,
        })

        next_level: list[str] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            next_level.append(_pair_hash(left, right))

        level = next_level
        current_index //= 2

    return proof


def verify_inclusion_proof(leaf_hash: str, proof: list[dict[str, str]], expected_root: str) -> bool:
    current = str(leaf_hash or "")
    for item in proof:
        sibling = str(item.get("hash") or "")
        position = str(item.get("position") or "right").strip().lower()
        if position == "left":
            current = _pair_hash(sibling, current)
        else:
            current = _pair_hash(current, sibling)
    return current == str(expected_root or "")


def append_leaf_and_anchor(
    leaf_store: list[str],
    payload: Any,
) -> dict[str, Any]:
    leaf = make_leaf(payload)
    leaf_store.append(leaf)
    index = len(leaf_store) - 1
    root = merkle_root(leaf_store)
    proof = build_inclusion_proof(leaf_store, index)
    return {
        "leaf_hash": leaf,
        "leaf_index": index,
        "leaf_count": len(leaf_store),
        "merkle_root": root,
        "inclusion_proof": proof,
    }


__all__ = [
    "append_leaf_and_anchor",
    "build_inclusion_proof",
    "make_leaf",
    "merkle_root",
    "stable_hash",
    "verify_inclusion_proof",
]
