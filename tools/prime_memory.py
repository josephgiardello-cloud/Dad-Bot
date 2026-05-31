from __future__ import annotations

from dadbot.memory.manager import MemoryManager
from dadbot.services.vector_memory import initialize_global_memory


def _print(msg: str) -> None:
    print(msg)


def main() -> int:
    _print("[PrimeMemory] Starting dependency-free Gideon readiness check")

    mro_names = [cls.__name__ for cls in MemoryManager.__mro__]
    _print(f"[PrimeMemory] MemoryManager MRO: {mro_names}")
    assert "MemoryLifecycleMixin" in str(MemoryManager.__mro__)
    assert "MemorySearchMixin" in str(MemoryManager.__mro__)
    assert "MemoryIntegrationMixin" in str(MemoryManager.__mro__)
    _print("[PrimeMemory] MRO spine verified")

    try:
        from dadbot.tools.memory_search_tool import MEMORY_SEARCH_SPEC

        _print(
            "[PrimeMemory] MemorySearch tool spec loaded: "
            f"name={MEMORY_SEARCH_SPEC.name}, version={MEMORY_SEARCH_SPEC.version}"
        )
    except Exception as exc:
        _print(f"[PrimeMemory] MemorySearch tool import deferred: {exc}")

    try:
        initialize_global_memory()
    except ImportError as exc:
        _print(f"[PrimeMemory] Vector memory deferred: {exc}")
        _print("[PrimeMemory] Awakening mode: structural wiring ready, vector backend pending")
        return 0

    _print("[PrimeMemory] SovereignMemory initialized")
    _print("[PrimeMemory] Gideon awakening complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())