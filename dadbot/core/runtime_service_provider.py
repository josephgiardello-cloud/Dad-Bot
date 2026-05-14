from __future__ import annotations

import logging
import os
from typing import Any, Protocol

from dadbot.core.policy_store import DadPolicyStore, InMemoryAsyncPolicyPersistence, SQLiteAsyncPolicyPersistence
from dadbot.core.world_model import InMemoryAsyncWorldModelPersistence, SQLiteAsyncWorldModelPersistence, WorldModelStore
from dadbot.memory.ledger import InMemoryAsyncMemoryLedgerPersistence, MemoryLedger, SQLiteAsyncMemoryLedgerPersistence


logger = logging.getLogger(__name__)


class CoreRuntimeServices(Protocol):
    def get_policy_store(self) -> DadPolicyStore: ...

    def get_memory_ledger(self) -> MemoryLedger: ...

    def get_world_model_store(self) -> WorldModelStore: ...


class DefaultCoreRuntimeServices:
    """Default dependency provider for core runtime persistence services.

    This class centralizes construction/wiring of turn-time stores so the core
    turn path can depend on an injectable provider rather than directly owning
    environment-coupled setup logic.
    """

    def __init__(self, bot: Any) -> None:
        self._bot = bot

    def get_policy_store(self) -> DadPolicyStore:
        policy_store = getattr(self._bot, "_turn_policy_store", None)
        if isinstance(policy_store, DadPolicyStore):
            return policy_store
        policy_db_path = str(os.environ.get("DADBOT_POLICY_DB_PATH", "runtime/dadbot_state.sqlite3") or "").strip()
        try:
            persistence = SQLiteAsyncPolicyPersistence(policy_db_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Policy sqlite persistence unavailable, falling back to in-memory: %s", exc)
            persistence = InMemoryAsyncPolicyPersistence()
        policy_store = DadPolicyStore(persistence)
        self._bot._turn_policy_store = policy_store
        return policy_store

    def get_memory_ledger(self) -> MemoryLedger:
        ledger = getattr(self._bot, "_turn_memory_ledger", None)
        if isinstance(ledger, MemoryLedger):
            return ledger

        persistence = getattr(self._bot, "_turn_memory_ledger_persistence", None)
        if persistence is None:
            ledger_db_path = str(
                os.environ.get("DADBOT_MEMORY_LEDGER_DB_PATH", "runtime/dadbot_state.sqlite3") or "",
            ).strip()
            try:
                persistence = SQLiteAsyncMemoryLedgerPersistence(ledger_db_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Memory ledger sqlite persistence unavailable, falling back to in-memory: %s", exc)
                persistence = InMemoryAsyncMemoryLedgerPersistence()
            self._bot._turn_memory_ledger_persistence = persistence

        ledger = MemoryLedger(persistence)
        self._bot._turn_memory_ledger = ledger
        return ledger

    def get_world_model_store(self) -> WorldModelStore:
        store = getattr(self._bot, "_turn_world_model_store", None)
        if isinstance(store, WorldModelStore):
            return store

        db_path = str(os.environ.get("DADBOT_WORLD_MODEL_DB_PATH", "runtime/dadbot_state.sqlite3") or "").strip()
        try:
            persistence = SQLiteAsyncWorldModelPersistence(db_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("World model sqlite persistence unavailable, falling back to in-memory: %s", exc)
            persistence = InMemoryAsyncWorldModelPersistence()
        store = WorldModelStore(persistence)
        self._bot._turn_world_model_store = store
        return store
