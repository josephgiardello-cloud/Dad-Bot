"""Checkpoint persistence adapters for Dad Bot.

Provides abstract checkpointer interface and SQLite implementation for durable state
management with hash-chain integrity, manifest drift detection, and pruning.
"""

from dadbot.core.persistence.base import (
    AbstractAsyncCheckpointer,
    AbstractCheckpointer,
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointNotFoundError,
)
from dadbot.core.persistence.checkpointer import (
    AsyncSQLiteCheckpointer,
    SQLiteCheckpointer,
)

__all__ = [
    "AbstractAsyncCheckpointer",
    "AbstractCheckpointer",
    "AsyncSQLiteCheckpointer",
    "CheckpointError",
    "CheckpointIntegrityError",
    "CheckpointNotFoundError",
    "SQLiteCheckpointer",
]
