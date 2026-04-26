"""Checkpoint persistence adapters for Dad Bot.

Provides abstract checkpointer interface and SQLite implementation for durable state
management with hash-chain integrity, manifest drift detection, and pruning.
"""

from dadbot.core.persistence.base import (
    AbstractCheckpointer,
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointNotFoundError,
)
from dadbot.core.persistence.sqlite_checkpointer import SQLiteCheckpointer

__all__ = [
    "AbstractCheckpointer",
    "CheckpointError",
    "CheckpointIntegrityError",
    "CheckpointNotFoundError",
    "SQLiteCheckpointer",
]
