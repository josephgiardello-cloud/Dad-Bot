"""Backward-compatible import shim for SQLite checkpointer."""

from dadbot.core.persistence.checkpointer import (
    AsyncSQLiteCheckpointer,
    SQLiteCheckpointer,
)

__all__ = ["AsyncSQLiteCheckpointer", "SQLiteCheckpointer"]
