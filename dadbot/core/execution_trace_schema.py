from __future__ import annotations

"""Compatibility re-export for legacy execution trace schema imports.

Canonical implementation lives in dadbot.core.execution_schema.
"""

from dadbot.core.execution_schema import EXECUTION_TRACE_CONTRACT_SCHEMA_VERSION
from dadbot.core.execution_schema import ExecutionTraceContractMigration
from dadbot.core.execution_schema import ExecutionTraceContractSchemaMigrator
from dadbot.core.execution_schema import get_trace_migrator
from dadbot.core.execution_schema import migrate_trace_contract
from dadbot.core.execution_schema import stamp_trace_contract_version

__all__ = [
    "EXECUTION_TRACE_CONTRACT_SCHEMA_VERSION",
    "ExecutionTraceContractMigration",
    "ExecutionTraceContractSchemaMigrator",
    "get_trace_migrator",
    "migrate_trace_contract",
    "stamp_trace_contract_version",
]
