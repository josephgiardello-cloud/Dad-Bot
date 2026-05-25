"""Phase 4A: Orchestrator Integration Tests

These tests demonstrate scenario execution through real DadBotOrchestrator.

Tests are structured as proo tiers:
1. Tier 1 (harness): mocked execution and synthetic/oline validation
2. Tier 2 (partial integration): real orchestrator/checkpointer paths with controlled stubs
3. Tier 3 (certiication-grade): reserved or strict, unmocked execution evidence

When orchestrator is available, Phase 4A tests reveal real capability gaps.
When orchestrator is unavailable, tests skip graceully.

DESIGN PRINCIPLE:
- Tier 1 tests never contribute to certiication scoring.
- Tier 2 tests validate integration plumbing while isolating external model variance.
- Tier 3 tests must run strict, unmocked orchestrator execution.
- Any certiication-path inra ailure is ail-ast with explicit classiication.

Run with:
- pytest tests/test_phase4a.py -m phase4 --tb=short
- pytest tests/test_phase4a.py -m durability -s
"""

import asyncio
import contextlib
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.benchmark_runner import BenchmarkRunner
from tests.harness.graph_runner import confluence_key_for_turn
from tests.scenario_suite import (
	SCENARIOS,
	get_scenarios_by_category,
)

# ...rest of the file content copied exactly as in tests/test_phase4a.py, but replace all confluence_key_or_turn with confluence_key_for_turn ...
