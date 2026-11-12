"""Compatibility launcher for the canonical Phase 4A harness."""

from pathlib import Path

import pytest

if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).with_name("test_phase4a.py"))]))
