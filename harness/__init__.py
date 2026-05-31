"""Compatibility namespace for test harness modules.

Allows imports like `harness.graph_runner` to resolve to `tests/harness`.
"""

from pathlib import Path

_package_path = list(globals().get("__path__", []))
_tests_harness_dir = Path(__file__).resolve().parent.parent / "tests" / "harness"

if _tests_harness_dir.exists() and str(_tests_harness_dir) not in _package_path:
    _package_path.append(str(_tests_harness_dir))

__path__ = _package_path
