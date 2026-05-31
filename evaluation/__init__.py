"""Compatibility package for legacy evaluation modules.

The canonical evaluation sources live under archive/legacy-audit-eval/evaluation.
This package extends its module search path so imports like
`evaluation.coherence_engine` continue to work in tests and tooling.
"""

from pathlib import Path

_legacy_eval_dir = (
    Path(__file__).resolve().parent.parent
    / "archive"
    / "legacy-audit-eval"
    / "evaluation"
)

_package_path = list(globals().get("__path__", []))
if _legacy_eval_dir.exists() and str(_legacy_eval_dir) not in _package_path:
    _package_path.append(str(_legacy_eval_dir))

# Defensive fallback for explicit imports like `evaluation.__init__`.
__path__ = _package_path
