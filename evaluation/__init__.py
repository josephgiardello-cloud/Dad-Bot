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

if _legacy_eval_dir.exists():
    __path__.append(str(_legacy_eval_dir))
