"""Determinism Edge Sealing.

Eliminates nondeterminism that would make two replays of the same turn
produce different checkpoint hashes even when the logical output is identical.

The four edges sealed here
--------------------------
1. Time leakage   — wall-clock timestamps that escape VirtualClock.
                    Detected by comparing ``time.time()`` anchors across state
                    snapshots; the ``TimeLeakDetector`` flags any float field
                    that changes between snapshots without an explicit mutation.
2. Float drift    — tiny floating-point rounding differences across runs.
                    ``FloatNormalizer`` rounds every float in a state dict to
                    ``precision`` decimal places before hashing.
3. Tool output    — extra whitespace, platform line-endings, or trailing bytes
                    that vary per-machine.  ``ToolOutputNormalizer`` strips and
                    collapses whitespace runs, normalises line endings.
4. Hidden random  — ``random.*`` calls that produce different values each run.
                    ``RandomnessSeal`` reports any ``random`` module import
                    found in the direct code path (static analysis helper).

Architecture role
-----------------
Pure utility layer — no imports from graph, kernel, or domain.  Called by:
  - ``TurnContext.checkpoint_snapshot()`` (normalises state before hashing)
  - ``GraphSideEffectsOrchestrator.emit_checkpoint()`` (optional strict mode)
  - Tests via ``DeterminismSeal.apply()``
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_time_like_float(value: float) -> bool:
    """Heuristic: epoch-range floats (> 1_000_000_000) are likely wall-clock."""
    return value > 1_000_000_000.0


def _round_float(value: float, precision: int) -> float:
    """Round *value* to *precision* decimal places, treating NaN/Inf as 0."""
    if not math.isfinite(value):
        return 0.0
    return round(value, precision)


# ---------------------------------------------------------------------------
# 1. Float normalizer
# ---------------------------------------------------------------------------

class FloatNormalizer:
    """Rounds all floats in a nested data structure to a fixed precision.

    This prevents tiny IEEE-754 rounding differences from invalidating
    checkpoint hashes across runs or platforms.

    Usage::

        norm = FloatNormalizer(precision=6)
        clean_state = norm.normalize(turn_context.state)
    """

    def __init__(self, *, precision: int = 6) -> None:
        self._precision = precision

    def normalize(self, obj: Any, *, _depth: int = 0) -> Any:
        """Return a copy of *obj* with all floats rounded."""
        if _depth > 64:
            return obj  # guard against pathological nesting
        if isinstance(obj, float):
            return _round_float(obj, self._precision)
        if isinstance(obj, dict):
            return {k: self.normalize(v, _depth=_depth + 1) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            normalised = [self.normalize(v, _depth=_depth + 1) for v in obj]
            return type(obj)(normalised) if isinstance(obj, tuple) else normalised
        return obj


# ---------------------------------------------------------------------------
# 2. Tool output normalizer
# ---------------------------------------------------------------------------

_WS_RUN = re.compile(r"[ \t]+")
_LINE_ENDING = re.compile(r"\r\n|\r")

class ToolOutputNormalizer:
    """Normalises text tool outputs to canonical form.

    Normalisation steps (in order):
    1. Replace ``\\r\\n`` and ``\\r`` with ``\\n``.
    2. Strip leading/trailing whitespace from each line.
    3. Collapse runs of spaces/tabs within a line to a single space.
    4. Remove trailing blank lines; strip the whole string.

    Usage::

        norm = ToolOutputNormalizer()
        clean_text = norm.normalize(raw_tool_output)
    """

    def normalize(self, text: Any) -> Any:
        """Return normalised copy.  Non-string values are returned unchanged."""
        if not isinstance(text, str):
            return text
        text = _LINE_ENDING.sub("\n", text)
        lines = [_WS_RUN.sub(" ", line.strip()) for line in text.split("\n")]
        # Remove trailing blank lines.
        while lines and not lines[-1]:
            lines.pop()
        return "\n".join(lines)

    def normalize_dict(self, obj: Any, *, _depth: int = 0) -> Any:
        """Recursively normalise all string values in a nested structure."""
        if _depth > 64:
            return obj
        if isinstance(obj, str):
            return self.normalize(obj)
        if isinstance(obj, dict):
            return {k: self.normalize_dict(v, _depth=_depth + 1) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            normalised = [self.normalize_dict(v, _depth=_depth + 1) for v in obj]
            return type(obj)(normalised) if isinstance(obj, tuple) else normalised
        return obj


# ---------------------------------------------------------------------------
# 3. Time leakage detection
# ---------------------------------------------------------------------------

class TimeLeakDetector:
    """Detects wall-clock timestamps that escaped VirtualClock.

    Usage::

        detector = TimeLeakDetector()
        detector.snapshot(state_before)
        ... stage executes ...
        leaks = detector.find_leaks(state_after)  # returns list of key paths
    """

    def __init__(self) -> None:
        self._before: dict[str, float] = {}

    def snapshot(self, state: Any) -> None:
        """Record all time-like floats in *state*."""
        self._before = {}
        self._collect(state, path="")

    def _collect(self, obj: Any, *, path: str, _depth: int = 0) -> None:
        if _depth > 32:
            return
        if isinstance(obj, float) and _is_time_like_float(obj):
            self._before[path] = obj
        elif isinstance(obj, dict):
            for k, v in obj.items():
                self._collect(v, path=f"{path}.{k}" if path else str(k), _depth=_depth + 1)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                self._collect(v, path=f"{path}[{i}]", _depth=_depth + 1)

    def find_leaks(self, state_after: Any) -> list[str]:
        """Return key paths where a new time-like float appeared or changed."""
        after: dict[str, float] = {}

        def _collect_after(obj: Any, *, path: str, depth: int = 0) -> None:
            if depth > 32:
                return
            if isinstance(obj, float) and _is_time_like_float(obj):
                after[path] = obj
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    _collect_after(v, path=f"{path}.{k}" if path else str(k), depth=depth + 1)
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    _collect_after(v, path=f"{path}[{i}]", depth=depth + 1)

        _collect_after(state_after, path="")
        leaks: list[str] = []
        for path, val in after.items():
            prev = self._before.get(path)
            if prev is None:
                leaks.append(path)  # new time-like float appeared
            elif abs(val - prev) > 0.5:
                leaks.append(path)  # changed significantly (not a stable timestamp)
        return leaks


# ---------------------------------------------------------------------------
# 4. Static randomness seal
# ---------------------------------------------------------------------------

class RandomnessSeal:
    """Static analysis: detect ``import random`` or ``random.*`` in source.

    Usage::

        seal = RandomnessSeal()
        violations = seal.audit_source(source_code, filename="node.py")
    """

    def audit_source(self, source: str, *, filename: str = "<unknown>") -> list[str]:
        """Return list of violation messages (empty → clean)."""
        violations: list[str] = []
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError:
            return []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "random":
                        violations.append(
                            f"{filename}:{node.lineno}: 'import random' — "
                            "use VirtualClock or a seeded generator passed through TurnContext"
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module == "random":
                    violations.append(
                        f"{filename}:{node.lineno}: 'from random import ...' — "
                        "use VirtualClock or a seeded generator passed through TurnContext"
                    )
            elif isinstance(node, ast.Attribute):
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "random"
                    and isinstance(node.attr, str)
                ):
                    violations.append(
                        f"{filename}:{getattr(node, 'lineno', '?')}: "
                        f"random.{node.attr}() — unsealed randomness"
                    )
        return violations


# ---------------------------------------------------------------------------
# 5. Unified DeterminismSeal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeterminismSealConfig:
    """Configuration for the unified seal.

    Attributes
    ----------
    float_precision:
        Decimal places to round floats to before hashing.  6 is sufficient for
        all latency/confidence fields while tolerating IEEE-754 rounding noise.
    normalize_tool_output:
        When True, string values in state are passed through ToolOutputNormalizer
        before hashing.  Recommended True for replay fidelity.
    detect_time_leaks:
        When True, after each stage the TimeLeakDetector will flag new
        wall-clock timestamps in state.  Violations are logged as warnings.
    """

    float_precision: int = 6
    normalize_tool_output: bool = True
    detect_time_leaks: bool = True


class DeterminismSeal:
    """Unified determinism normalisation pipeline.

    Apply to state or any nested data structure before computing a
    checkpoint hash.  The default config matches the settings used by
    ``TurnContext.checkpoint_snapshot()``.

    Usage::

        seal = DeterminismSeal()
        normalised_state = seal.apply(turn_context.state)
        # hash(normalised_state) is the same across all replays of this turn
    """

    def __init__(self, config: DeterminismSealConfig | None = None) -> None:
        self._config = config or DeterminismSealConfig()
        self._float_norm = FloatNormalizer(precision=self._config.float_precision)
        self._output_norm = ToolOutputNormalizer()
        self._time_leak = TimeLeakDetector() if self._config.detect_time_leaks else None

    @property
    def config(self) -> DeterminismSealConfig:
        return self._config

    def apply(self, obj: Any) -> Any:
        """Return a fully normalised copy of *obj*.

        Applies float rounding then (optionally) string normalisation.
        Time-leak detection is advisory: it mutates internal state but does NOT
        modify the returned object.
        """
        result = self._float_norm.normalize(obj)
        if self._config.normalize_tool_output:
            result = self._output_norm.normalize_dict(result)
        return result

    def snapshot_for_leak_detection(self, state: Any) -> None:
        """Record a before-snapshot for time-leak detection.

        Call once before a stage executes; then call ``find_time_leaks`` after.
        """
        if self._time_leak is not None:
            self._time_leak.snapshot(state)

    def find_time_leaks(self, state_after: Any) -> list[str]:
        """Return key paths where a new wall-clock float appeared after a stage.

        Returns an empty list when detect_time_leaks is False.
        """
        if self._time_leak is None:
            return []
        return self._time_leak.find_leaks(state_after)

    @staticmethod
    def audit_for_randomness(source: str, *, filename: str = "<unknown>") -> list[str]:
        """Delegate to RandomnessSeal.audit_source."""
        return RandomnessSeal().audit_source(source, filename=filename)


# ---------------------------------------------------------------------------
# Module-level default seal (for import convenience)
# ---------------------------------------------------------------------------

#: Default DeterminismSeal instance with default config.  Import and reuse
#: rather than constructing a new one per call.
DEFAULT_SEAL: DeterminismSeal = DeterminismSeal()
