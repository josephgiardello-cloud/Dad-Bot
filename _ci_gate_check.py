"""Compatibility wrapper for CI gates.

Supported flags (handled by tools.ci_gate):
    --contract-gate
    --fail-on-untested
    --adversarial-closure-gate
"""

from tools.ci_gate import main


if __name__ == "__main__":
    raise SystemExit(main())
