from __future__ import annotations

import os
from pathlib import Path

from dadbot.utils import env_truthy


def _parse_env_line(line: str) -> tuple[str, str] | None:
    raw = str(line or "").strip()
    if not raw or raw.startswith("#"):
        return None
    if raw.lower().startswith("export "):
        raw = raw[7:].strip()
    if "=" not in raw:
        return None

    key, value = raw.split("=", 1)
    key = str(key or "").strip()
    value = str(value or "").strip()
    if not key:
        return None

    # Strip one layer of surrounding quotes when present.
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        value = value[1:-1]
    return key, value


def load_env_file(path: Path, *, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs from an env file into process environment.

    - Ignores comments and blank lines.
    - Supports optional leading ``export``.
    - By default, preserves already-defined environment variables.
    """
    loaded: dict[str, str] = {}
    if not path.exists() or not path.is_file():
        return loaded

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parsed = _parse_env_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def bootstrap_environment() -> dict[str, str]:
    """Load environment defaults from local env files when appropriate.

    Priority order:
    1) ``DADBOT_ENV_FILE`` (if set)
    2) ``.env`` at repository root
    3) ``.env.production`` at repository root

    Existing process environment always wins over file values.
    Test runs do not auto-load env files unless explicitly forced.
    """
    if ("PYTEST_CURRENT_TEST" in os.environ) and not env_truthy("DADBOT_LOAD_ENV_IN_TESTS", default=False):
        return {}

    repo_root = Path(__file__).resolve().parents[2]
    requested = str(os.environ.get("DADBOT_ENV_FILE") or "").strip()
    candidates: list[Path]
    if requested:
        configured = Path(requested)
        if not configured.is_absolute():
            configured = repo_root / configured
        candidates = [configured]
    else:
        candidates = [repo_root / ".env", repo_root / ".env.production"]

    merged: dict[str, str] = {}
    for candidate in candidates:
        merged.update(load_env_file(candidate, override=False))
    return merged


def validate_startup_environment(*, serve_api: bool) -> list[str]:
    """Return startup validation errors for security-critical configuration."""
    errors: list[str] = []

    auth_required = env_truthy("DADBOT_API_AUTH_REQUIRED", default=True)
    token_secret = str(os.environ.get("DADBOT_API_TOKEN_SECRET") or "").strip()
    if serve_api and auth_required and not token_secret:
        errors.append("DADBOT_API_TOKEN_SECRET is required when API auth is enabled.")

    enforce_egress = env_truthy("DADBOT_EGRESS_ENFORCE", default=False)
    allowlist = str(os.environ.get("DADBOT_EGRESS_ALLOWLIST") or "").strip()
    if enforce_egress and not allowlist:
        errors.append("DADBOT_EGRESS_ALLOWLIST is required when DADBOT_EGRESS_ENFORCE is enabled.")

    return errors
