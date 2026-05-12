"""
Centralized configuration resolver for strict persistence schema mode.

Single source of truth for determining whether strict schema validation
and rejection is enabled across manager, service, and any other callers.
"""

import os
from typing import Optional


def is_strict_persistence_schema_mode(
    service_strict_mode: Optional[bool] = None,
) -> bool:
    """
    Determine if strict persistence schema mode is enabled.

    Checks sources in priority order:
    1. Service strict_mode flag (if provided)
    2. Environment variable DADBOT_PERSISTENCE_SCHEMA_STRICT_REJECT
    3. Default: False (permissive mode, warns instead of rejects)

    Args:
        service_strict_mode: Optional explicit flag from service instance.
                            Takes precedence if provided.

    Returns:
        True if strict mode is enabled; False otherwise.
    """
    # Priority 1: Service explicit flag
    if service_strict_mode is not None:
        return service_strict_mode

    # Priority 2: Environment variable
    env_strict = os.environ.get("DADBOT_PERSISTENCE_SCHEMA_STRICT_REJECT", "").strip().lower()
    if env_strict in ("1", "true", "yes", "on"):
        return True

    # Default: Permissive mode
    return False


__all__ = ["is_strict_persistence_schema_mode"]
