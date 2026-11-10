"""Identity consistency checks at service execution boundaries.

These checks do not authenticate a principal or establish resource ownership.
"""

import os
from dataclasses import dataclass

from .tenant_identity import require_tenant_id


def require_scoped_tenant_id(value: object) -> str:
    tenant_id = require_tenant_id(value)
    if tenant_id == "default":
        from dadbot.runtime.env_loader import is_production_environment

        enabled = os.environ.get("DADBOT_ALLOW_LEGACY_DEFAULT_TENANT", "").strip().lower()
        if is_production_environment() or enabled not in {"1", "true", "yes", "on"}:
            raise ValueError("The default tenant is allowed only in gated non-production environments")
    return tenant_id


def require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} identity is missing or invalid")
    return value


@dataclass(frozen=True, slots=True)
class ExecutionIdentity:
    tenant_id: str
    session_id: str
    request_id: str
    task_id: str

    def __post_init__(self) -> None:
        require_scoped_tenant_id(self.tenant_id)
        require_identifier(self.session_id, "session")
        require_identifier(self.request_id, "request")
        require_identifier(self.task_id, "task")

    def require_match(self, *, tenant_id: object, session_id: object, request_id: object) -> None:
        if (tenant_id, session_id, request_id) != (self.tenant_id, self.session_id, self.request_id):
            raise ValueError("execution identity mismatch")
