"""Identity consistency checks at service execution boundaries.

These checks do not authenticate a principal or establish resource ownership.
"""

from dataclasses import dataclass

from .tenant_identity import require_tenant_id


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
        require_tenant_id(self.tenant_id)
        require_identifier(self.session_id, "session")
        require_identifier(self.request_id, "request")
        require_identifier(self.task_id, "task")

    def require_match(self, *, tenant_id: object, session_id: object, request_id: object) -> None:
        if (tenant_id, session_id, request_id) != (self.tenant_id, self.session_id, self.request_id):
            raise ValueError("execution identity mismatch")
