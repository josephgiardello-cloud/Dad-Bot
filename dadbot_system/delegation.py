from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import time
from contextlib import closing, contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from .execution_identity import require_identifier
from .security import AuthorizationError, ServiceTokenManager
from .tenant_identity import require_tenant_id

if TYPE_CHECKING:
    from .contracts import WorkerTask
    from .security import ServicePrincipal

_execution_authority: ContextVar[tuple[SQLiteDelegationStore, WorkerTask] | None] = ContextVar(
    "delegated_execution_authority", default=None
)
_runtime_authority: ContextVar[tuple[SQLiteDelegationStore, ServicePrincipal, str] | None] = ContextVar(
    "authenticated_runtime_authority", default=None
)
_live_effect_authority_required: ContextVar[bool] = ContextVar(
    "live_effect_authority_required", default=False
)
_mutation_fence: ContextVar[tuple[str, str] | None] = ContextVar("tenant_mutation_fence", default=None)


def verify_live_authority() -> None:
    authority = _execution_authority.get()
    if authority is not None:
        store, task = authority
        store.verify(task.delegation, task, operation="execute")
        return
    runtime_authority = _runtime_authority.get()
    if runtime_authority is not None:
        store, principal, scope = runtime_authority
        if principal.expires_at <= time.time():
            raise AuthorizationError("Runtime authority principal has expired")
        principal.require_scope(scope)
        store.require_membership(
            issuer=principal.issuer,
            subject=principal.subject,
            tenant_id=principal.tenant_id,
            scope=scope,
        )
        return
    if _live_effect_authority_required.get():
        raise AuthorizationError("Verified live execution authority is required")


def live_authority_audit_context() -> dict[str, str]:
    """Return a privacy-preserving identity for an already-verified execution."""
    authority = _execution_authority.get()
    if authority is not None:
        _store, task = authority
        return {
            "authority_mode": "delegated",
            "tenant_id": str(task.tenant_id or ""),
            "session_id": str(task.session_id or ""),
            "task_id": str(task.task_id or ""),
        }

    runtime_authority = _runtime_authority.get()
    if runtime_authority is not None:
        _store, principal, scope = runtime_authority
        principal_ref = hashlib.sha256(
            f"{principal.issuer}:{principal.subject}".encode("utf-8"),
        ).hexdigest()[:20]
        return {
            "authority_mode": "authenticated",
            "tenant_id": str(principal.tenant_id or ""),
            "principal_ref": principal_ref,
            "scope": str(scope or ""),
        }

    if _live_effect_authority_required.get():
        return {"authority_mode": "required_missing"}
    return {"authority_mode": "trusted_local"}


@contextmanager
def require_live_effect_authority(*, required: bool = True):
    token = _live_effect_authority_required.set(bool(required))
    try:
        yield
    finally:
        _live_effect_authority_required.reset(token)


@contextmanager
def runtime_authority_scope(
    store: SQLiteDelegationStore,
    principal: ServicePrincipal,
    *,
    scope: str,
    tenant_id: str,
):
    tenant = require_tenant_id(tenant_id)
    if principal.tenant_id != tenant:
        raise AuthorizationError("Runtime authority tenant mismatch")
    if principal.expires_at <= time.time():
        raise AuthorizationError("Runtime authority principal has expired")
    principal.require_scope(scope)
    with store.publication_fence():
        store.require_membership(
            issuer=principal.issuer,
            subject=principal.subject,
            tenant_id=tenant,
            scope=scope,
        )
        authority_token = _runtime_authority.set((store, principal, scope))
        required_token = _live_effect_authority_required.set(True)
        try:
            yield
        finally:
            _live_effect_authority_required.reset(required_token)
            _runtime_authority.reset(authority_token)


def require_live_authority() -> None:
    if _execution_authority.get() is None and _runtime_authority.get() is None:
        raise AuthorizationError("Delegated execution authority is missing")
    verify_live_authority()


def require_tenant_authority(tenant_id: str) -> None:
    authority = _execution_authority.get()
    tenant = require_tenant_id(tenant_id)
    if authority is not None:
        store, task = authority
        store.verify(task.delegation, task, operation="execute")
        if task.tenant_id != tenant:
            raise AuthorizationError("Delegated execution tenant mismatch")
        return
    runtime_authority = _runtime_authority.get()
    if runtime_authority is None:
        raise AuthorizationError("Delegated execution authority is missing")
    store, principal, scope = runtime_authority
    if principal.tenant_id != tenant:
        raise AuthorizationError("Runtime authority tenant mismatch")
    verify_live_authority()


def require_tenant_authority_if_active(tenant_id: str, *, required: bool = False) -> None:
    if required or _execution_authority.get() is not None or _runtime_authority.get() is not None:
        require_tenant_authority(tenant_id)


@contextmanager
def tenant_mutation_fence(tenant_id: str, *, required: bool):
    authority = _execution_authority.get()
    if authority is None:
        if _runtime_authority.get() is not None:
            require_tenant_authority(tenant_id)
            yield
            return
        if required:
            raise AuthorizationError("Delegated execution authority is missing")
        yield
        return
    store, task = authority
    binding = (store.path, task.delegation)
    if _mutation_fence.get() == binding:
        require_tenant_authority(tenant_id)
        yield
        return
    with store.publication_fence():
        require_tenant_authority(tenant_id)
        token = _mutation_fence.set(binding)
        try:
            yield
        finally:
            _mutation_fence.reset(token)


class AuthorityUnavailableError(RuntimeError):
    pass


class SQLiteDelegationStore:
    """Local-installation authority store; the database must be explicitly initialized."""

    def __init__(self, path: str, secret: str, *, issuer: str = "dadbot") -> None:
        if not path or path == ":memory:" or not secret:
            raise ValueError("A persistent authority path and signing secret are required")
        self.path = str(Path(path).resolve())
        self._uri = Path(self.path).as_uri() + "?mode=rw"
        self._key = hmac.new(secret.encode(), b"dadbot-delegation-v1", hashlib.sha256).digest()
        self._tokens = ServiceTokenManager(secret, issuer=issuer)

    @staticmethod
    def initialize(path: str) -> None:
        if not path or path == ":memory:":
            raise ValueError("A persistent authority path is required")
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS memberships (issuer TEXT NOT NULL, subject TEXT NOT NULL, "
                "tenant TEXT NOT NULL, scopes TEXT NOT NULL, version INTEGER NOT NULL, active INTEGER NOT NULL, "
                "PRIMARY KEY (issuer, subject, tenant))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS delegations (id TEXT PRIMARY KEY, payload TEXT NOT NULL, "
                "signature TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS session_revocation_clock "
                "(singleton INTEGER PRIMARY KEY CHECK (singleton = 1), revision INTEGER NOT NULL)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO session_revocation_clock VALUES (1, 0)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS session_revocations "
                "(tenant_id TEXT NOT NULL, session_id TEXT NOT NULL, revision INTEGER NOT NULL, "
                "PRIMARY KEY (tenant_id, session_id))"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS session_revocations_revision_idx "
                "ON session_revocations (revision)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS authority_identity "
                "(singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
                "store_id TEXT NOT NULL, version INTEGER NOT NULL)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO authority_identity VALUES (1, ?, 1)",
                (uuid4().hex,),
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS tenant_identities "
                "(tenant TEXT PRIMARY KEY, issued_by_issuer TEXT NOT NULL, "
                "issued_by_subject TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS tenant_identity_aliases "
                "(issuer TEXT NOT NULL, alias TEXT NOT NULL, tenant TEXT NOT NULL, "
                "PRIMARY KEY (issuer, alias))"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = None
        try:
            connection = sqlite3.connect(self._uri, uri=True, timeout=5)
            connection.execute("PRAGMA synchronous=FULL")
            return connection
        except sqlite3.Error as exc:
            if connection is not None:
                connection.close()
            raise AuthorityUnavailableError("Delegation authority store unavailable") from exc

    def _signature(self, payload: str) -> str:
        return hmac.new(self._key, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def storage_identity(self) -> str:
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT store_id, version FROM authority_identity WHERE singleton = 1"
                ).fetchone()
        except sqlite3.Error as exc:
            raise AuthorityUnavailableError("Authority store identity is unavailable") from exc
        if row is None or not isinstance(row[0], str) or len(row[0]) != 32 or row[1] != 1:
            raise AuthorityUnavailableError("Authority store identity is invalid or has an unsupported version")
        return row[0]

    def check_available(self) -> None:
        try:
            with closing(self._connect()) as connection:
                connection.execute("SELECT id, payload, signature, revoked FROM delegations LIMIT 0")
                connection.execute("SELECT issuer, subject, tenant, scopes, version, active FROM memberships LIMIT 0")
                revision = connection.execute(
                    "SELECT revision FROM session_revocation_clock WHERE singleton = 1"
                ).fetchone()
                connection.execute("SELECT tenant_id, session_id, revision FROM session_revocations LIMIT 0")
        except sqlite3.Error as exc:
            raise AuthorityUnavailableError("Delegation authority store unavailable or uninitialized") from exc
        if revision is None or not isinstance(revision[0], int) or revision[0] < 0:
            raise AuthorityUnavailableError("Delegation authority revocation schema is invalid")

    def session_revocation_cursor(self) -> int:
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT revision FROM session_revocation_clock WHERE singleton = 1"
                ).fetchone()
        except sqlite3.Error as exc:
            raise AuthorityUnavailableError("Session revocation cursor is unavailable") from exc
        if row is None or not isinstance(row[0], int) or row[0] < 0:
            raise AuthorityUnavailableError("Session revocation cursor is invalid")
        return row[0]

    def session_revocations_after(self, revision: int, *, limit: int = 256) -> list[tuple[int, str, str]]:
        if not isinstance(revision, int) or revision < 0 or not isinstance(limit, int) or limit < 1:
            raise ValueError("A nonnegative revocation cursor and positive limit are required")
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    "SELECT revision, tenant_id, session_id FROM session_revocations "
                    "WHERE revision>? ORDER BY revision LIMIT ?",
                    (revision, limit),
                ).fetchall()
        except sqlite3.Error as exc:
            raise AuthorityUnavailableError("Session revocations are unavailable") from exc
        result = []
        try:
            for row_revision, tenant_id, session_id in rows:
                if not isinstance(row_revision, int) or row_revision <= revision:
                    raise ValueError("Invalid session revocation revision")
                result.append(
                    (
                        row_revision,
                        require_tenant_id(tenant_id),
                        require_identifier(session_id, "session"),
                    )
                )
        except (TypeError, ValueError) as exc:
            raise AuthorityUnavailableError("Session revocation record is invalid") from exc
        return result

    @contextmanager
    def publication_fence(self):
        """Order publication against membership and delegation changes in this authority database."""
        with closing(self._connect()) as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
            except sqlite3.Error as exc:
                raise AuthorityUnavailableError("Delegation authority publication fence unavailable") from exc
            try:
                yield
            finally:
                connection.rollback()

    @staticmethod
    def _task_digest(task) -> str:
        from .contracts import ChatRequest

        encoded = json.dumps(
            {
                "request": ChatRequest.from_dict(task.request.to_dict()).to_dict(),
                "session_state": task.session_state,
            },
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def sign_result(self, task, result) -> str:
        self.verify(task.delegation, task, operation="execute")
        payload = result.to_dict()
        payload.pop("authority_proof", None)
        encoded = json.dumps(
            {"delegation": task.delegation, "result": payload},
            sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        )
        return hmac.new(self._key, b"result-v1:" + encoded.encode(), hashlib.sha256).hexdigest()

    def verify_result(self, task, result) -> None:
        proof = self.sign_result(task, result)
        if not isinstance(result.authority_proof, str) or not hmac.compare_digest(proof, result.authority_proof):
            raise AuthorizationError("Invalid worker result authority proof")

    def grant_membership(self, *, issuer: str, subject: str, tenant_id: str, scopes: set[str]) -> None:
        issuer = require_identifier(issuer, "issuer")
        subject = require_identifier(subject, "principal")
        tenant_id = require_tenant_id(
            tenant_id,
            issuer=issuer,
            subject=subject,
            acceptance_path="SQLiteDelegationStore.grant_membership",
        )
        if not scopes or any(not isinstance(scope, str) or not scope or scope != scope.lower() for scope in scopes):
            raise ValueError("Explicit normalized membership scopes are required")
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    "INSERT INTO memberships VALUES (?, ?, ?, ?, 1, 1) "
                    "ON CONFLICT(issuer, subject, tenant) DO UPDATE SET "
                    "scopes=excluded.scopes, version=memberships.version+1, active=1",
                    (issuer, subject, tenant_id, json.dumps(sorted(scopes))),
                )
        except sqlite3.Error as exc:
            raise AuthorityUnavailableError("Delegation authority store unavailable") from exc

    def provision_tenant_identity(
        self,
        *,
        issuer: str,
        subject: str,
        tenant_id: str,
        scopes: set[str],
        aliases: tuple[str, ...] | list[str],
    ) -> None:
        issuer = require_identifier(issuer, "issuer")
        subject = require_identifier(subject, "principal")
        tenant_id = require_tenant_id(
            tenant_id,
            issuer=issuer,
            subject=subject,
            acceptance_path="SQLiteDelegationStore.provision_tenant_identity",
        )
        if not scopes or any(not isinstance(scope, str) or not scope or scope != scope.lower() for scope in scopes):
            raise ValueError("Explicit normalized membership scopes are required")
        if any(not isinstance(alias, str) or not alias for alias in aliases):
            raise ValueError("Tenant aliases must be nonempty exact strings")
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute(
                    "SELECT 1 FROM tenant_identities WHERE tenant=?", (tenant_id,)
                ).fetchone():
                    raise ValueError("Tenant identity is already provisioned")
                for alias in aliases:
                    existing = connection.execute(
                        "SELECT tenant FROM tenant_identity_aliases WHERE issuer=? AND alias=?",
                        (issuer, alias),
                    ).fetchone()
                    if existing is not None and existing[0] != tenant_id:
                        raise ValueError(f"Tenant alias is already assigned: {alias!r}")
                connection.execute(
                    "INSERT INTO tenant_identities VALUES (?, ?, ?, ?)",
                    (tenant_id, issuer, subject, time.time()),
                )
                connection.execute(
                    "INSERT INTO memberships VALUES (?, ?, ?, ?, 1, 1)",
                    (issuer, subject, tenant_id, json.dumps(sorted(scopes))),
                )
                connection.executemany(
                    "INSERT INTO tenant_identity_aliases VALUES (?, ?, ?)",
                    ((issuer, alias, tenant_id) for alias in aliases),
                )
        except ValueError:
            raise
        except sqlite3.Error as exc:
            raise AuthorityUnavailableError("Tenant identity provisioning failed") from exc

    def resolve_tenant_alias(self, *, issuer: str, alias: str) -> str | None:
        issuer = require_identifier(issuer, "issuer")
        if not isinstance(alias, str) or not alias:
            raise ValueError("Tenant alias must be a nonempty exact string")
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT tenant FROM tenant_identity_aliases WHERE issuer=? AND alias=?",
                    (issuer, alias),
                ).fetchone()
        except sqlite3.Error as exc:
            raise AuthorityUnavailableError("Tenant alias lookup failed") from exc
        return (
            require_tenant_id(
                row[0],
                issuer=issuer,
                subject="alias-resolution",
                acceptance_path="SQLiteDelegationStore.resolve_tenant_alias",
            )
            if row is not None
            else None
        )

    def revoke_membership(self, *, issuer: str, subject: str, tenant_id: str) -> None:
        issuer = require_identifier(issuer, "issuer")
        subject = require_identifier(subject, "principal")
        tenant_id = require_tenant_id(
            tenant_id,
            issuer=issuer,
            subject=subject,
            acceptance_path="SQLiteDelegationStore.revoke_membership",
        )
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    "UPDATE memberships SET active=0, version=version+1 WHERE issuer=? AND subject=? AND tenant=?",
                    (issuer, subject, tenant_id),
                )
        except sqlite3.Error as exc:
            raise AuthorityUnavailableError("Delegation authority store unavailable") from exc

    def require_membership(self, *, issuer: str, subject: str, tenant_id: str, scope: str) -> None:
        issuer = require_identifier(issuer, "issuer")
        subject = require_identifier(subject, "principal")
        tenant_id = require_tenant_id(
            tenant_id,
            issuer=issuer,
            subject=subject,
            acceptance_path="SQLiteDelegationStore.require_membership",
        )
        normalized_scope = str(scope or "").strip().lower()
        if not normalized_scope:
            raise ValueError("A required membership scope is required")
        try:
            with closing(self._connect()) as connection:
                membership = connection.execute(
                    "SELECT scopes, active FROM memberships WHERE issuer=? AND subject=? AND tenant=?",
                    (issuer, subject, tenant_id),
                ).fetchone()
                scopes = set(json.loads(membership[0])) if membership else set()
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            raise AuthorityUnavailableError("Delegation authority store unavailable") from exc
        if not membership or not membership[1] or not ({"admin", normalized_scope} & scopes):
            raise AuthorizationError(f"Principal has no active tenant membership for {normalized_scope}")

    def issue(self, token: str, task) -> str:
        principal = self._tokens.verify(token)
        return self.issue_for_principal(principal, task)

    def issue_for_principal(self, principal, task) -> str:
        principal.require_scope("write")
        task_tenant_id = require_tenant_id(
            task.tenant_id,
            issuer=principal.issuer,
            subject=principal.subject,
            acceptance_path="SQLiteDelegationStore.issue",
        )
        if principal.tenant_id != task_tenant_id:
            raise AuthorizationError("Delegation tenant mismatch")
        require_identifier(task.task_id, "task")
        require_identifier(task.session_id, "session")
        require_identifier(task.request.request_id, "request")
        if task.session_id != task.request.session_id or task.tenant_id != task.request.tenant_id:
            raise AuthorizationError("Delegation request identity mismatch")
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                membership = connection.execute(
                    "SELECT scopes, version, active FROM memberships WHERE issuer=? AND subject=? AND tenant=?",
                    (principal.issuer, principal.subject, principal.tenant_id),
                ).fetchone()
                if not membership or not membership[2] or "write" not in json.loads(membership[0]):
                    raise AuthorizationError("Principal has no active tenant membership for write")
                identifier = uuid4().hex
                payload = json.dumps(
                    {
                        "id": identifier, "issuer": principal.issuer, "subject": principal.subject,
                        "token_id": principal.token_id, "tenant_id": principal.tenant_id,
                        "task_id": task.task_id, "session_id": task.session_id,
                        "request_id": task.request.request_id, "operations": ["execute"],
                        "task_digest": self._task_digest(task),
                        "issued_at": time.time(), "expires_at": principal.expires_at,
                        "policy_version": membership[1],
                    },
                    sort_keys=True, separators=(",", ":"),
                )
                signature = self._signature(payload)
                connection.execute(
                    "INSERT INTO delegations (id, payload, signature) VALUES (?, ?, ?)",
                    (identifier, payload, signature),
                )
                connection.commit()
                return f"{identifier}.{signature}"
        except sqlite3.Error as exc:
            raise AuthorityUnavailableError("Delegation authority store unavailable") from exc

    def verify(self, delegation: str, task, *, operation: str) -> None:
        if not isinstance(delegation, str):
            raise AuthorizationError("Missing or malformed delegation")
        identifier, separator, supplied_signature = delegation.partition(".")
        if not separator or not identifier or not supplied_signature:
            raise AuthorizationError("Missing or malformed delegation")
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT d.payload, d.signature, d.revoked, m.scopes, m.version, m.active "
                    "FROM delegations d LEFT JOIN memberships m "
                    "ON m.issuer=json_extract(d.payload, '$.issuer') "
                    "AND m.subject=json_extract(d.payload, '$.subject') "
                    "AND m.tenant=json_extract(d.payload, '$.tenant_id') WHERE d.id=?",
                    (identifier,),
                ).fetchone()
                if not row:
                    raise AuthorizationError("Delegation not found")
                payload, signature, revoked, scopes, version, active = row
                if not hmac.compare_digest(self._signature(payload), signature) or not hmac.compare_digest(
                    signature, supplied_signature
                ):
                    raise AuthorizationError("Invalid delegation signature")
                data = json.loads(payload)
                if revoked:
                    raise AuthorizationError("Delegation revoked")
                if data["id"] != identifier or data["expires_at"] <= time.time():
                    raise AuthorizationError("Delegation expired or invalid")
                if (
                    data["tenant_id"] != task.tenant_id or data["task_id"] != task.task_id
                    or data["session_id"] != task.session_id or data["request_id"] != task.request.request_id
                    or task.request.tenant_id != task.tenant_id or task.request.session_id != task.session_id
                    or operation not in data["operations"]
                    or not hmac.compare_digest(data["task_digest"], self._task_digest(task))
                ):
                    raise AuthorizationError("Delegation identity or operation mismatch")
                if (
                    not active or version != data["policy_version"]
                    or "write" not in json.loads(scopes)
                ):
                    raise AuthorizationError("Delegation membership revoked or policy changed")
        except (sqlite3.Error, ValueError, KeyError, TypeError) as exc:
            raise AuthorityUnavailableError("Delegation authority store unavailable or invalid") from exc

    def revoke(self, delegation: str) -> None:
        identifier, separator, signature = delegation.partition(".")
        if not identifier or not separator:
            raise AuthorizationError("Malformed delegation")
        try:
            with closing(self._connect()) as connection, connection:
                cursor = connection.execute(
                    "UPDATE delegations SET revoked=1 WHERE id=? AND signature=?", (identifier, signature)
                )
                if cursor.rowcount != 1:
                    raise AuthorizationError("Delegation not found")
        except sqlite3.Error as exc:
            raise AuthorityUnavailableError("Delegation authority store unavailable") from exc

    def revoke_session(self, *, tenant_id: str, session_id: str) -> int:
        tenant = require_tenant_id(tenant_id)
        session = require_identifier(session_id, "session")
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    "UPDATE delegations SET revoked=1 WHERE revoked=0 "
                    "AND json_extract(payload, '$.tenant_id')=? "
                    "AND json_extract(payload, '$.session_id')=?",
                    (tenant, session),
                )
                connection.execute(
                    "UPDATE session_revocation_clock SET revision=revision+1 WHERE singleton=1"
                )
                revision_row = connection.execute(
                    "SELECT revision FROM session_revocation_clock WHERE singleton=1"
                ).fetchone()
                if revision_row is None or not isinstance(revision_row[0], int):
                    raise AuthorityUnavailableError("Session revocation clock is unavailable")
                connection.execute(
                    "INSERT INTO session_revocations (tenant_id, session_id, revision) VALUES (?, ?, ?) "
                    "ON CONFLICT (tenant_id, session_id) DO UPDATE SET revision=excluded.revision",
                    (tenant, session, revision_row[0]),
                )
                connection.commit()
                return cursor.rowcount
        except sqlite3.Error as exc:
            raise AuthorityUnavailableError("Delegation authority store unavailable") from exc
