from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest

from dadbot_system.delegation import SQLiteDelegationStore
from dadbot_system.execution_identity import require_tenant_id
from dadbot_system.security import ServiceTokenManager
from dadbot_system.tenant_identity import (
    issue_tenant_identity,
    mint_tenant_id,
    parse_tenant_id,
    resolve_tenant_reference,
)
from tools.tenant_identity_migration import TenantObservation, render_dry_run


@pytest.mark.parametrize(
    "value",
    [
        "tenant/a",
        "TENANT-A",
        "tenant a",
        "tenant:a",
        "Alpha",
        "alpha",
        "-alpha-",
        None,
        "",
        "   ",
        "....",
        "é",
        "e\u0301",
        "tenant-" + ("a" * 300),
        "t1_" + ("a" * 27),
    ],
)
def test_tenant_id_contract_rejects_noncanonical_values(value):
    with pytest.raises(ValueError, match="tenant identity must be canonical"):
        parse_tenant_id(value)


def test_tenant_id_contract_accepts_an_issued_opaque_id_without_rewriting():
    issued = str(mint_tenant_id())

    assert issued.startswith("t1_")
    assert require_tenant_id(issued) == issued
    assert len(issued) == 29


def test_tenant_id_contract_preserves_explicit_legacy_canonical_id():
    assert require_tenant_id("tenant-a") == "tenant-a"


def test_lossy_normalizer_name_now_fails_closed():
    from dadbot_system.contracts import normalize_tenant_id

    assert normalize_tenant_id("tenant-a") == "tenant-a"
    with pytest.raises(ValueError):
        normalize_tenant_id(None)
    with pytest.raises(ValueError):
        normalize_tenant_id("tenant/a")


def test_tenant_id_provisioning_records_membership_and_exact_alias(tmp_path):
    authority_path = str(tmp_path / "authority.sqlite")
    SQLiteDelegationStore.initialize(authority_path)
    authority = SQLiteDelegationStore(authority_path, "provisioning-secret")

    tenant_id = issue_tenant_identity(
        authority,
        issuer="provisioner",
        subject="service-account",
        scopes={"read", "write"},
        aliases=("tenant/a",),
    )

    assert require_tenant_id(tenant_id) == tenant_id
    assert resolve_tenant_reference(
        "tenant/a",
        issuer="provisioner",
        resolver=authority,
    ) == tenant_id
    assert authority.resolve_tenant_alias(issuer="provisioner", alias="tenant-a") is None

    with authority._connect() as connection:
        identity = connection.execute(
            "SELECT issued_by_issuer, issued_by_subject FROM tenant_identities WHERE tenant=?",
            (tenant_id,),
        ).fetchone()
        membership = connection.execute(
            "SELECT scopes, active FROM memberships WHERE issuer=? AND subject=? AND tenant=?",
            ("provisioner", "service-account", tenant_id),
        ).fetchone()
    assert identity == ("provisioner", "service-account")
    assert json.loads(membership[0]) == ["read", "write"]
    assert membership[1] == 1


def test_tenant_provisioning_rejects_alias_collision(tmp_path):
    authority_path = str(tmp_path / "authority.sqlite")
    SQLiteDelegationStore.initialize(authority_path)
    authority = SQLiteDelegationStore(authority_path, "provisioning-secret")
    issue_tenant_identity(
        authority,
        issuer="provisioner",
        subject="first",
        scopes={"read"},
        aliases=("legacy-id",),
    )

    with pytest.raises(ValueError, match="already assigned"):
        issue_tenant_identity(
            authority,
            issuer="provisioner",
            subject="second",
            scopes={"read"},
            aliases=("legacy-id",),
        )


def test_token_issuance_validates_but_verification_preserves_signed_legacy_claim(caplog):
    secret = "tenant-identity-compat-secret"
    manager = ServiceTokenManager(secret)
    with pytest.raises(ValueError, match="tenant identity must be canonical"):
        manager.issue(subject="legacy", tenant_id="tenant/a", scopes={"read"})

    payload = {
        "v": 1,
        "sub": "legacy",
        "tenant_id": "tenant/a",
        "scopes": ["read"],
        "allowed_tools": [],
        "iat": time.time(),
        "exp": time.time() + 300,
        "iss": "dadbot",
        "jti": "legacy-token",
    }
    encoded_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(encoded_payload).decode("ascii").rstrip("=")
    signature = hmac.new(secret.encode("utf-8"), encoded_payload, hashlib.sha256).hexdigest()

    principal = manager.verify(f"{encoded}.{signature}")

    assert principal.tenant_id == "tenant/a"
    accepted = [
        record
        for record in caplog.records
        if record.getMessage() == "Legacy tenant identity accepted during compatibility window"
    ]
    assert len(accepted) == 1
    assert accepted[0].tenant_issuer == "dadbot"
    assert accepted[0].tenant_subject == "legacy"
    assert accepted[0].tenant_accepted_at_unix > 0
    assert accepted[0].tenant_acceptance_path == "ServiceTokenManager.verify"
    with pytest.raises(ValueError, match="tenant identity must be canonical"):
        require_tenant_id(principal.tenant_id)


def test_legacy_token_issuance_logs_actor_and_timestamp(caplog):
    manager = ServiceTokenManager("legacy-identity-log-secret")

    manager.issue(subject="legacy-client", tenant_id="tenant-a", scopes={"read"})

    record = next(
        entry
        for entry in caplog.records
        if entry.getMessage() == "Legacy tenant identity accepted during compatibility window"
    )
    assert record.tenant_issuer == "dadbot"
    assert record.tenant_subject == "legacy-client"
    assert record.tenant_accepted_at_unix > 0
    assert record.tenant_acceptance_path == "ServiceTokenManager.issue"


def test_legacy_membership_revocation_logs_actor_and_timestamp(tmp_path, caplog):
    authority_path = str(tmp_path / "authority.sqlite")
    SQLiteDelegationStore.initialize(authority_path)
    authority = SQLiteDelegationStore(authority_path, "provisioning-secret")
    authority.grant_membership(issuer="issuer", subject="subject", tenant_id="tenant-a", scopes={"read"})
    caplog.clear()

    authority.revoke_membership(issuer="issuer", subject="subject", tenant_id="tenant-a")

    record = next(
        entry
        for entry in caplog.records
        if entry.getMessage() == "Legacy tenant identity accepted during compatibility window"
    )
    assert record.tenant_issuer == "issuer"
    assert record.tenant_subject == "subject"
    assert record.tenant_accepted_at_unix > 0
    assert record.tenant_acceptance_path == "SQLiteDelegationStore.revoke_membership"


def test_migration_dry_run_reports_collisions_without_resolving_them():
    report = render_dry_run(
        [
            TenantObservation("tenant/a", "fixture-a:1"),
            TenantObservation("tenant-a", "fixture-b:2"),
            TenantObservation(None, "fixture-c:3"),
            TenantObservation("default", "fixture-d:4"),
        ]
    )

    assert "DRY RUN ONLY" in report
    assert "UNRESOLVED COLLISION" in report
    assert "`tenant-a`" in report
    assert "`default`" in report
