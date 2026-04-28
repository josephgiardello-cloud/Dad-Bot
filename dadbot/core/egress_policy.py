from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse


@dataclass(frozen=True)
class EgressDecision:
    allowed: bool
    reason: str
    host: str
    url: str


def _normalize_hosts(hosts: Iterable[str]) -> set[str]:
    normalized: set[str] = set()
    for host in hosts:
        value = str(host or "").strip().lower()
        if value:
            normalized.add(value)
    return normalized


def default_allowlist() -> tuple[str, ...]:
    configured = str(os.environ.get("DADBOT_EGRESS_ALLOWLIST", "")).strip()
    if configured:
        hosts = [item.strip() for item in configured.split(",")]
        return tuple(host for host in hosts if host)
    return ("localhost", "127.0.0.1", "api.duckduckgo.com")


def is_enforced() -> bool:
    value = str(os.environ.get("DADBOT_EGRESS_ENFORCE", "0")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def evaluate_url(url: str, *, allowlist: Iterable[str] | None = None) -> EgressDecision:
    parsed = urlparse(str(url or "").strip())
    host = str(parsed.hostname or "").strip().lower()
    if not host:
        return EgressDecision(allowed=False, reason="missing_host", host="", url=str(url or ""))
    allowed_hosts = _normalize_hosts(allowlist or default_allowlist())
    if host in allowed_hosts:
        return EgressDecision(allowed=True, reason="allowlisted", host=host, url=str(url or ""))
    return EgressDecision(allowed=False, reason="host_not_allowlisted", host=host, url=str(url or ""))


def enforce_url(url: str, *, allowlist: Iterable[str] | None = None) -> EgressDecision:
    decision = evaluate_url(url, allowlist=allowlist)
    if decision.allowed or not is_enforced():
        return decision
    raise PermissionError(
        f"Egress blocked by allowlist policy: host={decision.host!r} reason={decision.reason} url={decision.url!r}"
    )


__all__ = [
    "EgressDecision",
    "default_allowlist",
    "enforce_url",
    "evaluate_url",
    "is_enforced",
]
