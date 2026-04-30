from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ConfigProposal:
    proposal_id: str
    key: str
    requested_value: Any
    requested_by: str
    created_at: str
    approvals: list[str]
    status: str
    applied_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "key": self.key,
            "requested_value": self.requested_value,
            "requested_by": self.requested_by,
            "created_at": self.created_at,
            "approvals": list(self.approvals),
            "status": self.status,
            "applied_at": self.applied_at,
        }


class ConfigApprovalWorkflow:
    def __init__(self, store_path: Path, *, approvals_required: int = 2):
        self.store_path = Path(store_path)
        self.approvals_required = max(1, int(approvals_required))

    def _load(self) -> dict[str, Any]:
        if not self.store_path.exists():
            return {"proposals": []}
        try:
            return json.loads(self.store_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {"proposals": []}

    def _save(self, payload: dict[str, Any]) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True, default=str),
            encoding="utf-8",
        )

    def propose(
        self,
        *,
        key: str,
        requested_value: Any,
        requested_by: str,
    ) -> ConfigProposal:
        raw = self._load()
        proposal = ConfigProposal(
            proposal_id=uuid.uuid4().hex,
            key=str(key or "").strip(),
            requested_value=requested_value,
            requested_by=str(requested_by or "system").strip() or "system",
            created_at=_utc_now_iso(),
            approvals=[],
            status="pending",
        )
        proposals = list(raw.get("proposals") or [])
        proposals.append(proposal.to_dict())
        raw["proposals"] = proposals
        self._save(raw)
        return proposal

    def approve(self, proposal_id: str, approver: str) -> ConfigProposal | None:
        raw = self._load()
        proposals = list(raw.get("proposals") or [])
        target = None
        for item in proposals:
            if str(item.get("proposal_id") or "") == str(proposal_id or ""):
                target = item
                break
        if target is None:
            return None

        approvals = list(target.get("approvals") or [])
        normalized = str(approver or "").strip()
        if normalized and normalized not in approvals:
            approvals.append(normalized)
        target["approvals"] = approvals
        if len(approvals) >= self.approvals_required and str(target.get("status") or "") == "pending":
            target["status"] = "approved"
        self._save({"proposals": proposals})
        return ConfigProposal(
            **{k: target.get(k) for k in ConfigProposal.__dataclass_fields__.keys()},
        )

    def consume_approved(self, proposal_id: str) -> ConfigProposal | None:
        raw = self._load()
        proposals = list(raw.get("proposals") or [])
        target = None
        for item in proposals:
            if str(item.get("proposal_id") or "") == str(proposal_id or ""):
                target = item
                break
        if target is None:
            return None
        if str(target.get("status") or "") != "approved":
            return None

        target["status"] = "applied"
        target["applied_at"] = _utc_now_iso()
        self._save({"proposals": proposals})
        return ConfigProposal(
            **{k: target.get(k) for k in ConfigProposal.__dataclass_fields__.keys()},
        )


__all__ = ["ConfigApprovalWorkflow", "ConfigProposal"]
