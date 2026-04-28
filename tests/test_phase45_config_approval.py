from __future__ import annotations

from pathlib import Path

from dadbot.config import DadBotConfig
from dadbot.core.config_approval import ConfigApprovalWorkflow


def test_config_approval_workflow_requires_multiple_approvals(tmp_path: Path) -> None:
    workflow = ConfigApprovalWorkflow(tmp_path / "approvals.json", approvals_required=2)
    proposal = workflow.propose(key="llm.profile", requested_value={"provider": "ollama", "model": "llama3.2"}, requested_by="alice")
    first = workflow.approve(proposal.proposal_id, "bob")
    assert first is not None
    assert first.status == "pending"
    second = workflow.approve(proposal.proposal_id, "carol")
    assert second is not None
    assert second.status == "approved"


def test_dadbot_config_applies_only_approved_change(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DADBOT_SESSION_LOG_DIR", str(tmp_path))
    config = DadBotConfig()
    proposal_id = config.propose_profile_llm_change("ollama", "llama3.2", requested_by="alice")
    assert not config.apply_profile_llm_change_if_approved(proposal_id)
    assert not config.approve_profile_llm_change(proposal_id, "bob")
    assert config.approve_profile_llm_change(proposal_id, "carol")
    assert config.apply_profile_llm_change_if_approved(proposal_id)
    assert config.llm_provider == "ollama"
    assert config.llm_model == "llama3.2"
