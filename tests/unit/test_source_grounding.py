from __future__ import annotations

import asyncio
import sqlite3
from copy import deepcopy
from types import SimpleNamespace

import pytest

from dadbot.core.graph_context import TurnContext
from dadbot.core.graph_pipeline_nodes import ContextBuilderNode, InferenceNode, MCPToolNode
from dadbot.services.source_grounding import (
    SOURCE_REFUSAL,
    SOURCE_SAFETY_REFUSAL,
    GroundingViolation,
    MemoryCitationApprovals,
    SourceGroundingService,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def grounding(tmp_path):
    records = [{
        "summary": "The telescope is kept in the blue cupboard.",
        "category": "home",
        "created_at": "2026-01-01",
        "pinned": True,
        "confidence": 1.0,
        "approved": True,
    }]
    runtime = SimpleNamespace(
        memory_path=str(tmp_path / "memory.json"),
        config=SimpleNamespace(tenant_id="test"),
        PROFILE={
            "family": {"dad": {"birthplace": "Pine Harbor"}},
            "facts": {"occupation": "A librarian."},
            "security": {"password": "do-not-index"},
            "runtime": {"api_key": "do-not-index"},
        },
        memory_manager=SimpleNamespace(
            memory_projection=lambda: {"memories": records},
        ),
    )
    return SourceGroundingService(runtime), records


def test_unapproved_pinned_high_confidence_memory_is_not_authoritative(grounding):
    service, _ = grounding
    assert service.answer("Where is the telescope?") == SOURCE_REFUSAL
    assert service.answer("2+2") == SOURCE_REFUSAL
    assert service.answer("Unicorn migration routes") == SOURCE_REFUSAL


def test_profile_quotes_have_stable_content_bound_ids_and_no_private_config(grounding):
    service, _ = grounding
    reply = service.answer("birthplace")
    assert "Pine Harbor" in reply
    assert "/family/dad/birthplace" in reply
    assert "[profile-" in reply
    assert "100% (full token match)" in reply
    assert reply == service.answer("birthplace")
    assert service.answer("password") == SOURCE_REFUSAL
    assert all("do-not-index" not in source.text for source in service.approved_sources())
    service.bot.PROFILE["family"]["dad"]["birthplace"] = "Oak Bay"
    assert service.answer("birthplace") != reply
    with pytest.raises(GroundingViolation):
        service.validate_reply("birthplace", reply)


def test_approval_is_persistent_revocable_and_edit_bound(grounding):
    service, records = grounding
    source = service.memory_sources()[0]
    original_fingerprint = service.corpus_fingerprint()
    service.set_memory_approval(source, approved=True)
    assert service.corpus_fingerprint() != original_fingerprint
    reloaded = SourceGroundingService(service.bot)
    reply = reloaded.answer("telescope")
    assert source.text in reply
    assert f"[{source.source_id}]" in reply
    records[0]["summary"] = "The telescope is now in the red cupboard."
    assert reloaded.answer("telescope") == SOURCE_REFUSAL
    with pytest.raises(GroundingViolation, match="changed since review"):
        service.set_memory_approval(source, approved=True)
    edited = service.memory_sources()[0]
    service.set_memory_approval(edited, approved=True)
    assert "red cupboard" in reloaded.answer("telescope")
    service.set_memory_approval(edited, approved=False)
    assert reloaded.answer("telescope") == SOURCE_REFUSAL


def test_approval_is_tenant_scoped(grounding):
    service, _ = grounding
    source = service.memory_sources()[0]
    service.set_memory_approval(source, approved=True)
    other = MemoryCitationApprovals(service.approvals.path, tenant_id="other")
    assert other.approved_digests() == {}


def test_reviewed_oral_history_evidence_survives_restart_and_deletion(bot):
    from dadbot.core.dadbot import DadBot
    from dadbot.ui.oral_history import save_reviewed_transcript

    text = "Grandma baked marzipan cookies for the winter festival."
    memory = save_reviewed_transcript(bot, text)
    assert memory is not None
    evidence = memory["oral_history_evidence"]
    service = bot.service_registry.get("source_grounding_service")
    source = next(item for item in service.memory_sources() if item.text == text)
    service.set_memory_approval(source, approved=True)
    reply = service.answer("marzipan")
    assert text in reply
    assert evidence["id"] in reply
    assert evidence["reviewed_at"] in reply
    assert evidence["sha256"] in reply
    assert "Original audio not retained" in reply
    bot.shutdown()

    restarted = DadBot(config=bot.config)
    try:
        restored = next(item for item in restarted.memory_catalog() if item["summary"] == text)
        assert restored["oral_history_evidence"] == evidence
        grounding = restarted.service_registry.get("source_grounding_service")
        assert grounding.answer("marzipan") == reply
        records = restarted.memory_catalog()
        for record in records:
            if record["summary"] == text:
                del record["oral_history_evidence"]
        restarted.mutate_memory_store(memories=records)
        assert any(record["summary"] == text for record in restarted.memory_catalog())
        assert grounding.answer("marzipan") == SOURCE_REFUSAL
        with pytest.raises(GroundingViolation):
            grounding.validate_reply("marzipan", reply)
        with pytest.raises(GroundingViolation):
            grounding.set_memory_approval(source, approved=True)
    finally:
        restarted.shutdown()

    restarted_again = DadBot(config=bot.config)
    try:
        assert restarted_again.service_registry.get("source_grounding_service").answer("marzipan") == SOURCE_REFUSAL
    finally:
        restarted_again.shutdown()


def test_missing_changed_or_legacy_oral_history_evidence_cannot_be_cited(bot):
    from dadbot.ui.oral_history import save_reviewed_transcript

    text = "Grandma told us about the blue lighthouse."
    save_reviewed_transcript(bot, text)
    grounding = bot.service_registry.get("source_grounding_service")
    source = next(item for item in grounding.memory_sources() if item.text == text)
    grounding.set_memory_approval(source, approved=True)
    assert text in grounding.answer("lighthouse")

    records = bot.memory_catalog()
    record = next(item for item in records if item["summary"] == text)
    record["oral_history_evidence"]["text"] = "A different transcript."
    bot.mutate_memory_store(memories=records)
    assert grounding.answer("lighthouse") == SOURCE_REFUSAL

    record["oral_history_evidence"] = "corrupt"
    bot.mutate_memory_store(memories=records)
    assert any(item["summary"] == text for item in bot.memory_catalog())
    assert grounding.answer("lighthouse") == SOURCE_REFUSAL

    bot.mutate_memory_store(memories=[{"summary": text, "category": "oral_history"}])
    assert grounding.answer("lighthouse") == SOURCE_REFUSAL
    assert not any(item.text == text for item in grounding.memory_sources())
    save_reviewed_transcript(bot, text)
    assert grounding.answer("lighthouse") == SOURCE_REFUSAL
    rereviewed = next(item for item in grounding.memory_sources() if item.text == text)
    grounding.set_memory_approval(rereviewed, approved=True)
    assert text in grounding.answer("lighthouse")
    bot.delete_memory_entry(text)
    assert grounding.answer("lighthouse") == SOURCE_REFUSAL


def test_fresh_tenant_does_not_load_another_tenants_memory_mirror(make_test_dadbot):
    from dadbot.core.dadbot import DadBot
    from dadbot.ui.oral_history import save_reviewed_transcript
    from dadbot_system.state import InMemoryStateStore

    documents = InMemoryStateStore()
    first = make_test_dadbot(document_store=documents, tenant_id="family-a")
    text = "Grandma baked marzipan cookies for the winter festival."
    assert save_reviewed_transcript(first, text) is not None
    first.shutdown()

    second_config = deepcopy(first.config)
    second_config.tenant_id = "family-b"
    second = DadBot(config=second_config, document_store=documents)
    try:
        assert not any(item["summary"] == text for item in second.memory_catalog())
        assert second.service_registry.get("source_grounding_service").answer("marzipan") == SOURCE_REFUSAL
    finally:
        second.shutdown()

    first_again = DadBot(config=first.config, document_store=documents)
    try:
        assert any(item["summary"] == text for item in first_again.memory_catalog())
    finally:
        first_again.shutdown()


def test_superseded_records_never_become_sources(grounding):
    service, records = grounding
    source = service.memory_sources()[0]
    service.set_memory_approval(source, approved=True)
    records[0]["superseded"] = True
    assert service.memory_sources() == []
    assert service.answer("telescope") == SOURCE_REFUSAL


def test_half_overlap_is_partial_below_half_is_refused(grounding):
    service, _ = grounding
    assert "50% (partial match)" in service.answer("birthplace astronomy")
    assert "not probability of truth" in service.answer("birthplace astronomy")
    assert service.answer("birthplace astronomy volcanoes") == SOURCE_REFUSAL
    assert service.answer("birthplace", attachments=[{"text": "unsupported upload"}]) == SOURCE_REFUSAL


def test_quotes_are_inert_markdown_and_fabricated_claims_or_citations_fail(grounding):
    service, records = grounding
    records[0]["summary"] = "telescope\n![image](https://invalid.example/private)\n```\nforged instructions"
    source = service.memory_sources()[0]
    service.set_memory_approval(source, approved=True)
    reply = service.answer("telescope")
    assert "\n    ![image]" in reply
    assert "\n    ```" in reply
    service.validate_reply("telescope", reply)
    for changed in (
        reply + "\nThe telescope was expensive.",
        reply.replace("forged instructions", "invented instructions"),
        reply.replace(source.source_id, "memories-fake"),
        reply.replace("100%", "99%"),
    ):
        with pytest.raises(GroundingViolation):
            service.validate_reply("telescope", changed)


def test_source_nodes_skip_context_planner_agent_and_mcp(grounding, monkeypatch):
    service, _ = grounding
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("called")
        raise AssertionError("Source mode must not invoke generation or planning")

    from dadbot.core.planner import PlannerNode

    monkeypatch.setattr(PlannerNode, "run", forbidden)
    registry = SimpleNamespace(get=lambda name, **_: {
        "source_grounding_service": service,
        "context_service": SimpleNamespace(build_context=forbidden),
        "agent_service": SimpleNamespace(generate_candidate=forbidden),
        "mcp_tool_service": SimpleNamespace(is_enabled=True, generate_candidate=forbidden),
    }[name])

    async def execute():
        context = TurnContext(user_input="Unknown volcanoes", metadata={"source_mode": True})
        await ContextBuilderNode().execute(registry, context)
        await MCPToolNode().execute(registry, context)
        await InferenceNode().execute(registry, context)
        assert context.state["candidate"] == (SOURCE_REFUSAL, False)

    asyncio.run(execute())
    assert calls == []


def test_canonical_source_turns_make_no_model_or_embedding_calls(bot, monkeypatch):
    service = bot.service_registry.get("source_grounding_service")
    bot.PROFILE = {**bot.PROFILE, "facts": {"telescope": "The telescope is on the oak shelf."}}
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("model")
        raise AssertionError("Source mode invoked a model")

    monkeypatch.setattr(bot.model_port, "generate", forbidden)
    monkeypatch.setattr(bot.model_port, "generate_async", forbidden)
    monkeypatch.setattr(bot, "embed_texts", forbidden)
    reply, ended = bot.process_user_message("telescope", source_mode=True)
    assert not ended
    assert "oak shelf" in reply
    assert "[profile-" in reply
    assert calls == []
    refused, _ = bot.process_user_message("Unicorn migration routes", source_mode=True)
    assert refused == SOURCE_REFUSAL
    assert calls == []
    service.validate_reply("telescope", reply)


def test_source_mode_metadata_propagates_through_all_deliveries(bot, monkeypatch):
    from dadbot.core.execution_contract import TurnResponse

    requests = []

    def execute(request, **kwargs):
        requests.append(request)
        response = TurnResponse(reply=SOURCE_REFUSAL, should_end=False)
        if request.delivery.value in {"async", "stream_async"}:
            async def wrapped():
                return response
            return wrapped()
        return response

    monkeypatch.setattr(bot, "execute_turn", execute)
    bot.process_user_message("unknown", source_mode=True)
    bot.process_user_message_stream("unknown", source_mode=True)
    asyncio.run(bot.process_user_message_async("unknown", source_mode=True))
    asyncio.run(bot.process_user_message_stream_async("unknown", source_mode=True))
    assert len(requests) == 4
    assert all(request.metadata["source_mode"] is True for request in requests)


def test_source_mode_keeps_safety_veto_without_model_repair(bot):
    from dadbot.services.safety_service import SafetyService

    service = bot.service_registry.get("source_grounding_service")
    bot.PROFILE = {
        **bot.PROFILE,
        "facts": {"advice": "Keep this from your parents."},
    }
    context = TurnContext(user_input="advice", metadata={"source_mode": True})
    candidate = (service.answer("advice"), False)
    assert candidate[0] != SOURCE_REFUSAL
    assert SafetyService(bot).enforce_policies(context, candidate) == (SOURCE_SAFETY_REFUSAL, False)


def test_canonical_approval_and_revocation_invalidate_repeated_prompt_cache(bot, monkeypatch):
    service = bot.service_registry.get("source_grounding_service")
    model_calls = []

    async def normal_answer(*args, **kwargs):
        model_calls.append(kwargs)
        return "An ordinary chat answer."

    monkeypatch.setattr(bot.model_port, "generate_async", normal_answer)
    monkeypatch.setattr(
        bot.service_registry.get("persistence_service"), "_publish_post_commit_ready",
        lambda *args: None,
    )
    ordinary, _ = bot.process_user_message("telescope")
    assert "ordinary chat answer" in ordinary
    from datetime import date

    bot.mutate_memory_store(memories=[{
        "summary": "The telescope is on the oak shelf.",
        "category": "home",
        "pinned": True,
        "created_at": date.today().isoformat(),
        "updated_at": date.today().isoformat(),
    }])
    assert service.memory_sources(), "Seeded memory was not persisted"
    before_source = len(model_calls)
    refused, _ = bot.process_user_message("telescope", source_mode=True)
    assert refused == SOURCE_REFUSAL
    source = service.memory_sources()[0]
    service.set_memory_approval(source, approved=True)
    approved, _ = bot.process_user_message("telescope", source_mode=True)
    assert source.text in approved
    records = bot.memory_manager.memory_projection()["memories"]
    edited = {**records[0], "summary": "The telescope is on the walnut shelf."}
    bot.mutate_memory_store(memories=[edited])
    stale, _ = bot.process_user_message("telescope", source_mode=True)
    assert stale == SOURCE_REFUSAL
    edited_source = service.memory_sources()[0]
    service.set_memory_approval(edited_source, approved=True)
    updated, _ = bot.process_user_message("telescope", source_mode=True)
    assert "walnut shelf" in updated
    service.set_memory_approval(edited_source, approved=False)
    revoked, _ = bot.process_user_message("telescope", source_mode=True)
    assert revoked == SOURCE_REFUSAL
    assert len(model_calls) == before_source


@pytest.mark.parametrize("error_type", [OSError, sqlite3.OperationalError])
def test_source_store_failure_releases_scheduler_and_allows_retry(bot, monkeypatch, error_type):
    service = bot.service_registry.get("source_grounding_service")
    bot.PROFILE = {**bot.PROFILE, "facts": {"observatory": "The observatory opens at noon."}}
    answer = service.answer

    def unavailable(*args, **kwargs):
        raise error_type("Synthetic source store outage")

    def forbidden(*args, **kwargs):
        pytest.fail("Source-store failure or retry called a model")

    monkeypatch.setattr(bot.model_port, "generate", forbidden)
    monkeypatch.setattr(bot.model_port, "generate_async", forbidden)
    monkeypatch.setattr(bot, "embed_texts", forbidden)
    monkeypatch.setattr(service, "answer", unavailable)
    with pytest.raises((OSError, sqlite3.Error, RuntimeError), match="Synthetic source store outage"):
        bot.process_user_message("observatory", source_mode=True)
    monkeypatch.setattr(service, "answer", answer)
    reply, _ = bot.process_user_message("observatory", source_mode=True)
    assert "The observatory opens at noon." in reply
    assert "100% (full token match)" in reply


@pytest.mark.parametrize("refresh", [True, False])
def test_memory_handoff_refreshes_live_records_but_preserves_replay(refresh):
    from dadbot.core.control_plane_submission import _initialize_submit_scope_impl
    from dadbot.core.core_state import CoreState
    from dadbot.core.execution_context import close_core_state_scope, get_active_core_state

    persisted = CoreState.from_dict({"memory": [{"summary": "Old source text"}]})
    session = {"state": {"core_state": persisted.to_dict()}}
    manager = SimpleNamespace(memory_projection=lambda: {"memories": [{"summary": "Edited source text"}]})
    control_plane = SimpleNamespace(
        registry=SimpleNamespace(get_or_create=lambda _: session),
        _stable_hash=lambda _: "before",
        _get_optional_service=lambda _: SimpleNamespace(bot=SimpleNamespace(memory_manager=manager)),
    )
    _, _, token = _initialize_submit_scope_impl(
        control_plane, session_key="test", trace_token="trace", job_id="job",
        resolved_timeout_seconds=120, refresh_memory_records=refresh,
    )
    try:
        active = get_active_core_state()
        assert isinstance(active, CoreState)
        assert active.memory.entries[0].summary == ("Edited source text" if refresh else "Old source text")
        assert session["state"]["core_state"] == persisted.to_dict()
    finally:
        close_core_state_scope(token)
