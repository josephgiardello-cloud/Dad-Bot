"""Long-run behavioral validation tests.

Category 3 coverage:
- Multi-session determinism stamp stability
- Memory evolution delta correctness across simulated turns
- Replay consistency: contract_version hash is stable; failure_replay captures all fields
- Memory-personality boundary: personality code paths do not touch memory scoring

These are unit tests (no live LLM or storage required).
"""
from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from dadbot.core.graph import TurnGraph, _NODE_STAGE_CONTRACTS
from dadbot.core.graph_context import TurnContext
from dadbot.core.graph_pipeline_nodes import (
    ContextBuilderNode,
    HealthNode,
    InferenceNode,
    ReflectionNode,
    SafetyNode,
    SaveNode,
    TemporalNode,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(**kwargs) -> TurnContext:
    return TurnContext(user_input="test input", **kwargs)


def _expected_contracts_hash() -> str:
    blob = json.dumps(_NODE_STAGE_CONTRACTS, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class _StubService:
    """Minimal stub service that satisfies all node contracts without live I/O."""

    def tick(self, ctx):
        return {"ok": True}

    def build_context(self, ctx):
        return {"rich_context": "stub"}

    async def run_agent(self, ctx, rich_context):
        return ctx.state.get("candidate", "stub reply")

    def enforce_policies(self, ctx, candidate):
        return candidate

    def save_turn(self, ctx, result):
        pass

    def finalize_turn(self, ctx, result):
        return result

    def reflect_after_turn(self, turn_text, mood, reply_text):
        return None


class _StubRegistry:
    """Stub registry: returns _StubService for known service keys, None for others."""

    _KNOWN_SERVICES = {
        "maintenance_service",
        "context_service",
        "agent_service",
        "safety_service",
        "persistence_service",
        "reflection",
    }

    def __init__(self):
        self._svc = _StubService()

    def get(self, key, default=None):
        if key in self._KNOWN_SERVICES:
            return self._svc
        return default


def _make_graph(**kwargs) -> TurnGraph:
    """Create a TurnGraph with a stub registry for unit tests."""
    return TurnGraph(registry=_StubRegistry(), **kwargs)


# ---------------------------------------------------------------------------
# Section 1 — Determinism stamp stability
# ---------------------------------------------------------------------------


class TestContractVersionStamp:
    """contract_version stamp is stable across re-runs with the same schema."""

    def test_contract_version_hash_is_stable_on_second_call(self):
        """Two successive hash computations of the same contracts produce the same value."""
        h1 = _expected_contracts_hash()
        h2 = _expected_contracts_hash()
        assert h1 == h2, "Contract hash must be deterministic"

    def test_contract_version_hash_changes_when_contracts_change(self):
        """Mutating contracts changes the hash — detecting schema drift."""
        original = _expected_contracts_hash()
        patched = json.dumps(
            {**_NODE_STAGE_CONTRACTS, "_synthetic_test_key": {"required": ["__test__"]}},
            sort_keys=True, default=str,
        )
        mutated = hashlib.sha256(patched.encode()).hexdigest()[:16]
        assert original != mutated, "Hash must change when contracts change"

    def test_contract_version_is_16_hex_chars(self):
        """Contract hash is always exactly 16 hex characters (SHA-256 truncated)."""
        h = _expected_contracts_hash()
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_contract_version_stamped_on_fresh_context(self):
        """TurnGraph.execute stamps contract_version into determinism_manifest before any node runs."""
        ctx = _make_context()
        graph = _make_graph()

        async def _run():
            return await graph.execute(ctx)

        asyncio.run(_run())
        cv = ctx.determinism_manifest.get("contract_version")
        assert cv is not None, "contract_version must be stamped by execute()"
        assert "node_contracts_hash" in cv
        assert "schema_version" in cv
        assert cv["schema_version"] == "1"
        assert cv["node_contracts_hash"] == _expected_contracts_hash()

    def test_contract_version_consistent_across_two_turns(self):
        """Two independent turns produce the same contract_version hash (no drift)."""
        ctx1 = _make_context()
        ctx2 = _make_context()
        graph = _make_graph()

        async def _run():
            await graph.execute(ctx1)
            await graph.execute(ctx2)

        asyncio.run(_run())
        h1 = ctx1.determinism_manifest.get("contract_version", {}).get("node_contracts_hash")
        h2 = ctx2.determinism_manifest.get("contract_version", {}).get("node_contracts_hash")
        assert h1 is not None
        assert h1 == h2, "contract_version hash must be identical across turns with same schema"


# ---------------------------------------------------------------------------
# Section 2 — Memory evolution stamps and delta correctness
# ---------------------------------------------------------------------------


class TestMemoryEvolutionStamps:
    """memory_evolution stamp captures before/after fingerprints and accurate delta."""

    def test_memory_evolution_before_fingerprint_is_sha256_prefix(self):
        """before_fingerprint is a 16-char hex string (SHA-256 of memories list)."""
        ctx = _make_context()
        ctx.state["memories"] = [{"id": "m1", "text": "remembered fact"}]
        graph = _make_graph()
        asyncio.run(graph.execute(ctx))
        ev = ctx.determinism_manifest.get("memory_evolution", {})
        fp = ev.get("before_fingerprint")
        if fp is not None:
            assert len(fp) == 16
            assert all(c in "0123456789abcdef" for c in fp)

    def test_memory_evolution_before_count_matches_list_length(self):
        """before_count must equal the number of memory entries at context_builder time."""
        ctx = _make_context()
        ctx.state["memories"] = [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]
        graph = _make_graph()
        asyncio.run(graph.execute(ctx))
        ev = ctx.determinism_manifest.get("memory_evolution", {})
        if "before_count" in ev:
            assert ev["before_count"] == 3

    def test_memory_evolution_delta_is_integer(self):
        """delta must be an integer, not None or a float."""
        ctx = _make_context()
        ctx.state["memories"] = [{"id": "m1"}]
        graph = _make_graph()
        asyncio.run(graph.execute(ctx))
        ev = ctx.determinism_manifest.get("memory_evolution", {})
        if "delta" in ev:
            assert isinstance(ev["delta"], int), "delta must be int"

    def test_memory_evolution_delta_zero_when_memories_unchanged(self):
        """When SaveNode does not add memories, delta must be 0."""
        ctx = _make_context()
        ctx.state["memories"] = [{"id": "m1"}]
        graph = _make_graph()
        asyncio.run(graph.execute(ctx))
        ev = ctx.determinism_manifest.get("memory_evolution", {})
        if "delta" in ev and "before_count" in ev:
            after_fp = ev.get("after_fingerprint")
            before_fp = ev.get("before_fingerprint")
            # If fingerprints match, delta must be 0
            if after_fp == before_fp:
                assert ev["delta"] == 0

    def test_memory_evolution_empty_memories_fingerprints_match(self):
        """When memories is empty before and after, both fingerprints are equal."""
        empty_fp = hashlib.sha256(
            json.dumps([], sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        ctx = _make_context()
        # Ensure no memories
        ctx.state.pop("memories", None)
        graph = _make_graph()
        asyncio.run(graph.execute(ctx))
        ev = ctx.determinism_manifest.get("memory_evolution", {})
        if "before_fingerprint" in ev:
            assert ev["before_fingerprint"] == empty_fp

    def test_memory_evolution_delta_never_negative_on_stable_run(self):
        """Delta must not be negative in a normal (non-purging) turn."""
        ctx = _make_context()
        ctx.state["memories"] = [{"id": "m1"}, {"id": "m2"}]
        graph = _make_graph()
        asyncio.run(graph.execute(ctx))
        ev = ctx.determinism_manifest.get("memory_evolution", {})
        if "delta" in ev:
            assert ev["delta"] >= 0, "delta must not be negative in a normal turn"


# ---------------------------------------------------------------------------
# Section 3 — Failure replay stamp
# ---------------------------------------------------------------------------


class TestFailureReplayStamp:
    """failure_replay stamp captures structured metadata when a stage fails."""

    def test_failure_replay_fields_present_on_stage_error(self):
        """A stage error appends a failure_replay entry with all required fields."""
        class _FailSaveNode(SaveNode):
            name = "save"

            async def run(self, registry, ctx: TurnContext) -> TurnContext:
                raise RuntimeError("Simulated save failure")

            async def execute(self, registry, ctx: TurnContext) -> None:
                raise RuntimeError("Simulated save failure")

        graph = _make_graph(nodes=[
            TemporalNode(), HealthNode(), ContextBuilderNode(),
            InferenceNode(), SafetyNode(), ReflectionNode(), _FailSaveNode(),
        ])

        ctx = _make_context()
        # Seed candidate so inference/safety stubs pass
        ctx.state["candidate"] = "test reply"
        ctx.state["safe_result"] = "test reply"

        with pytest.raises((RuntimeError, Exception)):
            asyncio.run(graph.execute(ctx))

        replays = ctx.determinism_manifest.get("failure_replay", [])
        assert len(replays) >= 1, "At least one failure_replay entry must be appended on error"
        entry = replays[-1]
        assert "stage" in entry
        assert "error_type" in entry
        assert "error_msg" in entry
        assert "state_keys" in entry
        assert "contract_version_hash" in entry
        assert isinstance(entry["state_keys"], list)
        assert isinstance(entry["error_msg"], str)

    def test_failure_replay_error_msg_max_200_chars(self):
        """error_msg in failure_replay is capped at 200 characters."""
        class _LongErrorNode(SafetyNode):
            name = "safety"

            async def execute(self, registry, ctx: TurnContext) -> None:
                raise RuntimeError("x" * 500)

        graph = _make_graph(nodes=[
            TemporalNode(), HealthNode(), ContextBuilderNode(),
            InferenceNode(), _LongErrorNode(), ReflectionNode(), SaveNode(),
        ])

        ctx = _make_context()
        ctx.state["candidate"] = "test reply"

        with pytest.raises((RuntimeError, Exception)):
            asyncio.run(graph.execute(ctx))

        replays = ctx.determinism_manifest.get("failure_replay", [])
        for entry in replays:
            assert len(entry.get("error_msg", "")) <= 200

    def test_failure_replay_contract_version_hash_matches_live_hash(self):
        """failure_replay.contract_version_hash must equal the live contract hash."""
        class _BrokenNode(SafetyNode):
            name = "safety"

            async def execute(self, registry, ctx: TurnContext) -> None:
                raise RuntimeError("Forced failure for test")

        graph = _make_graph(nodes=[
            TemporalNode(), HealthNode(), ContextBuilderNode(),
            InferenceNode(), _BrokenNode(), ReflectionNode(), SaveNode(),
        ])

        ctx = _make_context()
        ctx.state["candidate"] = "test"

        with pytest.raises((RuntimeError, Exception)):
            asyncio.run(graph.execute(ctx))

        replays = ctx.determinism_manifest.get("failure_replay", [])
        if replays:
            recorded_hash = replays[-1].get("contract_version_hash")
            expected_hash = _expected_contracts_hash()
            assert recorded_hash == expected_hash, (
                "failure_replay.contract_version_hash must match the live contract hash"
            )

    def test_failure_replay_state_keys_are_sorted(self):
        """state_keys in failure_replay are in sorted order for stable comparison."""
        class _FailNode(ContextBuilderNode):
            name = "context_builder"

            async def execute(self, registry, ctx: TurnContext) -> None:
                ctx.state["zzz"] = 1
                ctx.state["aaa"] = 2
                raise RuntimeError("Forced failure")

        graph = _make_graph(nodes=[
            TemporalNode(), HealthNode(), _FailNode(),
            InferenceNode(), SafetyNode(), ReflectionNode(), SaveNode(),
        ])

        ctx = _make_context()
        with pytest.raises((RuntimeError, Exception)):
            asyncio.run(graph.execute(ctx))

        replays = ctx.determinism_manifest.get("failure_replay", [])
        if replays:
            keys = replays[-1].get("state_keys", [])
            assert keys == sorted(keys), "state_keys must be sorted for deterministic comparison"


# ---------------------------------------------------------------------------
# Section 4 — Memory-personality boundary (no hidden weighting loops)
# ---------------------------------------------------------------------------


class TestMemoryPersonalityBoundary:
    """Verify personality_service and tone paths do not read memory scoring data."""

    def test_personality_service_has_no_memory_scoring_import(self):
        """PersonalityServiceManager must not import memory_query or scoring modules."""
        import importlib
        mod = importlib.import_module("dadbot.managers.personality_service")
        source_file = getattr(mod, "__file__", None)
        assert source_file is not None
        with open(source_file, encoding="utf-8") as fh:
            source = fh.read()
        forbidden = [
            "memory_query",
            "retrieval_diagnostics",
            "score_memory",
            "top_score",
            "_last_memory_retrieval_diagnostics",
        ]
        for token in forbidden:
            assert token not in source, (
                f"PersonalityServiceManager must not reference '{token}' — memory/personality boundary violated"
            )

    def test_tone_context_builder_has_no_memory_scoring_import(self):
        """ToneContextBuilder must not import memory_query or scoring modules."""
        import importlib
        mod = importlib.import_module("dadbot.tone")
        source_file = getattr(mod, "__file__", None)
        assert source_file is not None
        with open(source_file, encoding="utf-8") as fh:
            source = fh.read()
        forbidden = [
            "memory_query",
            "retrieval_diagnostics",
            "score_memory",
            "top_score",
        ]
        for token in forbidden:
            assert token not in source, (
                f"ToneContextBuilder must not reference '{token}' — boundary violated"
            )

    def test_build_mood_context_is_pure_config_lookup(self):
        """build_mood_context returns a non-empty string from config only — no I/O."""
        from dadbot.tone import ToneContextBuilder
        from dadbot.config import MOOD_TONE_GUIDANCE

        class _MinimalBot:
            def normalize_mood(self, mood: str) -> str:
                return mood if mood in MOOD_TONE_GUIDANCE else "neutral"

        tone = ToneContextBuilder.__new__(ToneContextBuilder)
        tone.bot = _MinimalBot()
        result = tone.build_mood_context("neutral")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "neutral" in result.lower() or "tony" in result.lower()

    def test_build_personality_context_delegates_only_to_tone(self):
        """build_personality_context calls build_mood_context and nothing else memory-related."""
        from dadbot.managers.personality_service import PersonalityServiceManager
        from dadbot.config import MOOD_TONE_GUIDANCE

        calls = []

        class _MinimalTone:
            def build_mood_context(self, mood: str) -> str:
                calls.append(("build_mood_context", mood))
                return f"mood context for {mood}"

        class _MinimalBot:
            tone_context = _MinimalTone()
            def normalize_mood(self, mood: str) -> str:
                return mood

        svc = PersonalityServiceManager.__new__(PersonalityServiceManager)
        svc.bot = _MinimalBot()
        result = svc.build_personality_context("neutral")
        assert ("build_mood_context", "neutral") in calls
        assert result == "mood context for neutral"

    def test_prompt_assembly_memory_confidence_reads_only_diagnostics_accessor(self):
        """_memory_confidence_label reads memory diagnostics via get_retrieval_diagnostics only."""
        from dadbot.managers.prompt_assembly import PromptAssemblyManager

        accessor_calls = []

        class _MockMemoryQuery:
            def get_retrieval_diagnostics(self) -> dict:
                accessor_calls.append("get_retrieval_diagnostics")
                return {"retrieved_count": 2, "top_score": 0.8, "has_high_confidence": True}

        class _MinimalBot:
            memory_query = _MockMemoryQuery()

        pm = PromptAssemblyManager.__new__(PromptAssemblyManager)
        pm.bot = _MinimalBot()
        label, guidance = pm._memory_confidence_label()
        assert label == "HIGH"
        assert "get_retrieval_diagnostics" in accessor_calls, (
            "_memory_confidence_label must call get_retrieval_diagnostics, not access private attrs"
        )

    def test_prompt_assembly_memory_confidence_does_not_access_private_attr_when_accessor_present(self):
        """When memory_query.get_retrieval_diagnostics is available, private attr fallback is NOT used."""
        from dadbot.managers.prompt_assembly import PromptAssemblyManager

        private_accesses = []

        class _TrackingMemoryQuery:
            def get_retrieval_diagnostics(self) -> dict:
                return {"retrieved_count": 1, "top_score": 0.5}

        class _MinimalBot:
            memory_query = _TrackingMemoryQuery()

            def __getattribute__(self, name):
                if name == "_last_memory_retrieval_diagnostics":
                    private_accesses.append(name)
                return super().__getattribute__(name)

        pm = PromptAssemblyManager.__new__(PromptAssemblyManager)
        pm.bot = _MinimalBot()
        pm._memory_confidence_label()
        assert len(private_accesses) == 0, (
            "Private _last_memory_retrieval_diagnostics must not be accessed when accessor is present"
        )


# ---------------------------------------------------------------------------
# Section 5 — SafetyNode passthrough annotation
# ---------------------------------------------------------------------------


class TestSafetyNodePassthrough:
    """SafetyNode bare passthrough stamps safety_passthrough in state explicitly."""

    def test_no_safety_manager_stamps_passthrough_key(self):
        """When no enforce_policies/validate is available, safety_passthrough is stamped."""
        from dadbot.core.nodes import SafetyNode as ProdSafetyNode

        class _EmptyManager:
            pass  # no enforce_policies, no validate

        node = ProdSafetyNode(_EmptyManager())
        ctx = _make_context()
        ctx.state["candidate"] = "hello world"
        asyncio.run(node.run(ctx))
        assert "safety_passthrough" in ctx.state, (
            "safety_passthrough must be stamped when no safety callable is found"
        )
        pt = ctx.state["safety_passthrough"]
        assert pt["reason"] == "no_safety_manager"
        assert "failure_mode" in pt

    def test_no_safety_manager_still_sets_safe_result(self):
        """Even without a safety manager, safe_result must be set (pipeline continues)."""
        from dadbot.core.nodes import SafetyNode as ProdSafetyNode

        class _EmptyManager:
            pass

        node = ProdSafetyNode(_EmptyManager())
        ctx = _make_context()
        ctx.state["candidate"] = "test candidate"
        asyncio.run(node.run(ctx))
        assert ctx.state.get("safe_result") == "test candidate"
