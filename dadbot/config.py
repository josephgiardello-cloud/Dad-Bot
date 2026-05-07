from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dadbot.utils import env_truthy
from dadbot_system import ServiceConfig, normalize_tenant_id

MOOD_CATEGORIES = {
    "positive": "happy, excited, proud, energetic, or upbeat",
    "neutral": "calm, neutral, or reflective",
    "stressed": "stressed, anxious, worried, or overwhelmed",
    "sad": "sad, down, disappointed, or low",
    "frustrated": "frustrated, angry, irritated, or annoyed",
    "tired": "tired, exhausted, or drained",
}

MOOD_ALIASES = {
    "happy": "positive",
    "excited": "positive",
    "proud": "positive",
    "upbeat": "positive",
    "relieved": "positive",
    "calm": "neutral",
    "reflective": "neutral",
    "anxious": "stressed",
    "worried": "stressed",
    "overwhelmed": "stressed",
    "pressure": "stressed",
    "pressured": "stressed",
    "down": "sad",
    "lonely": "sad",
    "hurt": "sad",
    "grieving": "sad",
    "angry": "frustrated",
    "annoyed": "frustrated",
    "irritated": "frustrated",
    "resentful": "frustrated",
    "exhausted": "tired",
    "drained": "tired",
    "sleepy": "tired",
    "burned out": "tired",
    "burnt out": "tired",
    "fatigued": "tired",
}

MOOD_TONE_GUIDANCE = {
    "positive": "Be extra warm, proud, and celebratory. Use encouraging language like 'That's my boy!' or 'Atta boy!'.",
    "neutral": "Stay supportive and steady as usual.",
    "stressed": "Be extra gentle, validating, and reassuring. Acknowledge feelings first, then offer calm support.",
    "sad": "Be very empathetic and comforting. Validate emotions and remind him he's not alone.",
    "frustrated": "Stay calm and level-headed. Acknowledge frustration without escalating, then help perspective or solutions.",
    "tired": "Be soft, understanding, and restorative. Suggest rest or lighten the mood gently.",
}

PERSONA_PRESETS = {
    "classic": {
        "label": "Classic Dad",
        "name": "Dad",
        "signoff": "Love you, buddy.",
        "behavior_rules": [
            "Always stay in character as a warm, encouraging, slightly old-school dad.",
            "Use casual language, dad jokes when appropriate, and short paragraphs.",
            "Never be overly formal or robotic.",
            "End most replies with the signoff unless it feels unnatural.",
            "Be honest but gentle - never harsh or dismissive.",
            "Never invent personal facts not present in the profile.",
        ],
    },
    "coach": {
        "label": "Coach Dad",
        "name": "Coach Dad",
        "signoff": "Proud of you, buddy.",
        "behavior_rules": [
            "Sound supportive, focused, and lightly challenging in a caring way.",
            "Help Tony break problems into manageable next steps.",
            "Celebrate effort and follow-through, not just outcomes.",
            "Keep replies grounded, confident, and encouraging.",
            "Never invent personal facts not present in the profile.",
        ],
    },
    "playful": {
        "label": "Playful Dad",
        "name": "Fun Dad",
        "signoff": "Love you, buddy.",
        "behavior_rules": [
            "Keep the tone warm, lively, and lightly teasing when appropriate.",
            "Use more playful phrasing or gentle jokes without becoming flippant.",
            "Stay emotionally attentive when Tony sounds low or stressed.",
            "Keep replies concise and natural.",
            "Never invent personal facts not present in the profile.",
        ],
    },
}


@dataclass
class DadRuntimeConfig:
    recent_history_window: int = 8
    max_history_messages_scan: int = 24
    summary_trigger_messages: int = 12
    relationship_reflection_interval: int = 4
    context_token_budget: int = 6000
    reserved_response_tokens: int = 900
    approx_chars_per_token: int = 4
    mood_detection_temperature: float = 0.0
    stream_timeout_seconds: int = 60
    stream_max_chars: int = 12000
    preferred_embedding_models: tuple[str, ...] = (
        "bge-m3",
        "nomic-embed-text",
        "mxbai-embed-large",
        "snowflake-arctic-embed",
    )
    mood_fastpath_confidence_threshold: int = 2
    memory_freshness_half_life_days: int = 30
    memory_min_freshness_weight: float = 0.2
    graph_refresh_debounce_seconds: int = 30
    preferred_vision_model_hints: tuple[str, ...] = (
        "qwen2.5-vl",
        "qwen2-vl",
        "llava",
        "minicpm-v",
        "moondream",
        "bakllava",
        "gemma3",
    )
    semantic_candidate_multiplier: int = 12
    semantic_candidate_minimum: int = 24
    graph_context_token_budget: int = 800
    graph_walk_hops: int = 2
    graph_walk_edge_limit: int = 18
    graph_walk_node_limit: int = 16
    egress_allowlist: tuple[str, ...] = (
        "localhost",
        "127.0.0.1",
        "api.duckduckgo.com",
    )
    egress_enforce: bool = False
    merkle_anchor_enabled: bool = True
    dual_control_enabled: bool = False
    dual_control_approvals_required: int = 2
    rtbf_proof_enabled: bool = True
    goal_alignment_guard_enabled: bool = True
    primary_identity_log_filenames: tuple[str, ...] = ("relational_ledger.jsonl",)

    cadence_defaults: dict[str, int] = field(
        default_factory=lambda: {
            "persona_evolution_min_sessions": 10,
            "persona_evolution_session_gap": 10,
            "wisdom_min_archived_sessions": 2,
            "wisdom_turn_interval": 3,
            "family_echo_turn_interval": 4,
            "life_pattern_min_archived_sessions": 4,
            "life_pattern_window": 12,
            "life_pattern_min_occurrences": 3,
            "life_pattern_confidence_threshold": 70,
            "life_pattern_queue_limit": 2,
            "deep_pattern_min_occurrences": 3,
            "scheduled_reminder_lead_minutes": 45,
            "scheduled_reminder_repeat_hours": 12,
            "scheduled_pattern_hour": 8,
            "scheduled_pattern_min_confidence": 80,
        },
    )

    store_limits: dict[str, int] = field(
        default_factory=lambda: {
            "consolidated_memories": 8,
            "persona_evolution": 8,
            "wisdom_insights": 16,
            "life_patterns": 12,
            "pending_proactive_messages": 8,
            "health_history": 240,
            "reminders": 50,
            "session_archive": 24,
            "recent_checkins": 24,
            "relationship_history": 180,
        },
    )

    lookbacks: dict[str, int] = field(
        default_factory=lambda: {
            "mood_detection_context": 6,
            "recent_mood_history": 12,
            "recent_wisdom_dedup": 6,
            "supporting_summaries": 4,
            "contradictions": 4,
            "pending_proactive_dedup": 3,
            "semantic_query_tokens": 6,
            "deep_pattern_archive_window": 18,
            "deep_pattern_candidates": 14,
            "deep_pattern_context_limit": 3,
        },
    )

    def tail(self, items: Iterable[Any], limit_key: str):
        values = list(items)
        limit = max(0, int(self.store_limits.get(limit_key, 0)))
        return values[-limit:] if limit else values

    def window(self, key: str, fallback: int = 0):
        return max(0, int(self.lookbacks.get(key, fallback)))

    # -- Path resolution ----------------------------------------------------------
    # All file paths are computed from env vars (highest priority) or the project
    # root inferred from this config module's location.

    @staticmethod
    def _project_root() -> Path:
        # config.py lives at dadbot/config.py; project root is one level up.
        return Path(__file__).resolve().parent.parent

    @staticmethod
    def _env_path(env_var: str, fallback: Path) -> Path:
        configured = str(os.environ.get(env_var) or "").strip()
        return Path(configured) if configured else fallback

    @property
    def profile_path(self) -> Path:
        return self._env_path(
            "DADBOT_PROFILE_PATH",
            self._project_root() / "dad_profile.json",
        )

    @property
    def memory_path(self) -> Path:
        return self._env_path(
            "DADBOT_MEMORY_PATH",
            self._project_root() / "dad_memory.json",
        )

    @property
    def semantic_memory_db_path(self) -> Path:
        return self._env_path(
            "DADBOT_SEMANTIC_DB_PATH",
            self._project_root() / "dad_memory_semantic.sqlite3",
        )

    @property
    def graph_store_db_path(self) -> Path:
        return self._env_path(
            "DADBOT_GRAPH_DB_PATH",
            self._project_root() / "dad_memory_graph.sqlite3",
        )

    @property
    def session_log_dir(self) -> Path:
        return self._env_path(
            "DADBOT_SESSION_LOG_DIR",
            self._project_root() / "session_logs",
        )


@dataclass
class DadBotConfig:
    model_name: str = "llama3.2"
    fallback_models: tuple[str, ...] = (
        "llama3.1:8b",
        "phi4:mini",
        "qwen3.5:4b",
        "gemma3:4b",
        "gemma2:2b",
        "llama3.2:1b",
    )
    append_signoff: bool = True
    light_mode: bool = False
    strict_graph_mode: bool = True
    tenant_id: str = ""
    runtime_config: DadRuntimeConfig = field(default_factory=DadRuntimeConfig)
    service_config: ServiceConfig = field(
        default_factory=ServiceConfig.from_environment,
    )

    def __post_init__(self) -> None:
        self.tenant_id = normalize_tenant_id(
            self.tenant_id or os.environ.get("DADBOT_TENANT_ID") or "",
        )
        self.append_signoff = bool(
            self.append_signoff and not env_truthy("DADBOT_NO_SIGNOFF", default=False),
        )
        self.light_mode = bool(
            self.light_mode or env_truthy("DADBOT_LIGHT_MODE", default=False),
        )
        self.strict_graph_mode = bool(
            self.strict_graph_mode and env_truthy("DADBOT_STRICT_GRAPH_MODE", default=True),
        )
        self.active_model = str(self.model_name).strip() or "llama3.2"
        self.active_embedding_model = None
        self.llm_provider = str(os.environ.get("DADBOT_LLM_PROVIDER", "ollama")).strip().lower() or "ollama"
        self.llm_model = str(os.environ.get("DADBOT_LLM_MODEL", self.active_model)).strip() or self.active_model
        self.preferred_embedding_models = tuple(
            self.runtime_config.preferred_embedding_models,
        )
        self.recent_history_window = self.runtime_config.recent_history_window
        self.max_history_messages_scan = self.runtime_config.max_history_messages_scan
        self.summary_trigger_messages = self.runtime_config.summary_trigger_messages
        self.relationship_reflection_interval = self.runtime_config.relationship_reflection_interval
        self.context_token_budget = self.runtime_config.context_token_budget
        self.reserved_response_tokens = self.runtime_config.reserved_response_tokens
        self.approx_chars_per_token = self.runtime_config.approx_chars_per_token
        self.mood_detection_temperature = self.runtime_config.mood_detection_temperature
        self.stream_timeout_seconds = self.runtime_config.stream_timeout_seconds
        self.stream_max_chars = self.runtime_config.stream_max_chars
        self.profile_path = self.runtime_config.profile_path
        self.memory_path = self.runtime_config.memory_path
        self.semantic_memory_db_path = self.runtime_config.semantic_memory_db_path
        self.graph_store_db_path = self.runtime_config.graph_store_db_path
        self.session_log_dir = self.runtime_config.session_log_dir
        self.health_snapshot_interval_seconds = 300
        self.proactive_heartbeat_interval_seconds = max(
            300,
            int(os.environ.get("DADBOT_PROACTIVE_HEARTBEAT_SECONDS") or 3600),
        )
        self.turn_graph_enabled = bool(
            env_truthy("DADBOT_ENABLE_TURN_GRAPH", default=True) and ("PYTEST_CURRENT_TEST" not in os.environ),
        )
        self.turn_graph_config_path = (
            str(
                os.environ.get("DADBOT_TURN_GRAPH_CONFIG_PATH") or "config.yaml",
            ).strip()
            or "config.yaml"
        )
        self.egress_enforce = bool(
            self.runtime_config.egress_enforce or env_truthy("DADBOT_EGRESS_ENFORCE", default=False),
        )
        configured_allowlist = str(
            os.environ.get("DADBOT_EGRESS_ALLOWLIST", ""),
        ).strip()
        if configured_allowlist:
            hosts = [item.strip() for item in configured_allowlist.split(",")]
            self.egress_allowlist = tuple(host for host in hosts if host)
        else:
            self.egress_allowlist = tuple(self.runtime_config.egress_allowlist)
        self.merkle_anchor_enabled = bool(
            self.runtime_config.merkle_anchor_enabled and env_truthy("DADBOT_MERKLE_ANCHOR_ENABLED", default=True),
        )
        self.dual_control_enabled = bool(
            self.runtime_config.dual_control_enabled or env_truthy("DADBOT_DUAL_CONTROL_ENABLED", default=False),
        )
        self.dual_control_approvals_required = max(
            1,
            int(
                os.environ.get(
                    "DADBOT_DUAL_CONTROL_APPROVALS",
                    self.runtime_config.dual_control_approvals_required,
                )
                or 1,
            ),
        )
        self.rtbf_proof_enabled = bool(
            self.runtime_config.rtbf_proof_enabled and env_truthy("DADBOT_RTBF_PROOF_ENABLED", default=True),
        )
        self.goal_alignment_guard_enabled = bool(
            self.runtime_config.goal_alignment_guard_enabled
            and env_truthy("DADBOT_GOAL_ALIGNMENT_GUARD_ENABLED", default=True),
        )
        self.primary_identity_log_filenames = tuple(self.runtime_config.primary_identity_log_filenames)
        self._approval_workflow = None

    def apply_profile_llm_settings(self, provider: str, model: str) -> None:
        normalized_provider = str(provider or self.llm_provider).strip().lower() or self.llm_provider
        normalized_model = str(model or self.llm_model).strip() or self.llm_model
        self.llm_provider = normalized_provider
        self.llm_model = normalized_model
        if self.llm_provider == "ollama" and self.llm_model:
            self.active_model = self.llm_model

    def _config_approval_workflow(self):
        if self._approval_workflow is not None:
            return self._approval_workflow
        from dadbot.core.config_approval import ConfigApprovalWorkflow

        store_path = self.session_log_dir / "config_approvals.json"
        self._approval_workflow = ConfigApprovalWorkflow(
            store_path,
            approvals_required=self.dual_control_approvals_required,
        )
        return self._approval_workflow

    def propose_profile_llm_change(
        self,
        provider: str,
        model: str,
        requested_by: str = "system",
    ) -> str:
        workflow = self._config_approval_workflow()
        proposal = workflow.propose(
            key="llm.profile",
            requested_value={
                "provider": str(provider or "").strip().lower(),
                "model": str(model or "").strip(),
            },
            requested_by=requested_by,
        )
        return proposal.proposal_id

    def approve_profile_llm_change(self, proposal_id: str, approver: str) -> bool:
        workflow = self._config_approval_workflow()
        proposal = workflow.approve(proposal_id, approver)
        if proposal is None:
            return False
        return str(proposal.status or "") in {"approved", "applied"}

    def apply_profile_llm_change_if_approved(self, proposal_id: str) -> bool:
        workflow = self._config_approval_workflow()
        proposal = workflow.consume_approved(proposal_id)
        if proposal is None:
            return False
        value = dict(proposal.requested_value or {})
        self.apply_profile_llm_settings(
            str(value.get("provider") or "").strip().lower(),
            str(value.get("model") or "").strip(),
        )
        return True
