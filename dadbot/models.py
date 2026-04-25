from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DadBotModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class MemoryEntry(DadBotModel):
    summary: str
    category: str = "general"
    mood: str = "neutral"
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    impact_score: float = 1.0
    importance_score: float = Field(0.0, ge=0.0, le=1.0)
    emotional_intensity: float = Field(0.25, ge=0.0, le=1.0)
    relationship_impact: float = Field(0.5, ge=0.0, le=1.0)
    pinned: bool = False
    created_at: date | datetime
    updated_at: date | datetime
    contradictions: list[str] = Field(default_factory=list)


class ConsolidatedMemory(DadBotModel):
    summary: str
    category: str = "general"
    source_count: int = Field(1, ge=1)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    importance_score: float = Field(0.0, ge=0.0, le=1.0)
    version: int = Field(1, ge=1)
    superseded: bool = False
    superseded_by: str = ""
    superseded_reason: str = ""
    superseded_at: date | datetime | None = None
    last_reinforced_at: date | datetime | None = None
    supporting_summaries: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    updated_at: date | datetime


class RelationshipState(DadBotModel):
    trust_level: int = Field(50, ge=0, le=100)
    openness_level: int = Field(50, ge=0, le=100)
    emotional_momentum: Literal["heavy", "steady", "warming"] = "steady"
    recurring_topics: dict[str, int] = Field(default_factory=dict)
    recent_checkins: list[dict[str, Any]] = Field(default_factory=list)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    active_hypothesis: str = "supportive_baseline"
    last_hypothesis_updated: date | datetime
    last_reflection: str = ""
    last_updated: date | datetime


class SupervisorJudgment(DadBotModel):
    approved: bool
    score: int = Field(..., ge=1, le=10)
    dad_likeness: int = Field(..., ge=1, le=10)
    groundedness: int = Field(..., ge=1, le=10)
    emotional_fit: int = Field(..., ge=1, le=10)
    issues: list[str] = Field(default_factory=list)
    revised_reply: str | None = None
    stage: str = "reply_supervisor"


class AgenticToolPlan(DadBotModel):
    needs_tool: bool = False
    tool: Literal["set_reminder", "web_search"] | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class PlannerDebugState(DadBotModel):
    updated_at: str | None = None
    user_input: str = ""
    current_mood: str = "neutral"
    planner_status: str = "idle"
    planner_reason: str = ""
    planner_tool: str = ""
    planner_parameters: dict[str, Any] = Field(default_factory=dict)
    planner_observation: str = ""
    fallback_status: str = "idle"
    fallback_reason: str = ""
    fallback_tool: str = ""
    fallback_observation: str = ""
    final_path: str = "idle"


class OutputModerationDecision(DadBotModel):
    approved: bool = True
    action: Literal["allow", "rewrite", "block"] = "allow"
    category: str = "none"
    source: str = "none"
    reason: str = ""
    revised_reply: str = ""


class ModerationSnapshot(DadBotModel):
    enabled: bool = True
    use_llm_classifier: bool = False
    blocked_pattern_count: int = Field(0, ge=0)
    last_decision: OutputModerationDecision = Field(default_factory=OutputModerationDecision)


class SupervisorDecisionState(DadBotModel):
    stage: str = "idle"
    approved: bool = True
    score: int = Field(0, ge=0, le=10)
    dad_likeness: int = Field(0, ge=0, le=10)
    groundedness: int = Field(0, ge=0, le=10)
    emotional_fit: int = Field(0, ge=0, le=10)
    issues: list[str] = Field(default_factory=list)
    revised: bool = False
    duration_ms: int = Field(0, ge=0)
    source: str = "none"


class PersonaTrait(DadBotModel):
    trait: str
    reason: str = ""
    announcement: str = ""
    session_count: int = Field(0, ge=0)
    applied_at: datetime
    last_reinforced_at: datetime | None = None
    strength: float = Field(1.0, ge=0.25, le=3.0)
    impact_score: float = 0.0
    critique_score: int = Field(0, ge=0, le=10)
    critique_feedback: str = ""


class WisdomInsight(DadBotModel):
    summary: str
    topic: str = "general"
    trigger: str = ""
    created_at: date | datetime


class ReminderEntry(DadBotModel):
    id: str
    title: str
    due_text: str = ""
    due_at: datetime | None = None
    status: Literal["open", "done"] = "open"
    created_at: datetime
    updated_at: datetime
    last_notified_at: datetime | None = None
    notification_count: int = Field(0, ge=0)


class SessionArchiveEntry(DadBotModel):
    id: str
    created_at: datetime
    summary: str
    topics: list[str] = Field(default_factory=list)
    dominant_mood: str = "neutral"
    turn_count: int = Field(0, ge=0)


class LifePattern(DadBotModel):
    summary: str
    topic: str = "general"
    mood: str = "neutral"
    day_hint: str = ""
    confidence: int = Field(0, ge=0, le=100)
    last_seen_at: datetime
    proactive_message: str = ""
    last_proactive_at: datetime | None = None


class ProactiveMessage(DadBotModel):
    message: str
    source: str = "general"
    created_at: datetime


class MoodHistoryEntry(DadBotModel):
    mood: str = "neutral"
    date: date | datetime


class ThreadRuntimeSnapshot(DadBotModel):
    history: list[dict[str, Any]] = Field(default_factory=list)
    session_moods: list[str] = Field(default_factory=list)
    session_summary: str = ""
    session_summary_updated_at: str | None = None
    session_summary_covered_messages: int = Field(0, ge=0)
    last_relationship_reflection_turn: int = Field(0, ge=0)
    pending_daily_checkin_context: bool = False
    active_tool_observation_context: str | None = None
    planner_debug: PlannerDebugState = Field(default_factory=PlannerDebugState)
    closed: bool = False


class ChatThreadState(DadBotModel):
    thread_id: str
    title: str
    created_at: str
    updated_at: str
    last_message: str = ""
    turn_count: int = Field(0, ge=0)
    closed: bool = False


class BackgroundTaskRecord(DadBotModel):
    task_id: str
    session_id: str = ""
    task_kind: str = "background"
    status: Literal["queued", "running", "completed", "failed", "unknown"] = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    queued_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    failed_at: str | None = None
    error: str = ""


class BackgroundTaskOverview(DadBotModel):
    tracked: int = Field(0, ge=0)
    queued: int = Field(0, ge=0)
    running: int = Field(0, ge=0)
    completed: int = Field(0, ge=0)
    failed: int = Field(0, ge=0)
    recent: list[BackgroundTaskRecord] = Field(default_factory=list)


class TurnPipelineStep(DadBotModel):
    name: str
    status: Literal["running", "completed", "skipped", "error"] = "completed"
    detail: str = ""
    timestamp: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnPipelineSnapshot(DadBotModel):
    mode: str = "sync"
    user_input: str = ""
    current_mood: str | None = None
    final_path: str = ""
    reply_source: str = ""
    should_end: bool = False
    started_at: str | None = None
    completed_at: str | None = None
    error: str = ""
    steps: list[TurnPipelineStep] = Field(default_factory=list)


class ReplySupervisorSnapshot(DadBotModel):
    enabled: bool = True
    active_hypothesis: str = "supportive_baseline"
    active_hypothesis_label: str = "Supportive Baseline"
    active_hypothesis_probability: float = 0.0
    last_decision: SupervisorDecisionState = Field(default_factory=SupervisorDecisionState)


class StatusTraitMetric(DadBotModel):
    trait: str
    strength: float = 0.0
    impact_score: float = 0.0


class RuntimeStatusSnapshot(DadBotModel):
    active_model: str = ""
    embedding_model: str = "inactive"
    saved_memories: int = Field(0, ge=0)
    archived_chats: int = Field(0, ge=0)
    pending_proactive: int = Field(0, ge=0)
    trust_level: int = Field(0, ge=0, le=100)
    openness_level: int = Field(0, ge=0, le=100)
    emotional_momentum: str = "steady"
    last_mood: str = "neutral"
    tenant_id: str = "default"
    top_trait_metrics: list[StatusTraitMetric] = Field(default_factory=list)
    health: dict[str, Any] = Field(default_factory=dict)


class ServiceStatusSnapshot(DadBotModel):
    base_url: str = ""
    event_stream_url: str = ""
    reachable: bool = False
    port_open: bool = False
    status: str = "offline"
    workers: int = Field(0, ge=0)
    queue_backend: str = ""
    state_backend: str = ""
    service_name: str = ""
    error: str = ""


class PersistenceStatusSnapshot(DadBotModel):
    tenant_id: str = "default"
    backend: str = "filesystem"
    acid_enabled: bool = False
    enabled: bool = False
    primary_store: str = "filesystem"
    profile_backend: str = "json_file"
    memory_backend: str = "json_file"
    json_mirror_enabled: bool = False


class VisionStatusSnapshot(DadBotModel):
    ready: bool = False
    message: str = ""


class SecurityStatusSnapshot(DadBotModel):
    require_pin: bool = False
    has_pin_hint: bool = False


class SessionStatusSnapshot(DadBotModel):
    turn_count: int = Field(0, ge=0)
    history_messages: int = Field(0, ge=0)
    summary_updated_at: str | None = None
    summary_covered_messages: int = Field(0, ge=0)


class ThreadsStatusSnapshot(DadBotModel):
    total: int = Field(0, ge=0)
    open: int = Field(0, ge=0)
    closed: int = Field(0, ge=0)


class ActiveThreadSnapshot(DadBotModel):
    thread_id: str = ""
    title: str = ""
    turn_count: int = Field(0, ge=0)
    closed: bool = False
    last_message: str = ""


class RelationshipStatusSnapshot(DadBotModel):
    trust_level: int = Field(0, ge=0, le=100)
    openness_level: int = Field(0, ge=0, le=100)
    emotional_momentum: str = "steady"
    active_hypothesis: str = "supportive_baseline"
    active_hypothesis_label: str = "Supportive Baseline"
    active_hypothesis_probability: float = 0.0
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    top_topics: list[str] = Field(default_factory=list)


class RuntimeIssueSnapshot(DadBotModel):
    timestamp: str = ""
    purpose: str = "runtime"
    fallback: str = ""
    error: str = ""
    level: str = "warning"
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphFallbackStatusSnapshot(DadBotModel):
    active: bool = False
    degraded_mode: str = "none"
    event_count: int = Field(0, ge=0)
    last_timestamp: str = ""
    last_purpose: str = ""
    last_fallback: str = ""
    last_error: str = ""
    message: str = ""


class PromptGuardStatusSnapshot(DadBotModel):
    trim_count: int = Field(0, ge=0)
    trimmed_tokens_total: int = Field(0, ge=0)
    last_purpose: str = ""
    last_original_tokens: int = Field(0, ge=0)
    last_final_tokens: int = Field(0, ge=0)
    last_trimmed: bool = False
    last_updated: str | None = None


class MemoryContextStatusSnapshot(DadBotModel):
    tokens: int = Field(0, ge=0)
    budget_tokens: int = Field(1, ge=1)
    selected_sections: int = Field(0, ge=0)
    total_sections: int = Field(0, ge=0)
    pruned: bool = False
    last_user_input: str = ""
    last_updated: str | None = None


class RuntimeHealthSnapshot(DadBotModel):
    level: Literal["green", "yellow", "red"] = "green"
    warnings: list[str] = Field(default_factory=list)
    memory_context_ratio: float = Field(0.0, ge=0.0)
    prompt_guard_trim_count: int = Field(0, ge=0)
    recent_runtime_issue_count: int = Field(0, ge=0)
    health_score: int = Field(100, ge=0, le=100)
    reasoning_confidence: float = Field(1.0, ge=0.0, le=1.0)
    projected_minutes_to_red: int = Field(-1, ge=-1)
    background_worker_limit: int = Field(1, ge=1)
    prompt_budget_factor: float = Field(1.0, ge=0.5, le=1.0)
    delayed_noncritical_maintenance: bool = False
    quiet_mode_active: bool = False
    clarification_recommended: bool = False
    clarification_message: str = ""
    optimization_recommended: bool = False
    optimization_applied: bool = False
    updated_at: str | None = None


class RuntimeHealthTrendPoint(DadBotModel):
    recorded_at: str | None = None
    level: Literal["green", "yellow", "red"] = "green"
    memory_context_ratio: float = Field(0.0, ge=0.0)
    prompt_guard_trim_count: int = Field(0, ge=0)
    recent_runtime_issue_count: int = Field(0, ge=0)


class CircuitBreakerStatusSnapshot(DadBotModel):
    active: bool = False
    severity: Literal["low", "medium", "high"] = "low"
    title: str = ""
    message: str = ""
    reasons: list[str] = Field(default_factory=list)
    suggested_prompts: list[str] = Field(default_factory=list)
    reputation_score: int = Field(0, ge=0, le=100)
    current_mood: str = "neutral"
    reasoning_confidence: float = Field(1.0, ge=0.0, le=1.0)


class RelationshipTrendPoint(DadBotModel):
    recorded_at: str | None = None
    trust_level: int = Field(0, ge=0, le=100)
    openness_level: int = Field(0, ge=0, le=100)
    source: str = "turn"


class DashboardStatusSnapshot(DadBotModel):
    status: RuntimeStatusSnapshot
    service: ServiceStatusSnapshot
    persistence: PersistenceStatusSnapshot
    moderation: ModerationSnapshot
    background_tasks: BackgroundTaskOverview
    semantic_memory: dict[str, Any] = Field(default_factory=dict)
    vision: VisionStatusSnapshot = Field(default_factory=VisionStatusSnapshot)
    planner_debug: PlannerDebugState = Field(default_factory=PlannerDebugState)
    runtime: dict[str, Any] = Field(default_factory=dict)
    agentic_tools: dict[str, Any] = Field(default_factory=dict)
    security: SecurityStatusSnapshot = Field(default_factory=SecurityStatusSnapshot)
    session: SessionStatusSnapshot = Field(default_factory=SessionStatusSnapshot)
    threads: ThreadsStatusSnapshot = Field(default_factory=ThreadsStatusSnapshot)
    active_thread: ActiveThreadSnapshot = Field(default_factory=ActiveThreadSnapshot)
    relationship: RelationshipStatusSnapshot = Field(default_factory=RelationshipStatusSnapshot)
    relationship_history: list[RelationshipTrendPoint] = Field(default_factory=list)
    memory_contradictions: list[dict[str, Any]] = Field(default_factory=list)
    memory_context: MemoryContextStatusSnapshot = Field(default_factory=MemoryContextStatusSnapshot)
    prompt_guard: PromptGuardStatusSnapshot = Field(default_factory=PromptGuardStatusSnapshot)
    health: RuntimeHealthSnapshot = Field(default_factory=RuntimeHealthSnapshot)
    circuit_breaker: CircuitBreakerStatusSnapshot = Field(default_factory=CircuitBreakerStatusSnapshot)
    health_history: list[RuntimeHealthTrendPoint] = Field(default_factory=list)
    recent_runtime_issues: list[RuntimeIssueSnapshot] = Field(default_factory=list)
    graph_fallback: GraphFallbackStatusSnapshot = Field(default_factory=GraphFallbackStatusSnapshot)
    maintenance: dict[str, Any] = Field(default_factory=dict)
    supervisor: ReplySupervisorSnapshot = Field(default_factory=ReplySupervisorSnapshot)
    living: dict[str, Any] = Field(default_factory=dict)
    turn_pipeline: TurnPipelineSnapshot | None = None


class MemoryGraphNode(DadBotModel):
    id: str
    label: str
    type: Literal["topic", "category", "mood"]
    weight: int = Field(1, ge=1)


class MemoryGraphEdge(DadBotModel):
    source: str
    target: str
    weight: int = Field(1, ge=1)


class MemoryGraph(DadBotModel):
    nodes: list[MemoryGraphNode] = Field(default_factory=list)
    edges: list[MemoryGraphEdge] = Field(default_factory=list)
    updated_at: str | None = None


class MemoryStore(DadBotModel):
    memories: list[MemoryEntry] = Field(default_factory=list)
    consolidated_memories: list[ConsolidatedMemory] = Field(default_factory=list)
    persona_evolution: list[PersonaTrait] = Field(default_factory=list)
    wisdom_insights: list[WisdomInsight] = Field(default_factory=list)
    life_patterns: list[LifePattern] = Field(default_factory=list)
    pending_proactive_messages: list[ProactiveMessage] = Field(default_factory=list)
    health_history: list[RuntimeHealthTrendPoint] = Field(default_factory=list)
    health_quiet_mode: bool = False
    runtime_optimization: dict[str, Any] = Field(default_factory=dict)
    last_consolidated_at: str | None = None
    last_pattern_detection_at: str | None = None
    last_mood: str = "neutral"
    last_mood_updated_at: date | datetime
    recent_moods: list[MoodHistoryEntry] = Field(default_factory=list)
    relationship_state: RelationshipState
    internal_state: dict[str, Any] = Field(default_factory=dict)
    relationship_history: list[RelationshipTrendPoint] = Field(default_factory=list)
    reminders: list[ReminderEntry] = Field(default_factory=list)
    session_archive: list[SessionArchiveEntry] = Field(default_factory=list)
    last_background_synthesis_at: str | None = None
    last_background_synthesis_turn: int = Field(0, ge=0)
    last_memory_compaction_at: str | None = None
    last_memory_compaction_summary: str = ""
    last_daily_checkin_at: str | None = None
    last_scheduled_proactive_at: str | None = None
    relationship_timeline: str = ""
    memory_graph: MemoryGraph = Field(default_factory=MemoryGraph)
    mcp_local_store: dict[str, Any] = Field(default_factory=dict)
    narrative_memories: list[dict[str, Any]] = Field(default_factory=list)
    heritage_cross_links: list[dict[str, Any]] = Field(default_factory=list)
    advice_audits: list[dict[str, Any]] = Field(default_factory=list)
    environmental_cues_history: list[dict[str, Any]] = Field(default_factory=list)
    longitudinal_insights: list[dict[str, Any]] = Field(default_factory=list)


__all__ = [
    "ActiveThreadSnapshot",
    "AgenticToolPlan",
    "BackgroundTaskRecord",
    "BackgroundTaskOverview",
    "ChatThreadState",
    "CircuitBreakerStatusSnapshot",
    "ConsolidatedMemory",
    "DadBotModel",
    "DashboardStatusSnapshot",
    "LifePattern",
    "MemoryContextStatusSnapshot",
    "MemoryEntry",
    "MemoryGraph",
    "MemoryGraphEdge",
    "MemoryGraphNode",
    "MemoryStore",
    "ModerationSnapshot",
    "MoodHistoryEntry",
    "OutputModerationDecision",
    "PersonaTrait",
    "PersistenceStatusSnapshot",
    "PlannerDebugState",
    "ProactiveMessage",
    "PromptGuardStatusSnapshot",
    "RelationshipStatusSnapshot",
    "ReminderEntry",
    "ReplySupervisorSnapshot",
    "RelationshipState",
    "RuntimeHealthSnapshot",
    "RuntimeHealthTrendPoint",
    "RuntimeStatusSnapshot",
    "RuntimeIssueSnapshot",
    "SecurityStatusSnapshot",
    "ServiceStatusSnapshot",
    "SessionArchiveEntry",
    "SessionStatusSnapshot",
    "StatusTraitMetric",
    "SupervisorDecisionState",
    "SupervisorJudgment",
    "ThreadRuntimeSnapshot",
    "ThreadsStatusSnapshot",
    "TurnPipelineSnapshot",
    "TurnPipelineStep",
    "VisionStatusSnapshot",
    "WisdomInsight",
]
