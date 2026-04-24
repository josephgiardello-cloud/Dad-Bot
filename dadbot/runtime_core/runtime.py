from __future__ import annotations

import copy
import hashlib
import json

from .bus import EventBus
from .models import Event, new_event
from .policy import PolicyEngine
from .services import RuntimeServices
from .store import ConversationStore


class AgentRuntime:
    """Deterministic event loop. No Streamlit imports.

    Execution boundary events are observational outputs only. Runtime correctness
    must not depend on downstream observers, telemetry, or logging hooks.
    """

    _EXECUTION_STRUCTURAL_STEP_FIELDS = (
        "id",
        "name",
        "step",
        "tool_name",
        "kind",
        "depends_on",
    )
    _EXECUTION_SEMANTIC_STEP_FIELDS = (
        "status",
        "result",
        "output",
        "output_hash",
        "execution_trace",
        "diagnostic",
        "metadata",
    )

    def __init__(self, services: RuntimeServices, store: ConversationStore, *, policy_engine: PolicyEngine | None = None) -> None:
        self.services = services
        self.store = store
        self.policy_engine = policy_engine or PolicyEngine()

    def handle_event(self, event: Event) -> list[Event]:
        if event.type == "user_message":
            self.store.apply_event(event)
            return self._handle_user_message(event)

        if event.type in {
            "assistant_reply",
            "assistant_attachment_added",
            "thinking_update",
            "execution_region_started",
            "execution_region_completed",
        }:
            self.store.apply_event(event)
            return []

        if event.type == "memory_write":
            self.services.write_memory(thread_id=event.thread_id, payload=dict(event.payload or {}))
            return []

        if event.type in {"tool_result", "tool_call", "thread_switch", "photo_request", "tts_request", "mood_update"}:
            return []

        return []

    @classmethod
    def _normalize_execution_step(cls, raw_step) -> dict:
        if isinstance(raw_step, dict):
            step = dict(raw_step or {})
            normalized = {}
            for field in cls._EXECUTION_STRUCTURAL_STEP_FIELDS:
                if field == "step":
                    normalized[field] = str(step.get("step") or step.get("name") or "")
                elif field == "tool_name":
                    normalized[field] = str(step.get("tool_name") or step.get("tool") or "")
                elif field == "kind":
                    normalized[field] = str(step.get("kind") or "reasoning")
                elif field == "depends_on":
                    normalized[field] = tuple(str(dep) for dep in list(step.get("depends_on") or []))
                else:
                    normalized[field] = str(step.get(field) or "")
            return normalized
        value = str(raw_step or "")
        return {
            "id": "",
            "name": value,
            "step": value,
            "tool_name": "",
            "kind": "reasoning",
            "depends_on": (),
        }

    @classmethod
    def _execution_structure_signature(cls, pipeline: dict) -> str:
        normalized_steps = [
            cls._normalize_execution_step(step)
            for step in list(dict(pipeline or {}).get("steps") or [])
        ]
        payload = json.dumps(normalized_steps, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _execution_structure_snapshot(cls, pipeline: dict) -> list[dict]:
        return [
            cls._normalize_execution_step(step)
            for step in list(dict(pipeline or {}).get("steps") or [])
        ]

    @classmethod
    def _execution_semantic_snapshot(cls, pipeline: dict) -> dict:
        pipeline_dict = dict(pipeline or {})
        semantic_pipeline = {
            key: copy.deepcopy(value)
            for key, value in pipeline_dict.items()
            if str(key) != "steps"
        }
        semantic_steps = []
        for raw_step in list(pipeline_dict.get("steps") or []):
            if not isinstance(raw_step, dict):
                semantic_steps.append({})
                continue
            semantic_steps.append(
                {
                    field: copy.deepcopy(raw_step[field])
                    for field in cls._EXECUTION_SEMANTIC_STEP_FIELDS
                    if field in raw_step
                }
            )
        return {
            "pipeline": semantic_pipeline,
            "steps": semantic_steps,
        }

    def _execution_boundary_started_event(self, *, event: Event, pipeline: dict) -> Event:
        return new_event(
            "execution_region_started",
            thread_id=event.thread_id,
            payload={
                "structural_fields": list(self._EXECUTION_STRUCTURAL_STEP_FIELDS),
                "semantic_fields": list(self._EXECUTION_SEMANTIC_STEP_FIELDS),
                "structural_signature": self._execution_structure_signature(pipeline),
                "structural_snapshot": self._execution_structure_snapshot(pipeline),
                "semantic_snapshot": self._execution_semantic_snapshot(pipeline),
            },
        )

    def _execution_boundary_completed_event(self, *, event: Event, before_pipeline: dict, after_pipeline: dict) -> Event:
        return new_event(
            "execution_region_completed",
            thread_id=event.thread_id,
            payload={
                "structural_fields": list(self._EXECUTION_STRUCTURAL_STEP_FIELDS),
                "semantic_fields": list(self._EXECUTION_SEMANTIC_STEP_FIELDS),
                "structural_signature_before": self._execution_structure_signature(before_pipeline),
                "structural_signature_after": self._execution_structure_signature(after_pipeline),
                "structural_snapshot": self._execution_structure_snapshot(after_pipeline),
                "semantic_snapshot": self._execution_semantic_snapshot(after_pipeline),
            },
        )

    def _after_execution_region(self, *, event: Event, result):
        _ = event
        return result

    def _run_execution_region(self, *, event: Event):
        result = self.services.handle_user_message(
            thread_id=event.thread_id,
            text=str(event.payload.get("text") or ""),
            attachments=list(event.payload.get("attachments") or []),
        )
        before_pipeline = copy.deepcopy(dict(result.pipeline or {}))
        started_event = self._execution_boundary_started_event(event=event, pipeline=before_pipeline)
        post_result = self._after_execution_region(event=event, result=result)
        after_pipeline = copy.deepcopy(dict(post_result.pipeline or {}))
        if self._execution_structure_signature(before_pipeline) != self._execution_structure_signature(after_pipeline):
            raise RuntimeError(
                "Execution boundary violation: execution region mutated structural pipeline"
            )
        completed_event = self._execution_boundary_completed_event(
            event=event,
            before_pipeline=before_pipeline,
            after_pipeline=after_pipeline,
        )
        # Boundary events are derived after local execution and returned as data;
        # no observer participates in execution correctness.
        return post_result, [started_event, completed_event]

    def _handle_user_message(self, event: Event) -> list[Event]:
        result, execution_boundary_events = self._run_execution_region(event=event)
        
        # Evaluate policies for this turn
        policies = self.policy_engine.evaluate(
            mood=str(result.mood or "neutral"),
            thread_id=event.thread_id,
            reply_text=str(result.reply or ""),
        )
        
        pipeline = copy.deepcopy(dict(result.pipeline or {}))

        events: list[Event] = list(execution_boundary_events) + [
            new_event(
                "assistant_reply",
                thread_id=event.thread_id,
                payload={
                    "text": result.reply,
                    "should_end": bool(result.should_end),
                    "mood": str(result.mood or "neutral"),
                    "pipeline": pipeline,
                    "attachments": [],
                },
            )
        ,
            new_event(
                "thinking_update",
                thread_id=event.thread_id,
                payload={
                    "mood_detected": str(pipeline.get("current_mood") or result.mood or "neutral"),
                    "final_path": str(pipeline.get("final_path") or "model_reply"),
                    "reply_source": str(pipeline.get("reply_source") or "model_generation"),
                    "pipeline_steps": list(pipeline.get("steps") or []),
                    "active_rules": list(result.active_rules or []),
                },
            )
        ]
        
        # Photo request based on policy
        if policies.should_generate_photo:
            events.append(
                new_event(
                    "photo_request",
                    thread_id=event.thread_id,
                    payload={"reason": "mood_support", "mood": result.mood},
                )
            )
        
        # TTS request based on policy
        if policies.should_request_tts:
            events.append(
                new_event(
                    "tts_request",
                    thread_id=event.thread_id,
                    payload={"text": str(result.reply or "")},
                )
            )
        
        return events

    def run_until_idle(self, bus: EventBus, *, max_events: int = 256) -> list[Event]:
        processed: list[Event] = []
        budget = max(1, int(max_events or 1))
        while budget > 0 and not bus.empty():
            budget -= 1
            event = bus.next()
            processed.append(event)
            follow_up = self.handle_event(event)
            for created in follow_up:
                bus.emit(created)
        return processed
