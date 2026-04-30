from __future__ import annotations

from copy import deepcopy

from .models import Event


class ConversationStore:
    """Deterministic thread state for replay-friendly chat history.

    This store is a passive projection over immutable events. It must not be
    used as an execution hook surface.
    """

    THREAD_VIEW_V1 = "v1"
    THREAD_VIEW_V2 = "v2"
    THREAD_VIEW_DEFAULT_VERSION = THREAD_VIEW_V2
    THREAD_VIEW_LATEST_VERSION = THREAD_VIEW_V2
    THREAD_VIEW_SUPPORTED_VERSIONS = (
        THREAD_VIEW_V1,
        THREAD_VIEW_V2,
    )

    def __init__(self) -> None:
        self.threads: dict[str, list[dict]] = {}
        self.thread_state: dict[str, dict] = {}

    def get_thread(self, thread_id: str) -> list[dict]:
        key = str(thread_id or "default")
        return self.threads.setdefault(key, [])

    def get_thread_state(self, thread_id: str) -> dict:
        key = str(thread_id or "default")
        return self.thread_state.setdefault(
            key,
            {
                "last_turn_thinking": {},
                "execution_boundaries": [],
            },
        )

    def seed_thread_messages(self, thread_id: str, messages: list[dict] | None) -> None:
        if self.get_thread(thread_id):
            return
        if not messages:
            return
        self.threads[str(thread_id or "default")] = [
            {
                "role": str(item.get("role") or "assistant"),
                "content": str(item.get("content") or ""),
                "attachments": list(item.get("attachments") or []),
            }
            for item in list(messages)
            if isinstance(item, dict)
        ]

    def append_user(
        self,
        thread_id: str,
        text: str,
        *,
        attachments: list[dict] | None = None,
    ) -> None:
        self.get_thread(thread_id).append(
            {
                "role": "user",
                "content": str(text or ""),
                "attachments": list(attachments or []),
            },
        )

    def append_assistant(
        self,
        thread_id: str,
        text: str,
        *,
        attachments: list[dict] | None = None,
    ) -> None:
        self.get_thread(thread_id).append(
            {
                "role": "assistant",
                "content": str(text or ""),
                "attachments": list(attachments or []),
            },
        )

    def append_to_last_assistant(self, thread_id: str, *, attachment: dict) -> None:
        thread = self.get_thread(thread_id)
        for message in reversed(thread):
            if str(message.get("role") or "") == "assistant":
                attachments = list(message.get("attachments") or [])
                attachments.append(dict(attachment or {}))
                message["attachments"] = attachments
                return

    def update_thinking(self, thread_id: str, thinking: dict) -> None:
        state = self.get_thread_state(thread_id)
        state["last_turn_thinking"] = dict(thinking or {})

    def append_execution_boundary(self, thread_id: str, payload: dict) -> None:
        state = self.get_thread_state(thread_id)
        boundaries = list(state.get("execution_boundaries") or [])
        boundaries.append(deepcopy(dict(payload or {})))
        state["execution_boundaries"] = boundaries

    def apply_event(self, event: Event) -> None:
        if event.type == "user_message":
            self.append_user(
                event.thread_id,
                str(event.payload.get("text") or ""),
                attachments=list(event.payload.get("attachments") or []),
            )
            return
        if event.type == "assistant_reply":
            self.append_assistant(
                event.thread_id,
                str(event.payload.get("text") or ""),
                attachments=list(event.payload.get("attachments") or []),
            )
            return
        if event.type == "assistant_attachment_added":
            self.append_to_last_assistant(
                event.thread_id,
                attachment=dict(event.payload.get("attachment") or {}),
            )
            return
        if event.type == "thinking_update":
            self.update_thinking(event.thread_id, dict(event.payload or {}))
            return
        if event.type in {"execution_region_started", "execution_region_completed"}:
            self.append_execution_boundary(
                event.thread_id,
                {
                    "type": event.type,
                    "thread_id": event.thread_id,
                    "payload": dict(event.payload or {}),
                },
            )

    def thread_messages(self, thread_id: str) -> list[dict]:
        return deepcopy(self.get_thread(thread_id))

    def thread_thinking(self, thread_id: str) -> dict:
        state = self.get_thread_state(thread_id)
        return deepcopy(state.get("last_turn_thinking") or {})

    def thread_execution_boundaries(self, thread_id: str) -> list[dict]:
        """Return passive execution-boundary projections for observers."""
        state = self.get_thread_state(thread_id)
        return deepcopy(state.get("execution_boundaries") or [])

    @classmethod
    def thread_view_schema_policy(cls) -> dict:
        return {
            "default_version": cls.THREAD_VIEW_DEFAULT_VERSION,
            "latest_version": cls.THREAD_VIEW_LATEST_VERSION,
            "live_version": cls.THREAD_VIEW_V2,
            "supported_versions": list(cls.THREAD_VIEW_SUPPORTED_VERSIONS),
            "semantics_window": {
                "status": "frozen",
                "rules": [
                    "no_new_v2_fields_unless_strictly_necessary",
                    "no_semantic_reinterpretation_of_existing_fields",
                    "only_additive_metadata_allowed",
                ],
            },
            "evolution_rule": "additive_only",
            "allowed_changes": [
                "field_additions",
                "optional_metadata_enrichment",
                "derived_fields_without_v1_output_changes",
            ],
            "forbidden_changes": [
                "remove_v1_fields",
                "reorder_v1_structures",
                "reinterpret_v1_values",
            ],
            "version_roles": {
                cls.THREAD_VIEW_V1: "replay_compatibility_contract",
                cls.THREAD_VIEW_V2: "current_live_projection_contract",
            },
            "compatibility_commitment": {
                cls.THREAD_VIEW_V1: "immutable_replay_audit_contract",
                cls.THREAD_VIEW_V2: "must_remain_additive_relative_to_v1",
            },
            "historical_reconstruction": {
                cls.THREAD_VIEW_V1: "deterministic_historical_reconstruction",
                cls.THREAD_VIEW_V2: "live_projection_with_additive_metadata",
            },
            "version_lifecycle": {
                cls.THREAD_VIEW_V1: {
                    "change_policy": "never_changes",
                    "roles": [
                        "replay_contract",
                        "audit_contract",
                        "deterministic_historical_reconstruction",
                    ],
                },
                cls.THREAD_VIEW_V2: {
                    "change_policy": "additive_metadata_only_while_semantics_window_frozen",
                    "compatible_additions": [
                        "field_additions",
                        "optional_metadata_enrichment",
                        "derived_fields_without_v1_output_changes",
                    ],
                    "next_major_version_trigger": [
                        "semantic_reinterpretation_of_existing_fields",
                        "removal_of_existing_fields",
                        "reordering_of_existing_structures",
                        "breaking_shape_change",
                    ],
                },
            },
            "consumer_rule": {
                "boundary_package": "dadbot/consumers",
                "consumer_scopes": ["streamlit", "api_clients", "analytics"],
                "read_from": ["event_api.py", "store.py"],
                "output": "derived_data_only",
                "forbidden": [
                    "runtime.py_imports",
                    "execution_phase_awareness",
                    "state_mutation_paths",
                    "raw_event_log_access",
                    "local_projection_reconstruction",
                ],
            },
        }

    def _thread_view_v1(self, thread_id: str) -> dict:
        normalized = str(thread_id or "default")
        return {
            "thread_id": normalized,
            "messages": self.thread_messages(normalized),
            "thinking": self.thread_thinking(normalized),
            "execution_boundaries": self.thread_execution_boundaries(normalized),
        }

    def _thread_view_v2(self, thread_id: str) -> dict:
        view = self._thread_view_v1(thread_id)
        view["view_version"] = self.THREAD_VIEW_V2
        view["schema_policy"] = self.thread_view_schema_policy()
        return view

    def thread_view(
        self,
        thread_id: str,
        *,
        version: str = THREAD_VIEW_DEFAULT_VERSION,
    ) -> dict:
        if version == self.THREAD_VIEW_V1:
            return self._thread_view_v1(thread_id)
        if version == self.THREAD_VIEW_V2:
            return self._thread_view_v2(thread_id)
        raise ValueError(f"Unsupported thread view version: {version}")
