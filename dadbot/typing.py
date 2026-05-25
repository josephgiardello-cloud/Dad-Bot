from typing import Protocol, runtime_checkable, Any

@runtime_checkable
class MemoryManager(Protocol):
    # Methods
    def store(self, key: str, value: Any) -> None: ...
    def delete(self, key: str) -> None: ...
    def memory_projection(self) -> dict[str, Any]: ...
    # Properties/attributes accessed by DadBot facade
    container: Any
    history: Any
    session_summary: Any
    session_summary_updated_at: Any
    session_summary_covered_messages: Any
    active_tool_observation_context: Any
    planner_debug: Any
    chat_threads: Any
    active_thread_id: Any
    thread_snapshots: Any

@runtime_checkable
class RelationshipManager(Protocol):
    last_relationship_reflection_turn: Any

@runtime_checkable
class MoodManager(Protocol):
    session_moods: Any
    pending_daily_checkin_context: Any

@runtime_checkable
class ProfileRuntime(Protocol):
    profile: Any
    style: Any

@runtime_checkable
class EventBus(Protocol):
    def emit(self, event: dict[str, Any]) -> None: ...
    def peek(self, limit: int = 64) -> list[dict[str, Any]]: ...
    def consume(self, limit: int = 128) -> list[dict[str, Any]]: ...

# Add other manager protocols as needed
