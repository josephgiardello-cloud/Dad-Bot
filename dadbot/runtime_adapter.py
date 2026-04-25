from __future__ import annotations

from typing import Protocol


class AppRuntimeContract(Protocol):
    """Minimum runtime class contract expected by dadbot.app_runtime."""

    @classmethod
    def initialize_profile_file(cls, profile_path=None, force=False):
        ...

    @staticmethod
    def default_planner_debug_state():
        ...

    def __init__(
        self,
        model_name: str = "llama3.2",
        *,
        append_signoff: bool = True,
        light_mode: bool = False,
        tenant_id: str = "",
    ) -> None:
        ...

    def clear_memory_store(self):
        ...

    def export_memory_store(self, export_path):
        ...

    def print_system_message(self, message):
        ...

    def chat_loop(self):
        ...

    def chat_loop_via_service(self, service_client, session_id=None):
        ...


_REQUIRED_RUNTIME_ATTRIBUTES = (
    "initialize_profile_file",
    "default_planner_debug_state",
    "clear_memory_store",
    "export_memory_store",
    "print_system_message",
    "chat_loop",
    "chat_loop_via_service",
)


def runtime_contract_errors(runtime_cls) -> list[str]:
    """Return app-runtime contract errors for the provided runtime class.

    This is a lightweight runtime guard and release scaffold for a future
    orchestrator handover adapter.
    """
    issues: list[str] = []
    if runtime_cls is None:
        return ["runtime class is None"]

    for attr_name in _REQUIRED_RUNTIME_ATTRIBUTES:
        attr_value = getattr(runtime_cls, attr_name, None)
        if attr_value is None:
            issues.append(f"missing attribute: {attr_name}")
            continue
        if not callable(attr_value):
            issues.append(f"attribute is not callable: {attr_name}")

    return issues


__all__ = ["AppRuntimeContract", "runtime_contract_errors"]
