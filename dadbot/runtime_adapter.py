from __future__ import annotations

import inspect
from typing import Protocol

from dadbot.core.execution_boundary import (
    canonical_execution_kernel,
    enforce_execution_role,
)

EXECUTION_ROLE = "disabled"


def assert_runtime_adapter_disabled() -> None:
    enforce_execution_role(module="dadbot.runtime_adapter", role=EXECUTION_ROLE)


def canonical_runtime_entrypoint() -> str:
    """Deprecated shim helper for callers migrating to the kernel entrypoint."""
    return canonical_execution_kernel()


class AppRuntimeContract(Protocol):
    """Minimum runtime class contract expected by dadbot.app_runtime."""

    @classmethod
    def initialize_profile_file(cls, profile_path=None, force=False): ...

    @staticmethod
    def default_planner_debug_state(): ...

    def __init__(
        self,
        model_name: str = "llama3.2",
        *,
        append_signoff: bool = True,
        light_mode: bool = False,
        tenant_id: str = "",
    ) -> None: ...

    def clear_memory_store(self): ...

    def export_memory_store(self, export_path): ...

    def print_system_message(self, message): ...

    def chat_loop(self): ...

    def chat_loop_via_service(self, service_client, session_id=None): ...


_REQUIRED_RUNTIME_ATTRIBUTES = (
    "initialize_profile_file",
    "default_planner_debug_state",
    "clear_memory_store",
    "export_memory_store",
    "print_system_message",
    "chat_loop",
    "chat_loop_via_service",
)

_REQUIRED_DESCRIPTORS = {
    "initialize_profile_file": classmethod,
    "default_planner_debug_state": staticmethod,
}

_REQUIRED_PARAMETERS = {
    "export_memory_store": ("export_path",),
    "print_system_message": ("message",),
    "chat_loop_via_service": ("service_client",),
}

_REQUIRED_INIT_PARAMETERS = (
    "model_name",
    "append_signoff",
    "light_mode",
    "tenant_id",
)


def _validate_descriptor(
    runtime_cls,
    attr_name: str,
    expected_descriptor: type,
) -> list[str]:
    issues: list[str] = []
    descriptor = inspect.getattr_static(runtime_cls, attr_name, None)
    if not isinstance(descriptor, expected_descriptor):
        issues.append(
            f"attribute has wrong descriptor: {attr_name} (expected {expected_descriptor.__name__})",
        )
    return issues


def _validate_required_parameters(
    target,
    *,
    attr_name: str,
    required_params: tuple[str, ...],
) -> list[str]:
    issues: list[str] = []
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return issues

    parameter_names = tuple(signature.parameters.keys())
    for required_name in required_params:
        if required_name not in parameter_names:
            issues.append(f"attribute missing parameter: {attr_name}.{required_name}")
    return issues


def _validate_init_signature(runtime_cls) -> list[str]:
    if "__init__" not in getattr(runtime_cls, "__dict__", {}):
        return []

    issues: list[str] = []
    init_fn = getattr(runtime_cls, "__init__", None)
    if init_fn is None:
        return ["missing attribute: __init__"]
    if not callable(init_fn):
        return ["attribute is not callable: __init__"]

    try:
        signature = inspect.signature(init_fn)
    except (TypeError, ValueError):
        return issues

    parameter_names = tuple(signature.parameters.keys())
    for required_name in _REQUIRED_INIT_PARAMETERS:
        if required_name not in parameter_names:
            issues.append(f"attribute missing parameter: __init__.{required_name}")
    return issues


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

        expected_descriptor = _REQUIRED_DESCRIPTORS.get(attr_name)
        if expected_descriptor is not None:
            issues.extend(
                _validate_descriptor(runtime_cls, attr_name, expected_descriptor),
            )

        required_params = _REQUIRED_PARAMETERS.get(attr_name)
        if required_params:
            issues.extend(
                _validate_required_parameters(
                    attr_value,
                    attr_name=attr_name,
                    required_params=required_params,
                ),
            )

    issues.extend(_validate_init_signature(runtime_cls))

    return sorted(set(issues))


__all__ = ["AppRuntimeContract", "runtime_contract_errors"]
