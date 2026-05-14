from __future__ import annotations

import inspect
from typing import Any


class InferenceService:
    async def run_agent(
        self,
        context: Any,
        rich_context: dict[str, Any],
    ) -> Any:  # pragma: no cover - contract only
        raise NotImplementedError


class HealthService:
    def tick(self, context: Any) -> dict[str, Any]:  # pragma: no cover - contract only
        raise NotImplementedError


def _check_required_method(
    instance: Any,
    method_name: str,
    *,
    must_be_async: bool = False,
) -> list[str]:
    issues: list[str] = []
    method = getattr(instance, method_name, None)
    if not callable(method):
        issues.append(f"missing required method: {method_name}")
        return issues
    if must_be_async and not inspect.iscoroutinefunction(method):
        issues.append(f"method {method_name} must be async")
    return issues


def validate_pipeline_services(
    service_contracts: dict[str, tuple[Any, type[Any]]],
    *,
    raise_on_failure: bool = False,
) -> list[str]:
    """Validate basic runtime service contract conformance."""
    issues: list[str] = []

    for service_name, (instance, contract) in service_contracts.items():
        if instance is None:
            issues.append(f"{service_name}: missing service instance")
            continue

        if contract is InferenceService:
            issues.extend(
                [
                    f"{service_name}: {msg}"
                    for msg in _check_required_method(
                        instance,
                        "run_agent",
                        must_be_async=True,
                    )
                ],
            )
        elif contract is HealthService:
            issues.extend(
                [f"{service_name}: {msg}" for msg in _check_required_method(instance, "tick")],
            )

    if issues and raise_on_failure:
        raise RuntimeError("contract violation: " + "; ".join(issues))
    return issues
