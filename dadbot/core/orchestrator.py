from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json as _json
import logging
import os
import platform
from importlib import metadata as importlib_metadata
from typing import Any, cast

from dadbot.contracts import AttachmentList, FinalizedTurnResult
from dadbot.core.control_plane import (
    ExecutionControlPlane,
    ExecutionJob,
    SessionRegistry,
)
from dadbot.core.graph import TurnContext, TurnGraph
from dadbot.core.interfaces import (
    HealthService,
    InferenceService,
    validate_pipeline_services,
)
from dadbot.core.job_builder import JobBuilder
from dadbot.core.lg_topology import create_topology_provider
from dadbot.core.persistence.base import CheckpointNotFoundError
from dadbot.core.trace_binder import TraceBinder
from dadbot.registry import ServiceRegistry, boot_registry

logger = logging.getLogger(__name__)


class DeterminismViolation(RuntimeError):
    """Retained for import compatibility with existing test files."""


def _stable_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        _json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode(
            "utf-8",
        ),
    ).hexdigest()


def _dependency_versions_snapshot() -> dict[str, str]:
    deps: dict[str, str] = {}
    for name in ("pytest", "pydantic", "langchain", "openai"):
        try:
            deps[name] = str(importlib_metadata.version(name))
        except Exception:  # noqa: BLE001 — importlib_metadata probe; skip missing packages
            continue
    return deps


def _build_determinism_manifest() -> dict[str, Any]:
    env_keys = sorted(str(key) for key in os.environ.keys())
    env_hash = _stable_sha256({"env_keys": env_keys})
    manifest = {
        "python_version": str(platform.python_version()),
        "env_hash": env_hash,
        "dependency_versions": _dependency_versions_snapshot(),
        "timezone": str(os.environ.get("TZ") or "local"),
    }
    manifest["manifest_hash"] = _stable_sha256(manifest)
    return manifest


class DadBotOrchestrator:
    """Graph-governed turn orchestrator routed through ExecutionControlPlane."""

    def __init__(
        self,
        registry: ServiceRegistry | None = None,
        *,
        strict: bool = False,
        enable_observability: bool = True,
        **kwargs: Any,
    ) -> None:
        config_path = str(kwargs.pop("config_path", "config.yaml"))
        bot = kwargs.pop("bot", None)
        checkpointer = kwargs.pop("checkpointer", None)
        self.bot = bot
        self.registry = registry or boot_registry(config_path=config_path, bot=bot)
        self._strict = bool(strict)
        self._enable_observability = bool(enable_observability)
        self._last_turn_context: TurnContext | None = None
        self._checkpointer = checkpointer

        storage = self.registry.get("storage", optional=True)
        if checkpointer is not None and storage is not None and callable(getattr(storage, "set_checkpointer", None)):
            storage.set_checkpointer(checkpointer)

        self._job_builder = JobBuilder()
        self._trace_binder = TraceBinder()
        self.graph = self._build_turn_graph()
        self.control_plane = ExecutionControlPlane(
            registry=SessionRegistry(),
            kernel_executor=self._execute_job,
            graph=self.graph,
            enable_observability=self._enable_observability,
        )
        self.session_registry = self.control_plane.registry

    def _build_turn_graph(self) -> TurnGraph:
        llm = self.registry.get("llm", optional=True)
        health = self.registry.get("health", optional=True)
        validate_pipeline_services(
            {
                "llm": (llm, InferenceService),
                "health": (health, HealthService),
            },
            raise_on_failure=self._strict,
        )
        return TurnGraph(
            registry=self.registry,
            topology_provider_factory=create_topology_provider,
        )

    def _build_turn_context(
        self,
        *,
        user_input: str,
        attachments: AttachmentList | None,
        session_id: str,
        metadata: dict[str, Any] | None,
    ) -> TurnContext:
        """Backward-compat shim — delegates to JobBuilder.build."""
        return self._job_builder.build(
            user_input=user_input,
            attachments=attachments,
            session_id=session_id,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # _execute_job helpers
    # ------------------------------------------------------------------

    def _check_manifest_drift(
        self,
        manifest: dict[str, Any],
        prior_manifest: dict[str, Any],
    ) -> None:
        """Check for runtime drift between current and prior manifest."""
        if not prior_manifest:
            return
        prior_py = str(prior_manifest.get("python_version") or "")
        prior_env = str(prior_manifest.get("env_hash") or "")
        current_py = str(manifest.get("python_version") or "")
        current_env = str(manifest.get("env_hash") or "")
        py_drift = prior_py and current_py and prior_py != current_py
        env_drift = prior_env and current_env and prior_env != current_env
        if py_drift or env_drift:
            message = (
                "Manifest drift detected before turn execution: "
                f"python_version {prior_py!r}->{current_py!r}, "
                f"env_hash {prior_env!r}->{current_env!r}"
            )
            if self._strict:
                raise DeterminismViolation(message)
            logger.warning(message)

    def _load_checkpoint_data(
        self,
        session_id: str,
        manifest: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Load checkpoint if checkpointer is configured; return ``None`` if absent."""
        if self._checkpointer is None:
            return None
        load_checkpoint = getattr(self._checkpointer, "load_checkpoint", None)
        if not callable(load_checkpoint):
            return None
        try:
            return cast(
                "dict[str, Any]",
                load_checkpoint(session_id, current_manifest=manifest, strict=bool(self._strict)),
            )
        except CheckpointNotFoundError:
            return None
        except DeterminismViolation:
            raise
        except Exception as exc:
            if self._strict:
                raise DeterminismViolation(
                    f"checkpoint manifest verification failed: {exc}",
                ) from exc
            return None

    def _stamp_determinism_metadata(
        self,
        context: TurnContext,
        job: ExecutionJob,
        manifest: dict[str, Any],
    ) -> None:
        """Compute and attach all determinism hashes to *context.metadata*."""
        determinism = dict(context.metadata.get("determinism") or {})
        determinism["manifest"] = dict(manifest)
        determinism["manifest_hash"] = str(manifest.get("manifest_hash") or "")
        determinism["determinism_manifest_hash"] = str(manifest.get("manifest_hash") or "")
        tool_trace_hash = _stable_sha256(
            {
                "user_input": str(context.user_input or ""),
                "attachments": list(getattr(job, "attachments", None) or []),
                "tool_plan": context.state.get("tool_ir") or context.state.get("tool_plan") or [],
            },
        )
        lock_hash = _stable_sha256(
            {
                "user_input": str(context.user_input or ""),
                "attachments": list(getattr(job, "attachments", None) or []),
                "manifest_hash": str(manifest.get("manifest_hash") or ""),
                "tool_trace_hash": tool_trace_hash,
            },
        )
        # Stable memory fingerprint — a hash over the current memory retrieval set.
        memory_fingerprint = _stable_sha256(
            {"retrieval_set": context.state.get("memory_retrieval_set") or []},
        )
        determinism["tool_trace_hash"] = tool_trace_hash
        determinism["lock_hash"] = lock_hash
        determinism["lock_version"] = max(2, int(determinism.get("lock_version") or 2))
        determinism["memory_fingerprint"] = memory_fingerprint
        determinism["env_hash"] = str(manifest.get("env_hash") or "")
        determinism["enforced"] = bool(self._strict)
        context.metadata["determinism"] = determinism
        context.metadata["determinism_manifest"] = dict(manifest)

    def _update_session_state_after_turn(
        self,
        session: dict[str, Any],
        context: TurnContext,
        result: FinalizedTurnResult,
        manifest: dict[str, Any],
    ) -> None:
        """Stamp last_result, terminal state, goals, and manifest into session state."""
        state = session.setdefault("state", {})
        if not isinstance(state, dict):
            return
        typed_state = cast("dict[str, Any]", state)
        typed_state["last_result"] = result
        goals_any_raw = context.state.get("goals")
        if isinstance(goals_any_raw, list):
            typed_state["goals"] = list(cast("list[Any]", goals_any_raw))
        else:
            prior_goals = typed_state.get("goals")
            typed_state["goals"] = list(cast("list[Any]", prior_goals)) if isinstance(prior_goals, list) else []
        final_output = str(result[0] if isinstance(result, tuple) else result or "")
        retrieval_set = context.state.get("memory_retrieval_set") or []
        kernel_policy = dict(context.metadata.get("kernel_policy") or {})
        memory_retrieval_hash = hashlib.sha256(
            _json.dumps(retrieval_set, sort_keys=True, default=str).encode(),
        ).hexdigest()[:32]
        policy_hash = hashlib.sha256(
            _json.dumps(kernel_policy, sort_keys=True, default=str).encode(),
        ).hexdigest()[:32]
        final_trace_hash = str(context.trace_id or "")
        typed_state["last_terminal_state"] = {
            "schema_version": "1",
            "final_output": final_output,
            "final_memory_view": {
                k: context.state.get(k)
                for k in [
                    "memory_full_history_id",
                    "memory_structured",
                    "memory_retrieval_set",
                ]
                if context.state.get(k) is not None
            },
            "final_trace_hash": final_trace_hash,
            "execution_dag_hash": str(context.metadata.get("tool_execution_graph_hash") or ""),
            "policy_snapshot": kernel_policy,
            "model_output_hashes": {},
            "memory_retrieval_hash": memory_retrieval_hash,
            "policy_hash": policy_hash,
            "determinism_closure_hash": "",
        }
        typed_state["last_execution_trace_context"] = {"final_hash": final_trace_hash}
        typed_state["last_determinism_manifest"] = dict(manifest)
        typed_state["last_memory_full_history_id"] = str(context.state.get("memory_full_history_id") or "")
        typed_state["last_checkpoint_hash"] = str(getattr(context, "last_checkpoint_hash", "") or "")
        typed_state["prev_checkpoint_hash"] = str(getattr(context, "prev_checkpoint_hash", "") or "")

    async def _execute_job(
        self,
        session: dict[str, Any],
        job: ExecutionJob,
    ) -> FinalizedTurnResult:
        context = self._job_builder.build(
            user_input=str(job.user_input or ""),
            attachments=job.attachments,
            session_id=str(job.session_id or "default"),
            metadata=dict(job.metadata or {}),
        )
        if not context.trace_id:
            raise RuntimeError("TurnContext.trace_id must be non-empty")

        manifest = _build_determinism_manifest()
        prior_manifest = dict(session.get("state", {}).get("last_determinism_manifest") or {})
        self._check_manifest_drift(manifest, prior_manifest)

        loaded_checkpoint = self._load_checkpoint_data(str(job.session_id or "default"), manifest)
        if isinstance(loaded_checkpoint, dict) and loaded_checkpoint:
            context.last_checkpoint_hash = str(loaded_checkpoint.get("checkpoint_hash") or "")
            context.prev_checkpoint_hash = str(loaded_checkpoint.get("prev_checkpoint_hash") or "")

        self._stamp_determinism_metadata(context, job, manifest)

        async def _run() -> FinalizedTurnResult:
            return await self.graph.execute(context)

        result = await self._trace_binder.run(
            trace_id=context.trace_id,
            prompt=str(job.user_input or ""),
            metadata={"session_id": str(job.session_id or "default")},
            fn=_run,
        )
        self._last_turn_context = context
        self._update_session_state_after_turn(session, context, result, manifest)
        return result

    async def handle_turn(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
        *,
        session_id: str = "default",
        timeout_seconds: float | None = None,
    ) -> FinalizedTurnResult:
        try:
            return await self.control_plane.submit_turn(
                session_id=session_id,
                user_input=user_input,
                attachments=attachments,
                metadata={},
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError:
            logger.warning("Inference timed out after %ss", timeout_seconds or 30)
            return ("Sorry, I timed out while thinking. Please try again.", False)
        except Exception as exc:  # noqa: BLE001 — outermost inference guard; must return user-facing fallback
            logger.error("Inference failed: %s", exc)
            return ("Something went wrong. Please try again.", False)

    def run(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        coro = self.handle_turn(user_input, attachments=attachments)
        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()  # type: ignore[arg-type]
        return asyncio.run(coro)

    async def run_async(
        self,
        user_input: str,
        attachments: AttachmentList | None = None,
    ) -> FinalizedTurnResult:
        return await self.handle_turn(user_input, attachments=attachments)
