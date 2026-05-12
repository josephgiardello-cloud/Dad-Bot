from __future__ import annotations

from typing import Any

from dadbot.core.execution_contract import TurnDelivery, live_turn_request
from dadbot.runtime.agent_driver_loop import AgentDriverLoop, DriverLoopPolicy, DriverLoopResult
from dadbot.runtime.context_pruner import ContextWindowPruner, build_pruned_observation_hook
from dadbot.runtime.semantic_memory_bridge import MemoryConsolidationJob, build_semantic_snippet_provider


class AssistantRuntime:
    """Stable, minimal assistant-facing facade over the execution kernel."""

    def __init__(self, kernel: Any):
        self.kernel = kernel

    def chat(self, message: str, *, debug: bool = False) -> dict[str, Any]:
        execute_turn = getattr(self.kernel, "execute_turn", None)
        if callable(execute_turn):
            response_obj = execute_turn(
                live_turn_request(str(message or ""), delivery=TurnDelivery.SYNC),
            )
            if hasattr(response_obj, "as_result"):
                result_payload = response_obj.as_result()
                if isinstance(result_payload, tuple) and len(result_payload) >= 2:
                    final_output, _should_save = result_payload
                else:
                    final_output = str(getattr(response_obj, "reply", "") or "")
                    _should_save = bool(getattr(response_obj, "should_end", False))
            elif isinstance(response_obj, tuple):
                final_output, _should_save = response_obj
            else:
                final_output = str(getattr(response_obj, "reply", "") or "")
                _should_save = bool(getattr(response_obj, "should_end", False))
        else:
            legacy_turn = getattr(self.kernel, "process_user_message", None)
            if callable(legacy_turn):
                final_output, _should_save = legacy_turn(str(message or ""))
            else:
                final_output = ""
                _should_save = False
        response = {
            "response": str(final_output or ""),
            "memory_updates": None,
            "tool_calls": None,
        }
        if debug:
            response["debug"] = {
                "trace": dict(getattr(self.kernel, "_last_turn_pipeline", {}) or {}),
                "planner": dict(getattr(self.kernel, "_last_planner_debug", {}) or {}),
                "memory_context": dict(getattr(self.kernel, "_last_memory_context_stats", {}) or {}),
                "reply_supervisor": dict(getattr(self.kernel, "_last_reply_supervisor", {}) or {}),
                "terminal_state": dict(getattr(self.kernel, "last_terminal_state", {}) or {}),
            }
        return response

    def run_task(self, message: str) -> str:
        orchestration = getattr(self.kernel, "runtime_orchestration", None)
        if orchestration is None:
            raise RuntimeError("runtime_orchestration manager is not available")
        execute_turn = getattr(self.kernel, "execute_turn", None)
        legacy_turn = getattr(self.kernel, "process_user_message", None)
        task_fn = execute_turn if callable(execute_turn) else legacy_turn
        if not callable(task_fn):
            raise RuntimeError("kernel does not expose execute_turn or process_user_message")
        task_arg = (
            live_turn_request(str(message or ""), delivery=TurnDelivery.SYNC)
            if callable(execute_turn)
            else str(message or "")
        )
        future = orchestration.submit_background_task(
            task_fn,
            task_arg,
            task_kind="assistant.run_task",
            metadata={"api_surface": "assistant"},
        )
        task_id = str(getattr(future, "dadbot_task_id", "") or "")
        if not task_id and callable(getattr(future, "result", None)):
            try:
                result = future.result()
                task_id = str((result or {}).get("task_id") or "") if isinstance(result, dict) else ""
            except Exception:
                task_id = ""
        if not task_id:
            raise RuntimeError("background task did not produce a task_id")
        return task_id

    def get_state(self, task_id: str) -> dict[str, Any]:
        tid = str(task_id or "").strip()
        if not tid:
            return {"task_id": "", "status": "unknown"}

        task_store = getattr(getattr(self.kernel, "runtime_state_container", None), "store", None)
        if task_store is not None and callable(getattr(task_store, "load_task", None)):
            payload = task_store.load_task(tid)
            if isinstance(payload, dict):
                return {
                    "task_id": tid,
                    "status": str(payload.get("status") or "unknown"),
                    "error": str(payload.get("error") or ""),
                    "updated_at": payload.get("updated_at"),
                }

        background = getattr(self.kernel, "background_tasks", None)
        task = background.get_task(tid) if background is not None and callable(getattr(background, "get_task", None)) else None
        if task is not None:
            return {
                "task_id": tid,
                "status": str(getattr(task, "status", "unknown") or "unknown"),
                "error": str(getattr(task, "error", "") or ""),
                "updated_at": getattr(task, "completed_at", None) or getattr(task, "started_at", None),
            }

        return {"task_id": tid, "status": "unknown"}

    def run_heartbeat(self, *, force: bool = True) -> dict[str, Any]:
        maintenance = getattr(self.kernel, "maintenance_scheduler", None)
        if maintenance is None or not callable(getattr(maintenance, "run_proactive_heartbeat", None)):
            raise RuntimeError("maintenance_scheduler is not available")
        return dict(maintenance.run_proactive_heartbeat(force=force) or {})

    def run_self_improvement(self, *, force: bool = True, background: bool = False) -> dict[str, Any]:
        manager = getattr(self.kernel, "long_term_signals", None)
        if manager is None:
            raise RuntimeError("long_term_signals manager is not available")

        if background and not force and callable(getattr(manager, "schedule_continuous_learning", None)):
            task = manager.schedule_continuous_learning()
            return {
                "status": "queued" if task is not None else "skipped",
                "task_id": str(getattr(task, "dadbot_task_id", "") or "") if task is not None else "",
            }

        perform_cycle = getattr(manager, "perform_continuous_learning_cycle", None)
        if not callable(perform_cycle):
            raise RuntimeError("continuous learning execution is not available")

        if background:
            submit = getattr(self.kernel, "submit_background_task", None)
            if not callable(submit):
                raise RuntimeError("submit_background_task is not available")
            task = submit(
                perform_cycle,
                task_kind="continuous-learning",
                metadata={"api_surface": "assistant", "forced": force},
            )
            return {"status": "queued", "task_id": str(getattr(task, "dadbot_task_id", "") or "")}

        return {"status": "completed", "result": dict(perform_cycle() or {})}

    def browser_status(self) -> dict[str, Any]:
        status = getattr(self.kernel, "local_mcp_status", None)
        if not callable(status):
            raise RuntimeError("local_mcp_status is not available")
        return dict(status() or {})

    def start_browser_tools(self, *, restart: bool = False) -> dict[str, Any]:
        start = getattr(self.kernel, "start_local_mcp_server_process", None)
        if not callable(start):
            raise RuntimeError("start_local_mcp_server_process is not available")
        return dict(start(restart=restart) or {})

    def stop_browser_tools(self) -> dict[str, Any]:
        stop = getattr(self.kernel, "stop_local_mcp_server_process", None)
        if not callable(stop):
            raise RuntimeError("stop_local_mcp_server_process is not available")
        return dict(stop() or {})

    def reset_session(self) -> None:
        self.kernel.reset_session_state()

    def memory(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        text = str(query or "")
        matcher = getattr(self.kernel, "find_memory_matches", None)
        if callable(matcher):
            matches = matcher(text)
            return list(matches or [])

        query_mgr = getattr(self.kernel, "memory_query", None)
        if query_mgr is not None and callable(getattr(query_mgr, "relevant_memories_for_input", None)):
            matches = query_mgr.relevant_memories_for_input(text, limit=limit)
            return list(matches or [])

        return []

    def run_agent_loop(
        self,
        initial_input: str,
        *,
        policy: DriverLoopPolicy | None = None,
        session_id: str = "default",
        reflection_hook: Any | None = None,
        observation_hook: Any | None = None,
        enable_semantic_memory: bool = True,
        semantic_top_k: int = 3,
        semantic_sync_index: bool = True,
    ) -> DriverLoopResult:
        loop = AgentDriverLoop(self.kernel, policy=policy)
        effective_observation_hook = observation_hook

        if effective_observation_hook is None and enable_semantic_memory:
            snippet_provider = build_semantic_snippet_provider(
                self.kernel,
                top_k=max(1, int(semantic_top_k)),
                session_id=str(session_id or "default"),
                sync_index=bool(semantic_sync_index),
            )
            core_identity = ""
            request_prompt_builder = getattr(self.kernel, "build_request_system_prompt", None)
            if callable(request_prompt_builder):
                try:
                    core_identity = str(request_prompt_builder() or "")
                except Exception:
                    core_identity = ""

            effective_observation_hook = build_pruned_observation_hook(
                ContextWindowPruner(max_turns=10),
                core_identity=core_identity,
                snippet_provider=snippet_provider,
            )

        startup_observation = self._executive_startup_observation(max_tasks=8)
        if startup_observation:
            base_hook = effective_observation_hook

            def _composed_observation_hook(ctx: dict[str, Any]) -> str:
                if callable(base_hook):
                    base_text = str(base_hook(ctx) or "")
                else:
                    turn_index = int(ctx.get("turn_index", 1) or 1)
                    base_text = (
                        str(ctx.get("initial_observation") or "")
                        if turn_index == 1
                        else str(ctx.get("last_reply") or ctx.get("initial_observation") or "")
                    )

                if int(ctx.get("turn_index", 1) or 1) != 1:
                    return base_text

                # Inject executive pending-task context once at startup so the loop can be proactive.
                return f"{startup_observation}\n\n{base_text}".strip()

            effective_observation_hook = _composed_observation_hook

        return loop.run(
            initial_input,
            session_id=session_id,
            reflection_hook=reflection_hook,
            observation_hook=effective_observation_hook,
        )

    @staticmethod
    def _executive_startup_observation(*, max_tasks: int = 8) -> str:
        try:
            from dadbot_system.local_mcp_server import get_pending_executive_tasks

            tasks = list(get_pending_executive_tasks(limit=max(1, int(max_tasks))))
        except Exception:
            return ""

        if not tasks:
            return ""

        lines = ["Startup executive tasks (pending):"]
        for task in tasks:
            task_id = str(task.get("id") or "").strip()
            title = str(task.get("title") or "").strip()
            due = str(task.get("due") or "").strip()
            priority = str(task.get("priority") or "normal").strip()
            if not title:
                continue
            due_fragment = f" due={due}" if due else ""
            lines.append(f"- [{task_id}] {title} (priority={priority}{due_fragment})")
        return "\n".join(lines).strip()

    def run_memory_consolidation(
        self,
        *,
        session_id: str = "default",
        window_size: int = 10,
        background: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        job = MemoryConsolidationJob(
            self.kernel,
            session_id=str(session_id or "default"),
            window_size=max(3, int(window_size)),
        )
        if background:
            return dict(job.schedule(force=force) or {})
        return dict(job.run_once(force=force) or {})


__all__ = ["AssistantRuntime"]
