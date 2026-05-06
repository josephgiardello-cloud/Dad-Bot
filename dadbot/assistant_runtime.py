from __future__ import annotations

from typing import Any

from dadbot.core.execution_contract import TurnDelivery, live_turn_request


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


__all__ = ["AssistantRuntime"]
