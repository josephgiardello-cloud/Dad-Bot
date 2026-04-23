from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from .contracts import ChatRequest, DEFAULT_TENANT_ID, HealthResponse, ServiceConfig
from .kernel import ControlPlane, build_control_plane

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    FastAPI = None
    HTTPException = None
    CORSMiddleware = None

    class WebSocket:  # pragma: no cover - fallback only when FastAPI is missing.
        pass

    class WebSocketDisconnect(Exception):
        pass


def create_api_app(orchestrator, *, worker_manager=None, config: ServiceConfig | None = None, control_plane: ControlPlane | None = None):
    if FastAPI is None or HTTPException is None or CORSMiddleware is None:
        exc = ImportError("FastAPI is not installed")
        raise RuntimeError("FastAPI is not installed. Install the 'service' dependencies to run the API layer.") from exc

    service_config = config or ServiceConfig()
    _control_plane = control_plane if control_plane is not None else build_control_plane()

    @asynccontextmanager
    async def lifespan(_app):
        async def _kernel_execute(session_entry: dict, payload: dict) -> dict:
            _session_id = str(payload.get("_session_id") or "")
            request = ChatRequest.from_dict(payload, session_id=_session_id)
            _timeout = float(payload.get("timeout_seconds") or 60.0)
            task = orchestrator.submit_chat(request)
            task_status, response = await _wait_for_task_completion(
                task.task_id, timeout_seconds=_timeout
            )
            return {
                "task": task_status,
                "response": response,
                "execution_graph": task.execution_graph.to_dict()
                if task.execution_graph is not None
                else None,
            }

        _scheduler_task = asyncio.create_task(
            _control_plane.scheduler.run(_kernel_execute)
        )
        try:
            yield
        finally:
            _control_plane.scheduler.stop()
            _scheduler_task.cancel()
            try:
                await _scheduler_task
            except asyncio.CancelledError:
                pass
            if worker_manager is not None:
                worker_manager.shutdown()

    app = FastAPI(title=service_config.api.title, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=service_config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def _wait_for_task_completion(task_id: str, *, timeout_seconds: float = 60.0):
        deadline = asyncio.get_running_loop().time() + max(1.0, float(timeout_seconds or 60.0))
        while asyncio.get_running_loop().time() < deadline:
            status = orchestrator.task_status(task_id)
            if status is None:
                return None, None
            response = orchestrator.response_for_task(task_id)
            if response is not None:
                return status, response
            if str(status.get("status") or "").strip().lower() == "failed":
                return status, None
            await asyncio.sleep(0.05)
        return orchestrator.task_status(task_id), orchestrator.response_for_task(task_id)

    @app.get("/health")
    async def health_check():
        health = HealthResponse(
            status="ok",
            workers=worker_manager.worker_count if worker_manager is not None else 0,
            queue_backend=service_config.queue.backend,
            state_backend=type(orchestrator.state_store).__name__,
            service_name=service_config.telemetry.service_name,
        )
        return health.to_dict()

    @app.post("/sessions/{session_id}/chat")
    async def submit_chat(session_id: str, payload: dict):
        request = ChatRequest.from_dict(payload, session_id=session_id)
        if not request.user_input and not request.attachments:
            raise HTTPException(status_code=400, detail="user_input or attachments are required")
        task = orchestrator.submit_chat(request)
        return {
            "task_id": task.task_id,
            "request_id": request.request_id,
            "status": "queued",
            "execution_graph": task.execution_graph.to_dict() if task.execution_graph is not None else None,
        }

    @app.post("/sessions/{session_id}/turn")
    async def execute_turn(session_id: str, payload: dict):
        # Validate input before submitting to the scheduler.
        _check_request = ChatRequest.from_dict(payload, session_id=session_id)
        if not _check_request.user_input and not _check_request.attachments:
            raise HTTPException(status_code=400, detail="user_input or attachments are required")

        timeout_seconds = float(payload.get("timeout_seconds") or 60.0)

        # Route through ControlPlane → Scheduler → kernel_execute_fn.
        job_payload = {**payload, "_session_id": session_id}
        result_future = await _control_plane.submit_turn(session_id, job_payload)

        try:
            result = await asyncio.wait_for(result_future, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Timed out waiting for turn completion")

        task_status = result.get("task")
        response = result.get("response")

        if task_status is None:
            raise HTTPException(status_code=404, detail="Task not found")
        if response is None and str((task_status or {}).get("status") or "").strip().lower() != "failed":
            raise HTTPException(status_code=504, detail="Timed out waiting for turn completion")

        return result

    @app.post("/sessions/{session_id}/stream")
    async def submit_stream(session_id: str, payload: dict):
        request = ChatRequest.from_dict(payload, session_id=session_id)
        if not request.user_input and not request.attachments:
            raise HTTPException(status_code=400, detail="user_input or attachments are required")

        task = orchestrator.submit_chat(request)
        tenant_id = request.tenant_id or DEFAULT_TENANT_ID
        return {
            "task_id": task.task_id,
            "request_id": request.request_id,
            "status": "queued",
            "stream": {
                "type": "websocket",
                "path": f"/sessions/{session_id}/events/stream",
                "tenant_id": tenant_id,
            },
            "execution_graph": task.execution_graph.to_dict() if task.execution_graph is not None else None,
        }

    @app.get("/tasks/{task_id}")
    async def get_task_status(task_id: str):
        status = orchestrator.task_status(task_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Task not found")
        response = orchestrator.response_for_task(task_id)
        return {"task": status, "response": response}

    @app.get("/sessions/{session_id}/events")
    async def get_session_events(session_id: str, tenant_id: str = DEFAULT_TENANT_ID):
        return orchestrator.session_events(session_id, tenant_id=tenant_id)

    @app.get("/sessions/{session_id}/replay")
    async def replay_session_events(
        session_id: str,
        tenant_id: str = DEFAULT_TENANT_ID,
        event_type: str = "",
        since_event_id: str = "",
        limit: int = 200,
    ):
        events = list(orchestrator.session_events(session_id, tenant_id=tenant_id))

        normalized_event_type = str(event_type or "").strip().lower()
        if normalized_event_type:
            events = [event for event in events if str(event.get("event_type") or "").strip().lower() == normalized_event_type]

        normalized_since_event_id = str(since_event_id or "").strip()
        if normalized_since_event_id:
            replay_from_index = 0
            for index, event in enumerate(events):
                if str(event.get("event_id") or "") == normalized_since_event_id:
                    replay_from_index = index + 1
                    break
            events = events[replay_from_index:]

        safe_limit = max(1, min(int(limit or 200), 2000))
        if len(events) > safe_limit:
            events = events[-safe_limit:]

        return {
            "session_id": session_id,
            "tenant_id": tenant_id,
            "event_count": len(events),
            "events": events,
        }

    @app.websocket("/sessions/{session_id}/events/stream")
    async def stream_session_events(websocket: WebSocket, session_id: str, tenant_id: str = DEFAULT_TENANT_ID):
        await websocket.accept()
        event_bus = getattr(orchestrator, "event_bus", None)
        subscriber = event_bus.subscribe() if event_bus is not None and hasattr(event_bus, "subscribe") else None
        try:
            for event in orchestrator.session_events(session_id, tenant_id=tenant_id):
                await websocket.send_json(event)

            if subscriber is None:
                while True:
                    await asyncio.sleep(15)
                    await websocket.send_json({"event_type": "heartbeat", "session_id": session_id, "tenant_id": tenant_id})

            while True:
                event = await asyncio.to_thread(subscriber.get, True, 15.0)
                payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
                if payload.get("session_id") != session_id or payload.get("tenant_id") != tenant_id:
                    continue
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            return
        finally:
            if subscriber is not None and event_bus is not None and hasattr(event_bus, "unsubscribe"):
                event_bus.unsubscribe(subscriber)

    return app