import asyncio
import datetime
import json
import traceback
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

import streamlit as st

from dadbot.components.voice import render_realtime_voice_call, render_reply_tts, render_voice_controls
from dadbot.config import DadBotConfig, DadRuntimeConfig
from dadbot.contracts import DadBotContext
from dadbot.core.dadbot import DadBot
from dadbot.managers.memory_manager import MemoryManager
from dadbot.managers.mood_manager import MoodManager
from dadbot.managers.profile_runtime import ProfileRuntimeManager
from dadbot.managers.relationship_manager import RelationshipManager
from dadbot.runtime.model.model_call_port import ModelConfig
from dadbot.ui.media.service import MediaService
from dadbot_system.events import InMemoryEventBus


def _now_stamp() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _ensure_ui_runtime_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chat_threads" not in st.session_state:
        seeded = list(st.session_state.chat_history or [])
        st.session_state.chat_threads = {"default": seeded}

    if "active_thread_id" not in st.session_state:
        st.session_state.active_thread_id = "default"

    if "runtime_events" not in st.session_state:
        st.session_state.runtime_events = []

    if "runtime_guardrails" not in st.session_state:
        st.session_state.runtime_guardrails = []

    if "message_outbox" not in st.session_state:
        st.session_state.message_outbox = []


def _active_thread_id() -> str:
    return str(st.session_state.get("active_thread_id") or "default")


def _thread_ids() -> list[str]:
    return sorted(str(k) for k in dict(st.session_state.get("chat_threads") or {}).keys())


def _thread_messages(thread_id: str | None = None) -> list[dict[str, Any]]:
    tid = str(thread_id or _active_thread_id())
    threads = dict(st.session_state.get("chat_threads") or {})
    values = list(threads.get(tid) or [])
    return [dict(item) for item in values if isinstance(item, dict)]


def _chat_input_payload(value: Any) -> tuple[str, list[Any]]:
    if value is None:
        return "", []
    if isinstance(value, str):
        return value, []

    text = str(getattr(value, "text", "") or "")
    files = getattr(value, "files", None)

    if isinstance(value, dict):
        text = str(value.get("text") or text or "")
        if files is None:
            files = value.get("files")

    return text, list(files or [])


def _set_thread_messages(messages: list[dict[str, Any]], thread_id: str | None = None) -> None:
    tid = str(thread_id or _active_thread_id())
    threads = dict(st.session_state.get("chat_threads") or {})
    threads[tid] = [dict(item) for item in list(messages or []) if isinstance(item, dict)]
    st.session_state.chat_threads = threads
    if tid == _active_thread_id():
        st.session_state.chat_history = list(threads[tid])


def _append_chat_message(role: str, text: str, *, stamp: str, thread_id: str | None = None) -> None:
    tid = str(thread_id or _active_thread_id())
    messages = _thread_messages(tid)
    messages.append({"role": str(role or "assistant"), "text": str(text or ""), "time": str(stamp or _now_stamp())})
    _set_thread_messages(messages, tid)


def _switch_active_thread(thread_id: str) -> None:
    tid = str(thread_id or "default")
    threads = dict(st.session_state.get("chat_threads") or {})
    if tid not in threads:
        threads[tid] = []
        st.session_state.chat_threads = threads
    st.session_state.active_thread_id = tid
    st.session_state.chat_history = _thread_messages(tid)


def _create_new_thread() -> str:
    tid = f"thread-{uuid.uuid4().hex[:8]}"
    threads = dict(st.session_state.get("chat_threads") or {})
    threads[tid] = []
    st.session_state.chat_threads = threads
    _switch_active_thread(tid)
    return tid


def _fork_active_thread() -> str:
    source_id = _active_thread_id()
    source = _thread_messages(source_id)
    tid = f"fork-{uuid.uuid4().hex[:8]}"
    threads = dict(st.session_state.get("chat_threads") or {})
    threads[tid] = list(source)
    st.session_state.chat_threads = threads
    _switch_active_thread(tid)
    return tid


def _record_runtime_event(event_type: str, summary: str, *, severity: str = "info", payload: dict[str, Any] | None = None) -> None:
    events = list(st.session_state.get("runtime_events") or [])
    events.append(
        {
            "time": _now_stamp(),
            "thread_id": _active_thread_id(),
            "event_type": str(event_type or "event"),
            "severity": str(severity or "info"),
            "summary": str(summary or ""),
            "payload": dict(payload or {}),
        }
    )
    st.session_state.runtime_events = events[-250:]


def _record_guardrail(rule: str, detail: str, *, severity: str = "warning") -> None:
    guardrails = list(st.session_state.get("runtime_guardrails") or [])
    guardrails.append(
        {
            "time": _now_stamp(),
            "thread_id": _active_thread_id(),
            "rule": str(rule or "runtime_guardrail"),
            "severity": str(severity or "warning"),
            "detail": str(detail or ""),
        }
    )
    st.session_state.runtime_guardrails = guardrails[-120:]


def _queue_failed_message(prompt: str, error: str) -> None:
    queue = list(st.session_state.get("message_outbox") or [])
    queue.append(
        {
            "thread_id": _active_thread_id(),
            "prompt": str(prompt or ""),
            "error": str(error or ""),
            "time": _now_stamp(),
        }
    )
    st.session_state.message_outbox = queue[-100:]


def _render_thread_sidebar() -> None:
    st.sidebar.subheader("Threads")
    thread_ids = _thread_ids()
    active = _active_thread_id()
    if active not in thread_ids:
        _switch_active_thread("default")
        thread_ids = _thread_ids()
        active = _active_thread_id()

    index = thread_ids.index(active) if active in thread_ids else 0
    selected = st.sidebar.selectbox("Active thread", thread_ids, index=index, key="thread-picker")
    if str(selected) != active:
        _switch_active_thread(str(selected))
        st.rerun()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("New", use_container_width=True):
            new_tid = _create_new_thread()
            _record_runtime_event("thread_created", f"Created {new_tid}")
            st.rerun()
    with col2:
        if st.button("Fork", use_container_width=True):
            new_tid = _fork_active_thread()
            _record_runtime_event("thread_forked", f"Forked into {new_tid}")
            st.rerun()

    st.sidebar.caption(f"Messages: {len(_thread_messages(active))}")


def _render_guardrail_strip() -> None:
    guardrails = list(st.session_state.get("runtime_guardrails") or [])
    if not guardrails:
        return
    recent = guardrails[-1]
    sev = str(recent.get("severity") or "warning").lower()
    msg = f"{recent.get('time', '')} · {recent.get('rule', 'guardrail')}: {recent.get('detail', '')}"
    if sev in {"critical", "error"}:
        st.error(msg)
    else:
        st.warning(msg)


def _render_runtime_timeline(limit: int = 30) -> None:
    events = list(st.session_state.get("runtime_events") or [])[-max(1, int(limit)) :]
    if not events:
        st.info("No runtime events yet.")
        return
    rows = []
    for item in reversed(events):
        rows.append(
            {
                "time": str(item.get("time") or ""),
                "thread": str(item.get("thread_id") or "default"),
                "event": str(item.get("event_type") or "event"),
                "severity": str(item.get("severity") or "info"),
                "summary": str(item.get("summary") or ""),
            }
        )
    st.dataframe(rows, use_container_width=True)


def _extract_ollama_text(response: Any) -> str:
    payload: dict[str, Any]
    if hasattr(response, "model_dump"):
        payload = dict(response.model_dump() or {})
    elif isinstance(response, dict):
        payload = dict(response)
    else:
        payload = {}

    message = payload.get("message") if isinstance(payload, dict) else None
    if isinstance(message, dict):
        text = str(message.get("content") or "").strip()
        if text:
            return text

    choices = payload.get("choices") if isinstance(payload, dict) else None
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            msg = choice.get("message") or choice.get("delta") or {}
            if isinstance(msg, dict):
                text = str(msg.get("content") or "").strip()
                if text:
                    return text

    text = str(payload.get("response") or "").strip() if isinstance(payload, dict) else ""
    return text


class _StreamlitControlPlane:
    def __init__(self, service: "StrictLLMService") -> None:
        self._service = service

    async def execute_from_graph_context(self, turn_context: Any, rich_context: dict[str, Any] | None = None):
        reply = await self._service.run_agent(turn_context, dict(rich_context or {}))
        return str(reply or ""), False


class StrictLLMService:
    def __init__(self, model_name: str, bot: Any = None) -> None:
        self.model_name = str(model_name or "llama3.2:latest")
        self.bot = bot
        self.control_plane = _StreamlitControlPlane(self)

    @staticmethod
    def _collect_attachments(context: Any, rich_context: dict[str, Any]) -> list[dict[str, Any]]:
        def _as_items(value: Any) -> list[Any]:
            if isinstance(value, list):
                return list(value)
            if isinstance(value, tuple):
                return list(value)
            return []

        candidates: list[Any] = []
        if isinstance(rich_context, dict):
            candidates.extend(_as_items(rich_context.get("attachments")))
            candidates.extend(_as_items(rich_context.get("norm_attachments")))

        state = getattr(context, "state", None)
        if isinstance(state, dict):
            candidates.extend(_as_items(state.get("attachments")))
            candidates.extend(_as_items(state.get("norm_attachments")))

        candidates.extend(_as_items(getattr(context, "attachments", [])))

        attachments: list[dict[str, Any]] = []
        for attachment in candidates:
            if not isinstance(attachment, dict):
                continue
            kind = str(attachment.get("type") or "").strip().lower()
            if kind not in {"image", "audio", "document"}:
                continue
            attachments.append(dict(attachment))
        return attachments[:6]

    @staticmethod
    def _extract_image_payload(attachments: list[dict[str, Any]]) -> list[str]:
        images: list[str] = []
        for attachment in list(attachments or []):
            if not isinstance(attachment, dict):
                continue
            kind = str(attachment.get("type") or "").strip().lower()
            if kind != "image":
                continue
            image_b64 = str(attachment.get("image_b64") or attachment.get("data_b64") or "").strip()
            if image_b64:
                images.append(image_b64)

        deduped: list[str] = []
        seen: set[str] = set()
        for payload in images:
            if payload in seen:
                continue
            seen.add(payload)
            deduped.append(payload)
        return deduped[:6]

    @staticmethod
    def _model_supports_image_input(model_name: str) -> bool:
        lowered = str(model_name or "").strip().lower()
        if not lowered:
            return False
        hints = (
            "vision",
            "llava",
            "bakllava",
            "moondream",
            "minicpm-v",
            "qwen2-vl",
            "qwen2.5-vl",
            "qwen2.5vl",
            "gemma3",
        )
        return any(hint in lowered for hint in hints)

    @staticmethod
    def _augment_messages_with_attachment_note(
        messages: list[dict[str, Any]],
        attachments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        image_names = [str(item.get("name") or "image") for item in attachments if item.get("type") == "image"]
        if not image_names:
            return messages

        note = "User uploaded image(s): " + ", ".join(image_names[:6])
        enriched = [dict(item) for item in list(messages or [])]
        for idx in range(len(enriched) - 1, -1, -1):
            if str(enriched[idx].get("role") or "").lower() != "user":
                continue
            content = str(enriched[idx].get("content") or "").strip()
            enriched[idx]["content"] = f"{content}\n\n{note}".strip()
            return enriched

        enriched.append({"role": "user", "content": note})
        return enriched

    def _to_messages(
        self,
        context: Any,
        rich_context: dict[str, Any],
        attachments: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        raw_messages = rich_context.get("messages") if isinstance(rich_context, dict) else None
        normalized: list[dict[str, Any]] = []
        if isinstance(raw_messages, list):
            for item in raw_messages:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role") or "user").strip().lower() or "user"
                content = str(item.get("content") or "").strip()
                if content:
                    normalized.append({"role": role, "content": content})

        if not normalized:
            user_input = str(getattr(context, "user_input", "") or "").strip()
            if user_input:
                normalized = [{"role": "user", "content": user_input}]
            else:
                normalized = [{"role": "user", "content": "Hello"}]

        selected_attachments = list(attachments or self._collect_attachments(context, rich_context))
        images = self._extract_image_payload(selected_attachments)
        if images:
            attached = False
            for idx in range(len(normalized) - 1, -1, -1):
                if str(normalized[idx].get("role") or "").lower() == "user":
                    enriched = dict(normalized[idx])
                    enriched["images"] = images
                    normalized[idx] = enriched
                    attached = True
                    break
            if not attached:
                normalized.append(
                    {
                        "role": "user",
                        "content": "Please review the attached image(s).",
                        "images": images,
                    }
                )

        return normalized

    def _chat_sync(self, messages: list[dict[str, Any]], attachments: list[dict[str, Any]]) -> str:
        import ollama

        selected_model = self.model_name
        image_turn = any(bool(list(item.get("images") or [])) for item in list(messages or []))

        if image_turn and not self._model_supports_image_input(selected_model):
            finder = getattr(self.bot, "find_available_vision_model", None)
            if callable(finder):
                with suppress(Exception):
                    candidate = str(finder() or "").strip()
                    if candidate:
                        selected_model = candidate

        selected_messages = list(messages or [])
        if image_turn and not self._model_supports_image_input(selected_model):
            selected_messages = self._augment_messages_with_attachment_note(selected_messages, attachments)

        response = ollama.chat(model=selected_model, messages=selected_messages)
        text = _extract_ollama_text(response)
        if text:
            return text
        raise RuntimeError("Ollama returned an empty response")

    async def run_agent(self, context: Any, rich_context: dict[str, Any]) -> str:
        attachments = self._collect_attachments(context, rich_context)
        messages = self._to_messages(context, rich_context, attachments)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self._chat_sync(messages, attachments))


def _build_bot() -> DadBot:
    class MinimalPersistenceService:
        def finalize_turn(self, turn_context: Any, result: Any):
            return result

        def save_turn(self, turn_context: Any, result: Any) -> None:
            pass

        def save_turn_event(self, event: dict[str, Any]) -> None:
            pass

        def save_graph_checkpoint(self, checkpoint: dict[str, Any], _skip_turn_event: bool = False) -> None:
            pass

    class MinimalSafetyService:
        def tick(self, context: Any = None) -> dict[str, Any]:
            return {"status": "ok"}

    class MinimalContextService:
        def build_context(self, turn_context: Any) -> dict[str, Any]:
            return {"messages": [], "temporal": dict(getattr(turn_context, "state", {}).get("temporal") or {})}

    class MinimalMaintenanceService:
        def tick(self, context: Any = None) -> dict[str, Any]:
            return {"status": "ok"}

    class MinimalReflectionService:
        def reflect_after_turn(self, turn_text: str, current_mood: str, reply_text: str) -> dict[str, Any]:
            return {"reflection": "noop"}

        def reflect(self, turn_context: Any, result: Any) -> dict[str, Any]:
            return {"reflection": "noop"}

    class ModelRuntimeStub:
        def __init__(self, active_model: str):
            self._active_model = str(active_model or "llama3.2:latest")

        def initialize_tokenizer(self, model_name: str | None = None):
            return None

        def current_tokenizer(self, model_name: str | None = None):
            class DummyTokenizer:
                @staticmethod
                def encode(text: str):
                    return str(text or "").split()

            return DummyTokenizer()

        @staticmethod
        def model_chars_per_token_estimate(model_name: str | None = None) -> float:
            return 4.0

        @staticmethod
        def normalized_llm_provider() -> str:
            return "ollama"

        def normalized_llm_model(self, model_name: str | None = None) -> str:
            return str(model_name or self._active_model)

        @staticmethod
        def resolve_temperature(options: dict[str, Any] | None = None, default: float = 0.7) -> float:
            if isinstance(options, dict):
                with suppress(TypeError, ValueError):
                    return float(options.get("temperature", default))
            return float(default)

        def litellm_model_identifier(self, model_name: str | None = None) -> str:
            return self.normalized_llm_model(model_name)

        @staticmethod
        def extract_stream_chunk_content(chunk: Any) -> str:
            if isinstance(chunk, dict):
                choices = chunk.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("delta") or choices[0].get("message") or {}
                    if isinstance(msg, dict):
                        return str(msg.get("content") or "")
            return ""

        def ollama_status(self) -> dict[str, Any]:
            try:
                import ollama

                models = ollama.list()
                payload = models.model_dump() if hasattr(models, "model_dump") else dict(models or {})
                items = list(payload.get("models") or []) if isinstance(payload, dict) else []
                return {
                    "connected": True,
                    "model": self._active_model,
                    "models_count": len(items),
                }
            except Exception as exc:
                return {
                    "connected": False,
                    "model": self._active_model,
                    "error": str(exc),
                }

    class _TempDadBot:
        pass

    temp_context = DadBotContext(_TempDadBot())
    memory_manager = MemoryManager(temp_context)
    relationship_manager = RelationshipManager(temp_context)
    mood_manager = MoodManager(temp_context)
    profile_runtime = ProfileRuntimeManager(temp_context)

    runtime_config = DadRuntimeConfig()
    config = DadBotConfig(runtime_config=runtime_config)
    event_bus = InMemoryEventBus()
    model_name = "llama3.2:latest"
    model_config = ModelConfig(active_model=model_name)
    model_runtime_stub = ModelRuntimeStub(model_config.active_model)

    from dadbot.core.interfaces import HealthService, InferenceService
    from dadbot.core.services import build_services
    from dadbot.registry import ServiceRegistry

    class StrictHealthService(HealthService):
        def tick(self, context: Any = None):
            return {"status": "ok"}

    class StrictInferenceService(StrictLLMService, InferenceService):
        pass

    registry = ServiceRegistry()
    for key, value in build_services().items():
        registry.register(key, value)

    llm_service = StrictInferenceService(model_name)
    registry.register("llm", llm_service)
    registry.register("agent_service", llm_service)
    registry.register("health", StrictHealthService())
    registry.register("context_service", MinimalContextService())
    registry.register("maintenance_service", MinimalMaintenanceService())
    registry.register("persistence_service", MinimalPersistenceService())
    registry.register("safety_service", MinimalSafetyService())
    registry.register("reflection", MinimalReflectionService())

    class PatchedDadBot(DadBot):
        def _get_turn_orchestrator(self):
            existing = getattr(self, "_turn_orchestrator", None)
            if existing is not None:
                return existing
            from dadbot.core.orchestrator import DadBotOrchestrator

            orchestrator = DadBotOrchestrator(
                registry=registry,
                config_path="config.yaml",
                bot=self,
                strict=True,
            )
            self._turn_orchestrator = orchestrator
            return orchestrator

    bot = PatchedDadBot(
        config=config,
        runtime_config=runtime_config,
        memory_manager=None,
        relationship_manager=None,
        mood_manager=None,
        profile_runtime=None,
        event_bus=event_bus,
        model_runtime=model_runtime_stub,
        validate_managers=False,
    )

    bot._memory_manager = memory_manager
    bot._relationship_manager = relationship_manager
    bot._mood_manager = mood_manager
    bot._profile_runtime = profile_runtime
    bot.services = None
    llm_service.bot = bot

    return bot


_STREAMLIT_RUNTIME_BUILD = "2026-05-30-inline-upload-v2"

if "bot" not in st.session_state:
    st.session_state.bot = _build_bot()
    st.session_state._streamlit_runtime_build = _STREAMLIT_RUNTIME_BUILD
elif str(st.session_state.get("_streamlit_runtime_build") or "") != _STREAMLIT_RUNTIME_BUILD:
    st.session_state.bot = _build_bot()
    st.session_state._streamlit_runtime_build = _STREAMLIT_RUNTIME_BUILD

_ensure_ui_runtime_state()
_switch_active_thread(_active_thread_id())

bot = st.session_state.bot
media_service = MediaService()

avatar_path = Path("static/dad_avatar.png")
if avatar_path.exists():
    st.sidebar.image(str(avatar_path), width=80)
st.sidebar.title("DadBot")
page = st.sidebar.radio("Navigation", ["Chat", "Smart Home", "Voice", "Status", "Workshop"])
_render_thread_sidebar()

if page == "Chat":
    st.title("DadBot Chat")
    _render_guardrail_strip()

    for msg in _thread_messages():
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            st.caption(msg["time"])

    try:
        chat_input_value = st.chat_input("Say something to Dad...", accept_file="multiple")
    except TypeError:
        chat_input_value = st.chat_input("Say something to Dad...")

    prompt, uploaded_files = _chat_input_payload(chat_input_value)
    if prompt or uploaded_files:
        attachments, attachment_issues = media_service.process_upload(uploaded_files)
        for issue in list(attachment_issues or []):
            st.warning(str(issue))
        if attachments:
            st.caption(f"Attachments ready: {len(attachments)}")
        if not prompt and attachments:
            prompt = "Please review the attached files."

        now = datetime.datetime.now().strftime("%H:%M")
        _append_chat_message("user", prompt, stamp=now)
        _record_runtime_event(
            "chat_user_message",
            "User submitted message",
            payload={"length": len(prompt), "attachments": len(attachments)},
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Dad is thinking..."):
            try:
                reply, _ = bot.process_user_message(prompt, attachments=attachments)
                reply = str(reply or "")
                _record_runtime_event("chat_response_ok", "Assistant response generated", payload={"length": len(reply)})
            except Exception as exc:
                traceback.print_exc()
                reply = f"Sorry, I hit a snag: {exc}"
                _record_runtime_event("chat_response_error", "Assistant response failed", severity="error", payload={"error": str(exc)})
                _record_guardrail("turn_execution_error", str(exc), severity="error")
                _queue_failed_message(prompt, str(exc))

        _append_chat_message("assistant", reply, stamp=now)
        with st.chat_message("assistant"):
            st.markdown(reply)
        with suppress(Exception):
            render_reply_tts(bot, reply)
        st.rerun()

elif page == "Smart Home":
    st.title("Smart Home Controls")
    st.caption("MQTT controls are live when a broker is configured.")

    broker = st.text_input("Broker host", value=str(st.session_state.get("smarthome_broker") or "localhost"))
    port = st.number_input("Broker port", min_value=1, max_value=65535, value=int(st.session_state.get("smarthome_port") or 1883), step=1)
    st.session_state.smarthome_broker = broker
    st.session_state.smarthome_port = int(port)

    topic = st.text_input("Topic", value="dadbot/home/living_room/light")
    payload = st.text_input("Payload", value="ON")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Connect MQTT", use_container_width=True):
            try:
                from dadbot.smart_home.mqtt_client import SmartHomeMQTTClient

                client = SmartHomeMQTTClient(broker, int(port))
                client.connect()
                st.session_state.mqtt_client = client
                st.success("Connected to MQTT broker")
            except Exception as exc:
                st.error(f"Connect failed: {exc}")
    with col2:
        if st.button("Publish", use_container_width=True):
            client = st.session_state.get("mqtt_client")
            if client is None:
                st.warning("Connect first")
            else:
                try:
                    client.publish(topic, payload)
                    st.success(f"Published to {topic}")
                except Exception as exc:
                    st.error(f"Publish failed: {exc}")
    with col3:
        if st.button("Disconnect", use_container_width=True):
            client = st.session_state.get("mqtt_client")
            if client is not None:
                with suppress(Exception):
                    client.disconnect()
            st.session_state.mqtt_client = None
            st.info("Disconnected")

elif page == "Voice":
    st.title("Voice")
    st.caption("Restored voice controls from the previous UI surface.")
    try:
        render_voice_controls(bot)
        render_realtime_voice_call(bot)
    except Exception as exc:
        st.error(f"Voice surface failed: {exc}")

elif page == "Status":
    st.title("Runtime Status")
    _render_guardrail_strip()
    shell = dict(bot.ui_shell_snapshot() or {})
    ollama = dict(shell.get("ollama") or {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Thread", str(shell.get("active_thread_id") or "default"))
    c2.metric("Mood", str(shell.get("last_mood") or "neutral"))
    c3.metric("Threads", str(len(list(shell.get("threads") or []))))

    st.subheader("Model Runtime")
    st.json(
        {
            "model": str(ollama.get("model") or "llama3.2:latest"),
            "connected": bool(ollama.get("connected", False)),
            "models_count": int(ollama.get("models_count") or 0),
            "error": str(ollama.get("error") or ""),
        },
    )

    if st.button("Probe one turn", use_container_width=True):
        try:
            probe_reply, _ = bot.process_user_message("Health probe: reply with one short sentence.")
            _record_runtime_event("status_probe_ok", "Status probe completed")
            st.success(str(probe_reply or "ok"))
        except Exception as exc:
            _record_runtime_event("status_probe_error", "Status probe failed", severity="error", payload={"error": str(exc)})
            _record_guardrail("status_probe_error", str(exc), severity="error")
            st.error(f"Probe failed: {exc}")

    st.subheader("Runtime Timeline")
    _render_runtime_timeline(limit=25)

    st.subheader("Guardrail Feed")
    guardrails = list(st.session_state.get("runtime_guardrails") or [])[-20:]
    if not guardrails:
        st.info("No guardrail events recorded.")
    else:
        st.dataframe(
            [
                {
                    "time": str(item.get("time") or ""),
                    "thread": str(item.get("thread_id") or "default"),
                    "rule": str(item.get("rule") or ""),
                    "severity": str(item.get("severity") or "warning"),
                    "detail": str(item.get("detail") or ""),
                }
                for item in reversed(guardrails)
            ],
            use_container_width=True,
        )

elif page == "Workshop":
    st.title("Workshop")
    st.caption("Operational tools restored for fast runtime inspection and recovery.")
    _render_guardrail_strip()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run live probe", use_container_width=True):
            try:
                reply, _ = bot.process_user_message("Workshop probe: confirm system ready.")
                _record_runtime_event("workshop_probe_ok", "Workshop probe completed")
                st.success(str(reply or "ok"))
            except Exception as exc:
                _record_runtime_event("workshop_probe_error", "Workshop probe failed", severity="error", payload={"error": str(exc)})
                _record_guardrail("workshop_probe_error", str(exc), severity="error")
                st.error(f"Probe failed: {exc}")

        if st.button("Clear chat history", use_container_width=True):
            _set_thread_messages([], _active_thread_id())
            _record_runtime_event("thread_cleared", "Cleared active thread messages")
            st.info("Chat history cleared")

    with col2:
        if st.button("Reset MQTT session", use_container_width=True):
            client = st.session_state.get("mqtt_client")
            if client is not None:
                with suppress(Exception):
                    client.disconnect()
            st.session_state.mqtt_client = None
            _record_runtime_event("mqtt_reset", "Reset MQTT session")
            st.info("MQTT session reset")

        if st.button("Reset voice runtime state", use_container_width=True):
            st.session_state.voice_runtime_state = {}
            _record_runtime_event("voice_state_reset", "Reset voice runtime state")
            st.info("Voice runtime state reset")

    st.subheader("Outbox")
    queue = list(st.session_state.get("message_outbox") or [])
    if not queue:
        st.info("Outbox is empty.")
    else:
        st.dataframe(
            [
                {
                    "time": str(item.get("time") or ""),
                    "thread": str(item.get("thread_id") or "default"),
                    "prompt": str(item.get("prompt") or "")[:120],
                    "error": str(item.get("error") or "")[:160],
                }
                for item in reversed(queue[-25:])
            ],
            use_container_width=True,
        )

        retry_col, clear_col = st.columns(2)
        with retry_col:
            if st.button("Retry queued", use_container_width=True):
                retried = 0
                remaining = []
                for item in queue:
                    prompt = str(item.get("prompt") or "").strip()
                    thread_id = str(item.get("thread_id") or "default")
                    if not prompt:
                        continue
                    _switch_active_thread(thread_id)
                    try:
                        now = datetime.datetime.now().strftime("%H:%M")
                        _append_chat_message("user", prompt, stamp=now, thread_id=thread_id)
                        reply, _ = bot.process_user_message(prompt)
                        _append_chat_message("assistant", str(reply or ""), stamp=now, thread_id=thread_id)
                        retried += 1
                    except Exception as exc:
                        remaining.append(
                            {
                                "thread_id": thread_id,
                                "prompt": prompt,
                                "error": str(exc),
                                "time": _now_stamp(),
                            }
                        )
                        _record_guardrail("outbox_retry_error", str(exc), severity="error")
                st.session_state.message_outbox = remaining
                _record_runtime_event("outbox_retry", f"Retried {retried} queued messages")
                st.success(f"Retried {retried} queued messages")
                st.rerun()
        with clear_col:
            if st.button("Clear outbox", use_container_width=True):
                st.session_state.message_outbox = []
                _record_runtime_event("outbox_cleared", "Cleared message outbox")
                st.info("Outbox cleared")

    st.subheader("Export / Import Active Thread")
    active_payload = {
        "thread_id": _active_thread_id(),
        "messages": _thread_messages(),
    }
    st.download_button(
        "Download active thread JSON",
        data=json.dumps(active_payload, indent=2, ensure_ascii=True),
        file_name=f"{_active_thread_id()}-thread.json",
        mime="application/json",
        use_container_width=True,
    )

    uploaded_thread = st.file_uploader("Import thread JSON", type=["json"], key="workshop-import-thread")
    if uploaded_thread is not None and st.button("Import into new thread", use_container_width=True):
        try:
            parsed = json.loads(uploaded_thread.getvalue().decode("utf-8", errors="replace"))
            incoming = list(parsed.get("messages") or []) if isinstance(parsed, dict) else []
            cleaned = [dict(item) for item in incoming if isinstance(item, dict)]
            new_tid = _create_new_thread()
            _set_thread_messages(cleaned, new_tid)
            _record_runtime_event("thread_imported", f"Imported thread into {new_tid}")
            st.success(f"Imported thread into {new_tid}")
            st.rerun()
        except Exception as exc:
            _record_guardrail("thread_import_error", str(exc), severity="error")
            st.error(f"Import failed: {exc}")