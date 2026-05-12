from __future__ import annotations

from dadbot.core.tool_memory_ingestion import ToolResultMemorySink


class _FakeMemory:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def commit_to_long_term(self, turn_id: str, event_payload: dict, metadata: dict) -> None:
        self.calls.append(
            {
                "turn_id": turn_id,
                "event_payload": dict(event_payload),
                "metadata": dict(metadata),
            }
        )


def test_tool_result_memory_sink_commits_execution_event():
    memory = _FakeMemory()
    sink = ToolResultMemorySink(memory_service=memory)

    sink(
        {
            "tool_name": "http_fetch",
            "status": "ok",
            "attempts": 1,
            "latency_ms": 21.0,
            "degraded_reason": "",
            "error": "",
            "output_preview": {"status_code": 200},
            "metadata": {"url": "https://example.com"},
            "captured_at_ms": 123456,
        }
    )

    assert len(memory.calls) == 1
    call = memory.calls[0]
    assert call["metadata"]["event_type"] == "TOOL_EXECUTION_RESULT"
    assert call["event_payload"]["tool_name"] == "http_fetch"
    assert call["event_payload"]["status"] == "ok"


def test_tool_result_memory_sink_handles_missing_memory_service():
    sink = ToolResultMemorySink(memory_service=None)
    sink({"tool_name": "x", "status": "ok", "captured_at_ms": 1})
