from __future__ import annotations

from dadbot.ui.voice_control_plane import VoiceSessionController


def test_voice_controller_defaults_and_mode_transition() -> None:
    voice = {"enabled": True, "mode": "push_to_talk"}
    runtime = {}
    controller = VoiceSessionController(voice, runtime_state=runtime)

    snap = controller.snapshot()
    assert snap["state"] == "IDLE"
    assert snap["mode"] == "push_to_talk"

    assert controller.set_mode("always_listening") is True
    snap = controller.snapshot()
    assert snap["mode"] == "always_listening"
    assert snap["state"] == "LISTENING"
    assert snap["safety_flags"]["auto_listen_allowed"] is True


def test_voice_controller_state_machine_for_transcription() -> None:
    voice = {
        "enabled": True,
        "mode": "push_to_talk",
        "wake_word_required": False,
    }
    controller = VoiceSessionController(voice, runtime_state={})

    transcript, error = controller.process_audio_capture(
        b"x" * 6000,
        transcribe_fn=lambda payload: ("hey dad this is a test", ""),
    )

    assert error == ""
    assert transcript == "hey dad this is a test"
    snap = controller.snapshot()
    assert snap["state"] == "THINKING"
    assert snap["last_transcript"] == "hey dad this is a test"


def test_voice_controller_wake_phrase_filter() -> None:
    voice = {
        "enabled": True,
        "mode": "always_listening",
        "wake_word_required": True,
        "wake_word_phrase": "hey dad",
    }
    controller = VoiceSessionController(voice, runtime_state={})

    transcript, error = controller.process_audio_capture(
        b"x" * 6000,
        transcribe_fn=lambda payload: ("hello there", ""),
    )

    assert transcript == ""
    assert error == ""
    snap = controller.snapshot()
    assert snap["state"] == "LISTENING"


def test_voice_controller_mute_and_error_paths() -> None:
    voice = {"enabled": True, "mode": "push_to_talk", "muted": False}
    controller = VoiceSessionController(voice, runtime_state={})

    controller.set_muted(True)
    assert controller.snapshot()["muted"] is True

    transcript, error = controller.process_audio_capture(
        b"x" * 6000,
        transcribe_fn=lambda payload: ("", "backend down"),
    )
    assert transcript == ""
    assert error == "backend down"
    snap = controller.snapshot()
    assert snap["state"] == "ERROR"
    assert snap["last_error"] == "backend down"

    controller.clear_error()
    assert controller.snapshot()["state"] == "IDLE"


def test_voice_controller_emits_ledger_state_and_event_records() -> None:
    emitted = []

    def _emit(event_type: str, payload: dict) -> None:
        emitted.append((event_type, payload))

    controller = VoiceSessionController(
        {"enabled": True, "mode": "push_to_talk"},
        runtime_state={},
        ledger_emitter=_emit,
        session_id_provider=lambda: "thread-1",
        trace_id_provider=lambda: "trace-1",
    )

    controller.begin_recording()
    controller.begin_transcribing()
    controller.begin_thinking()

    event_types = [item[0] for item in emitted]
    assert "VOICE_STATE_TRANSITION" in event_types
    assert "VOICE_EVENT" in event_types
    assert any(item[1].get("session_id") == "thread-1" for item in emitted)
    assert any(item[1].get("trace_id") == "trace-1" for item in emitted)


def test_voice_controller_guarded_transition_rejects_invalid_step() -> None:
    controller = VoiceSessionController({"enabled": True, "mode": "push_to_talk"}, runtime_state={})

    # BEGIN_TRANSCRIBING from IDLE is invalid under guarded transitions.
    controller.begin_transcribing()
    snap = controller.snapshot()

    assert snap["state"] == "ERROR"
    assert "Invalid voice transition" in snap["last_error"]


def test_voice_controller_interruptibility_and_cancel_consumption() -> None:
    controller = VoiceSessionController(
        {
            "enabled": True,
            "mode": "always_listening",
            "interruptions_enabled": True,
            "barge_in_enabled": True,
            "barge_in_min_audio_bytes": 2000,
        },
        runtime_state={},
    )

    controller.begin_recording()
    controller.begin_transcribing()
    controller.begin_thinking()
    controller.begin_speaking()
    assert controller.can_barge_in(b"x" * 4000) is True

    assert controller.request_cancel(reason="barge_in_audio", priority="high") is True
    first = controller.consume_cancel()
    second = controller.consume_cancel()

    assert first["cancel_requested"] is True
    assert first["cancel_reason"] == "barge_in_audio"
    assert first["priority_override"] == "high"
    assert second["cancel_requested"] is False
