import logging
import tempfile
from pathlib import Path

logging.disable(logging.CRITICAL)

# Patch response engine to log candidate selection.
import dadbot.core.response_engine as re_mod

orig_run = re_mod.ResponseEngine.run


def patched_run(self, context):
    initial = getattr(context, "initial_response", "?")
    result = orig_run(self, context)
    print(f"[RE.run] initial_response={initial!r} -> result={result!r}")
    return result


re_mod.ResponseEngine.run = patched_run

from tests.stress.phase4_certification_gate import build_bot


def _build_bot_from_disk(temp_path, *, reply="Trace path OK."):
    return build_bot(temp_path, reply=reply, restore_from_disk=True)


def mixed_input(i):
    msgs = [
        "Can you help me debug this?",
        "I feel overwhelmed today",
        "I keep overthinking everything before bed.",
        "My project deadline is tomorrow",
        "I think I need a break",
        "What should I focus on?",
        "Thank you for listening",
        "Goodnight",
    ]
    return msgs[i % len(msgs)]


def main() -> int:
    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        clean_path = Path(td1)
        fail_path = Path(td2)
        clean_bot = build_bot(clean_path, reply="Trace path OK.")
        failed_bot = build_bot(fail_path, reply="Trace path OK.")

        # Run clean bot through 3 turns.
        for i in range(3):
            msg = mixed_input(i)
            r = clean_bot.process_user_message(msg)
            print(f"clean turn {i + 1}: {r}")

        print("--- failed bot turns 1-2 then crash/restart turn 3 ---")
        for i in range(2):
            msg = mixed_input(i)
            r = failed_bot.process_user_message(msg)
            print(f"failed turn {i + 1}: {r}")

        # Crash turn 3.
        orig = failed_bot.relationship_manager.materialize_projection

        def crash_once(*a, **kw):
            failed_bot.relationship_manager.materialize_projection = orig
            raise RuntimeError("crash injection")

        failed_bot.relationship_manager.materialize_projection = crash_once
        try:
            failed_bot.process_user_message(mixed_input(2))
            print("DEBUG_RESTART_FAIL: expected crash did not occur")
            return 1
        except Exception as e:
            print(f"expected crash: {type(e).__name__}: {e}")

        failed_bot.shutdown()
        failed_bot2 = _build_bot_from_disk(fail_path, reply="Trace path OK.")
        msg3 = mixed_input(2)
        r3 = failed_bot2.process_user_message(msg3)
        print(f"restarted turn 3: {r3}")

        recovered_text = str((r3[0] if isinstance(r3, tuple) and r3 else r3) or "")
        if recovered_text != msg3:
            print("DEBUG_RESTART_FAIL: restarted reply drift detected")
            return 1

        print("DEBUG_RESTART_OK")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"DEBUG_RESTART_FAIL: unhandled {type(exc).__name__}: {exc}")
        raise SystemExit(1)