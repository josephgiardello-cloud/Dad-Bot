from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TURN_SERVICE_PATH = ROOT / "dadbot" / "services" / "turn_service.py"
PROMPT_ASSEMBLY_PATH = ROOT / "dadbot" / "managers" / "prompt_assembly.py"
REPLY_FINALIZATION_PATH = ROOT / "dadbot" / "managers" / "reply_finalization.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = []
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name):
            parts.append(func.id)
        return ".".join(reversed(parts))
    return ""


def _collect_calls(tree: ast.Module) -> list[str]:
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            calls.append(_call_name(node))
    return calls


def run_check() -> dict[str, object]:
    turn_service_tree = _parse(TURN_SERVICE_PATH)
    prompt_tree = _parse(PROMPT_ASSEMBLY_PATH)
    finalizer_tree = _parse(REPLY_FINALIZATION_PATH)

    turn_calls = _collect_calls(turn_service_tree)
    prompt_calls = _collect_calls(prompt_tree)
    finalizer_calls = _collect_calls(finalizer_tree)

    violations: list[str] = []

    if any("autonomous_tool_result_for_input" in call for call in turn_calls):
        violations.append("tool decision must be single-origin; autonomous_tool_result_for_input fallback is still wired")

    tool_origin_marks = [
        call
        for call in turn_calls
        if call.endswith("mark_turn_coherence")
    ]
    if len(tool_origin_marks) < 2:
        violations.append("tool decision coherence mark missing in sync/async prepare paths")

    memory_context_calls = [call for call in prompt_calls if call.endswith("_resolve_turn_memory_context")]
    if len(memory_context_calls) != 1:
        violations.append("memory context must be injected exactly once in prompt assembly")

    direct_memory_builder_calls = [
        call
        for call in prompt_calls
        if call.endswith("context_builder.build_memory_context")
    ]
    if len(direct_memory_builder_calls) != 1:
        violations.append("memory retrieval must be single path via begin_turn_memory_context")

    personality_calls = [
        call
        for call in finalizer_calls
        if call.endswith("personality_service.apply_authoritative_voice")
    ]
    if len(personality_calls) != 2:
        violations.append("personality authority must be applied exactly once per sync/async finalizer")

    personality_assertion_calls = [
        call
        for call in finalizer_calls
        if call.endswith("assert_personality_applied_exactly_once")
    ]
    if len(personality_assertion_calls) != 2:
        violations.append(
            "assert_personality_applied_exactly_once must be called in both sync and async finalizer paths "
            "(end-to-end guarantee that personality is applied exactly once per turn)"
        )

    for forbidden in (
        "tone_context.blend_daily_checkin_reply",
        "maybe_add_family_echo",
    ):
        if any(call.endswith(forbidden) for call in finalizer_calls):
            violations.append(f"finalizer must not own tone shaping directly: {forbidden}")

    return {
        "ok": not violations,
        "violations": violations,
    }


def main() -> int:
    report = run_check()
    if report["ok"]:
        print("[turn_coherence_check] OK")
        return 0
    print("[turn_coherence_check] FAIL")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
