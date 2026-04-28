from __future__ import annotations

from typing import Any


def _rewrite_if_invalid(*, reply_contract, user_text: str, result, draft_reply: str, contract_report: dict) -> tuple[str, dict]:
    rewritten = reply_contract.rewrite(
        user_text=user_text,
        mood=str(result.mood or "neutral"),
        original_reply=draft_reply,
    )
    rewritten_report = reply_contract.validate(rewritten)
    if not bool(rewritten_report.get("ok")):
        raise RuntimeError(
            "Reply contract enforcement failed after rewrite: "
            + ",".join(list(rewritten_report.get("violations") or []))
        )
    final_reply = rewritten
    contract_report = {
        **rewritten_report,
        "rewritten": True,
        "original_violations": list(contract_report.get("violations") or []),
    }
    return final_reply, contract_report


def _enforce_reply_contract(*, reply_contract, user_text: str, result) -> tuple[str, dict]:
    draft_reply = str(result.reply or "").strip()
    contract_report = reply_contract.validate(draft_reply)

    if not bool(contract_report.get("ok")):
        return _rewrite_if_invalid(
            reply_contract=reply_contract,
            user_text=user_text,
            result=result,
            draft_reply=draft_reply,
            contract_report=contract_report,
        )

    final_reply = draft_reply
    contract_report = {**contract_report, "rewritten": False, "original_violations": []}
    return final_reply, contract_report


def _turn_phase_reply(*, turn_state: dict[str, Any], execution_result: dict[str, Any]) -> tuple:
    user_text = turn_state["user_text"]
    reply_contract = turn_state["reply_contract"]
    normalize_side_effect_labels = turn_state["normalize_side_effect_labels"]
    sanitize_runtime_text = turn_state["sanitize_runtime_text"]
    result = execution_result["result"]

    side_effects = normalize_side_effect_labels(getattr(result, "side_effects", []))

    final_reply, contract_report = _enforce_reply_contract(
        reply_contract=reply_contract,
        user_text=user_text,
        result=result,
    )

    scrubbed_reply, scrub_meta = sanitize_runtime_text(final_reply)
    reply_was_scrubbed = scrubbed_reply != final_reply
    final_reply = scrubbed_reply
    contract_report = {
        **dict(contract_report or {}),
        "security": {
            "output_sanitized": True,
            "output_was_scrubbed": bool(reply_was_scrubbed),
            "output_pii_types": list(scrub_meta.get("pii_types") or []),
            "output_truncated": bool(scrub_meta.get("truncated", False)),
            "output_control_chars_removed": bool(scrub_meta.get("control_chars_removed", False)),
        },
    }

    return final_reply, contract_report, side_effects


class ReplyEngine:
    def build(self, turn_state: dict[str, Any], execution_result: dict[str, Any]) -> tuple:
        return _turn_phase_reply(turn_state=turn_state, execution_result=execution_result)
