from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from typing import cast

from dadbot.contracts import DadBotContext, SupportsDadBotAccess
from dadbot.core.execution_contract import TurnDelivery, TurnResponse, live_turn_request

try:
    from rich.console import Console
    from rich.panel import Panel
except Exception:  # pragma: no cover - optional dependency in interactive UX only
    Console = None
    Panel = None


@dataclass(slots=True)
class ChatInsight:
    title: str
    lines: list[str] = field(default_factory=list)
    severity: str = "info"


@dataclass(slots=True)
class ChatTurn:
    assistant_text: str
    overlays: list[ChatInsight] = field(default_factory=list)
    severity_flags: list[str] = field(default_factory=list)


class RuntimeInterfaceManager:
    """Owns interactive CLI chat loops for the DadBot facade."""

    STORY_MODE_MAX_FAILED_ATTEMPTS = 3
    STORY_MODE_LOCKOUT_BASE_SECONDS = 30
    STORY_MODE_LOCKOUT_MAX_SECONDS = 900

    CONTEXTUAL_LEARNING_KEYWORDS: tuple[str, ...] = (
        "family",
        "mom",
        "mother",
        "dad",
        "father",
        "sister",
        "brother",
        "wife",
        "husband",
        "son",
        "daughter",
        "grandma",
        "grandpa",
        "health",
        "hospital",
        "diagnosis",
        "medication",
        "surgery",
        "therapy",
        "illness",
        "life event",
        "birthday",
        "wedding",
        "anniversary",
        "graduation",
        "funeral",
        "new job",
        "promotion",
        "moved",
        "moving",
        "divorce",
    )

    def __init__(self, bot: DadBotContext | SupportsDadBotAccess):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot
        configured_mode = str(
            getattr(self.bot, "UI_MODE", "")
            or os.environ.get("DADBOT_UI_MODE", "")
            or "chat"
        ).strip().lower()
        self.ui_mode = configured_mode if configured_mode in {"chat", "debug", "operator", "story"} else "chat"
        self.story_mode_password = str(
            getattr(self.bot, "STORY_MODE_PASSWORD", "") or os.environ.get("DADBOT_STORY_MODE_PASSWORD", ""),
        ).strip()
        self.story_mode_failed_attempts = 0
        self.story_mode_locked_until_ts = 0.0

    def _story_mode_lockout_remaining_seconds(self) -> int:
        remaining = float(getattr(self, "story_mode_locked_until_ts", 0.0) or 0.0) - time.monotonic()
        return max(0, int(remaining))

    def _reset_story_mode_password_failures(self) -> None:
        self.story_mode_failed_attempts = 0
        self.story_mode_locked_until_ts = 0.0

    def _register_story_mode_password_failure(self) -> int:
        failed_attempts = int(getattr(self, "story_mode_failed_attempts", 0) or 0) + 1
        self.story_mode_failed_attempts = failed_attempts
        threshold = int(self.STORY_MODE_MAX_FAILED_ATTEMPTS)
        if failed_attempts < threshold:
            return 0

        lockout_level = failed_attempts - threshold
        lockout_seconds = int(self.STORY_MODE_LOCKOUT_BASE_SECONDS) * (2**lockout_level)
        lockout_seconds = min(lockout_seconds, int(self.STORY_MODE_LOCKOUT_MAX_SECONDS))
        self.story_mode_locked_until_ts = time.monotonic() + float(lockout_seconds)
        return lockout_seconds

    @staticmethod
    def _ansi(enabled: bool, code: str) -> str:
        return f"\x1b[{code}m" if enabled else ""

    @staticmethod
    def _meter(value: float, width: int = 14) -> str:
        clamped = max(0.0, min(1.0, float(value)))
        filled = int(round(clamped * width))
        return "[" + ("#" * filled) + ("-" * max(0, width - filled)) + "]"

    @staticmethod
    def _authority_glyph(turn_index: int) -> str:
        wheel = ("o", "O", "0", "@")
        return wheel[int(turn_index) % len(wheel)]

    @staticmethod
    def _evidence_chain_preview(reflection_summary: dict | None, ux_feedback: dict | None) -> list[str]:
        reflection = dict(reflection_summary or {})
        feedback = dict(ux_feedback or {})
        digest = dict(feedback.get("evidence_graph_digest") or {})
        if not digest:
            digest = dict(reflection.get("evidence_graph") or {})

        preview = list(digest.get("chain_preview") or [])
        if preview:
            return [str(item) for item in preview if str(item).strip()][:3]

        edges = list(digest.get("edges") or digest.get("top_edges") or [])
        lines: list[str] = []
        for edge in edges:
            if not isinstance(edge, dict):
                continue
            source = str(edge.get("source") or "").strip()
            target = str(edge.get("target") or "").strip()
            if source and target:
                lines.append(f"{source} -> {target}")
            if len(lines) >= 3:
                break
        return lines

    @staticmethod
    def _build_soft_insight(reflection_summary: dict | None, ux_feedback: dict | None) -> ChatInsight | None:
        reflection = dict(reflection_summary or {})
        feedback = dict(ux_feedback or {})
        lines: list[str] = []

        memory_hint = str(feedback.get("memory_message") or "").strip()
        if memory_hint:
            lines.append(f"Memory influenced this: {memory_hint}")

        risk_level = str(reflection.get("current_risk_level") or "").strip()
        drift_prob = float(reflection.get("predicted_drift_probability") or 0.0)
        trigger = str(reflection.get("likely_trigger_category") or "").strip()
        if risk_level:
            lines.append(
                f"System context: risk={risk_level}, drift={drift_prob:.2f}, trigger={trigger or 'unknown'}"
            )

        evidence_lines = RuntimeInterfaceManager._evidence_chain_preview(reflection, feedback)
        if evidence_lines:
            lines.append("Evidence chain: " + " | ".join(evidence_lines[:2]))

        if not lines:
            return None
        return ChatInsight(title="System Insight", lines=lines, severity="info")

    @staticmethod
    def _turn_from_context(reply_text: str, turn_context) -> ChatTurn:
        if isinstance(turn_context, dict):
            state = dict(turn_context)
        else:
            state = dict(getattr(turn_context, "state", {}) or {}) if turn_context is not None else {}
        reflection = dict(state.get("reflection_summary") or {}) if state else {}
        ux_feedback = dict(state.get("ux_feedback") or {}) if state else {}
        mandatory_halt = bool(state.get("goal_alignment_mandatory_halt", False))

        overlays: list[ChatInsight] = []
        soft = RuntimeInterfaceManager._build_soft_insight(reflection, ux_feedback)
        if soft is not None:
            overlays.append(soft)

        flags = ["halt"] if mandatory_halt else []
        return ChatTurn(assistant_text=str(reply_text or ""), overlays=overlays, severity_flags=flags)

    def _render_chat_message(self, text: str) -> None:
        self.bot.print_speaker_message("Dad", str(text or ""))

    def _render_session_header(self, title: str) -> None:
        self.bot.print_system_message(title)
        self.bot.print_system_message("Type 'bye' or 'goodnight' to head out.")
        self.bot.print_system_message(self.bot.command_help_text())
        self.bot.print_system_message(f"UI[{self.ui_mode}]: /mode chat|debug|operator|story, /insight, /hud")
        print()

    def _render_soft_metadata(self, chat_turn: ChatTurn) -> None:
        if not chat_turn.overlays:
            return
        print("(system insight available: type /insight)")

    def _render_insight_overlay(self, chat_turn: ChatTurn) -> None:
        if not chat_turn.overlays:
            print("(no system insight for this turn)")
            return
        for overlay in chat_turn.overlays:
            print(f"[Why I said this] {overlay.title}")
            for line in overlay.lines:
                print(f"  - {line}")

    def _render_turn(self, chat_turn: ChatTurn) -> None:
        self._render_chat_message(chat_turn.assistant_text)

        if self.ui_mode in {"chat", "story"}:
            self._render_soft_metadata(chat_turn)
            return

        if self.ui_mode == "debug":
            self._render_soft_metadata(chat_turn)
            self._render_audit_hud()
            return

        # operator mode: full HUD plus expanded insight layer.
        self._render_insight_overlay(chat_turn)
        self._render_audit_hud()

    def _handle_ui_command(self, user_input: str, chat_turn: ChatTurn | None = None) -> bool:
        raw_input = str(user_input or "").strip()
        normalized = raw_input.lower()
        if not normalized.startswith("/"):
            return False

        resolved_turn = chat_turn
        if resolved_turn is None:
            ctx = getattr(self.bot, "_last_turn_context", None)
            if ctx is not None:
                safe_result = getattr(ctx, "safe_result", "")
                reply_text = safe_result[0] if isinstance(safe_result, tuple) and safe_result else safe_result
                resolved_turn = self._turn_from_context(str(reply_text or ""), ctx)

        if normalized.startswith("/mode"):
            parts = raw_input.split()
            if len(parts) == 1:
                self.bot.print_system_message(f"UI mode is '{self.ui_mode}' (chat | debug | operator | story)")
                return True
            requested = str(parts[1]).strip().lower()
            if requested in {"chat", "debug", "operator", "story"}:
                if requested == "story":
                    configured_password = str(getattr(self, "story_mode_password", "") or "").strip()
                    provided_password = str(parts[2]).strip() if len(parts) > 2 else ""
                    lockout_remaining = self._story_mode_lockout_remaining_seconds()
                    if lockout_remaining > 0:
                        self.bot.print_system_message(
                            f"Story mode is temporarily locked after failed attempts. Try again in {lockout_remaining}s.",
                        )
                        return True
                    if not configured_password:
                        self.bot.print_system_message(
                            "Story mode is locked. Set DADBOT_STORY_MODE_PASSWORD first, then use /mode story <password>."
                            " Example (PowerShell): $env:DADBOT_STORY_MODE_PASSWORD='your-password'",
                        )
                        return True
                    if provided_password != configured_password:
                        lockout_seconds = self._register_story_mode_password_failure()
                        if lockout_seconds > 0:
                            self.bot.print_system_message(
                                "Incorrect story mode password. "
                                f"Story mode is now locked for {lockout_seconds}s.",
                            )
                        else:
                            attempts_left = max(0, int(self.STORY_MODE_MAX_FAILED_ATTEMPTS) - self.story_mode_failed_attempts)
                            self.bot.print_system_message(
                                "Incorrect story mode password. "
                                f"{attempts_left} attempt(s) left before temporary lockout.",
                            )
                        return True
                    self._reset_story_mode_password_failures()
                self.ui_mode = requested
                self.bot.print_system_message(f"UI mode switched to '{self.ui_mode}'")
            else:
                self.bot.print_system_message(
                    "Unknown UI mode. Use: /mode chat | /mode debug | /mode operator | /mode story",
                )
            return True

        if normalized == "/insight":
            if resolved_turn is None:
                self.bot.print_system_message("No turn insight available yet.")
                return True
            self._render_insight_overlay(resolved_turn)
            return True

        if normalized == "/hud":
            if self.ui_mode == "chat":
                self.bot.print_system_message("HUD is hidden in chat mode. Use /mode debug or /mode operator.")
                return True
            self._render_audit_hud()
            return True

        if normalized == "/help ui":
            self.bot.print_system_message("UI commands: /mode chat|debug|operator|story, /insight, /hud")
            self.bot.print_system_message(
                "Story mode setup: set DADBOT_STORY_MODE_PASSWORD, then run /mode story <password>.",
            )
            return True

        self.bot.print_system_message("Unknown command. Try /help ui")
        return True

    def _trigger_story_learning(self) -> None:
        if self.ui_mode != "story":
            return
        self._trigger_learning_update()

    @classmethod
    def _should_trigger_contextual_learning(cls, user_input: str) -> bool:
        normalized = str(user_input or "").strip().lower()
        if not normalized:
            return False
        return any(keyword in normalized for keyword in cls.CONTEXTUAL_LEARNING_KEYWORDS)

    def _trigger_learning_update(self) -> None:
        try:
            self.bot.apply_relationship_feedback("supportive")
        except Exception:
            pass
        learn_now = getattr(self.bot, "perform_continuous_learning_cycle", None)
        if callable(learn_now):
            try:
                learn_now()
            except Exception:
                pass

    def _trigger_contextual_learning(self, user_input: str) -> None:
        if self.ui_mode == "story" or self._should_trigger_contextual_learning(user_input):
            self._trigger_learning_update()

    def _render_halt_mode_hud(
        self,
        *,
        turn_index: int,
        reflection_summary: dict | None,
        ux_feedback: dict | None = None,
    ) -> None:
        """
        Render high-contrast alert state for MANDATORY_HALT.
        
        This emphasizes cognitive interruption, not visual spectacle.
        Shows detected patterns and actionable recommendations.
        """
        enabled = str(getattr(self.bot, "NO_COLOR", "") or "").strip().lower() not in {"1", "true", "yes", "on"}
        reset = self._ansi(enabled, "0")
        bold_red = self._ansi(enabled, "1;31")  # Bold red
        white = self._ansi(enabled, "37")
        dim = self._ansi(enabled, "2")

        print()
        print(f"{bold_red}{'=' * 80}{reset}")
        print(f"{bold_red}[ALIGNMENT INTERRUPT]{reset}")
        print(f"{bold_red}{'=' * 80}{reset}")
        print()

        print(f"{white}Current action appears inconsistent with declared goal trajectory.{reset}")
        print()

        if reflection_summary:
            trigger_category = reflection_summary.get("likely_trigger_category", "unknown")
            recent_episodes = reflection_summary.get("recent_episode_count", 0)
            primary_pattern = reflection_summary.get("primary_pattern_name", "")
            recommended = reflection_summary.get("recommended_intervention", "")
            intervention_justification = reflection_summary.get("intervention_justification", "")

            print(f"{dim}Detected Pattern Analysis:{reset}")
            if primary_pattern:
                print(f"  • Pattern: {primary_pattern}")
            print(f"  • Trigger Category: {trigger_category}")
            print(f"  • Recent Episodes: {recent_episodes}")
            print()

            if intervention_justification:
                print(f"{dim}Analysis:{reset}")
                print(f"  {intervention_justification}")
                print()

            evidence_lines = self._evidence_chain_preview(reflection_summary, ux_feedback)
            if evidence_lines:
                print(f"{dim}Evidence Chain:{reset}")
                for line in evidence_lines:
                    print(f"  • {line}")
                print()

        print(f"{white}Recommended next step:{reset}")
        print(f"  • Return to active task")
        print(f"  • Or initiate goal recalibration via 'recalibrate' command")
        print()
        print(f"{dim}Enter realignment phrase to resume or 'recalibrate' to adjust goals.{reset}")
        print(f"{bold_red}{'=' * 80}{reset}")
        print()

    def _render_rich_hud(
        self,
        *,
        turn_index: int,
        trust: float,
        overlap: float,
        drift: bool,
        streak: int,
        pressure: float,
        dominant_topic: str,
        checkpoint_hash: str,
        mandatory_halt: bool,
        reflection_summary: dict | None = None,
        ux_feedback: dict | None = None,
    ) -> bool:
        if Console is None or Panel is None:
            return False
        console = Console()
        
        # If in halt mode, render alert state instead of normal HUD
        if mandatory_halt:
            alert_body = (
                f"[bold red]ALIGNMENT INTERRUPT[/bold red]\n\n"
                f"Current action appears inconsistent with declared goal trajectory.\n"
            )
            if reflection_summary:
                pattern = reflection_summary.get("primary_pattern_name", "")
                category = reflection_summary.get("likely_trigger_category", "unknown")
                episodes = reflection_summary.get("recent_episode_count", 0)
                if pattern:
                    alert_body += f"\n[yellow]Pattern:[/yellow] {pattern}\n"
                alert_body += f"[yellow]Trigger:[/yellow] {category}  [yellow]Recent:[/yellow] {episodes}\n"
            alert_body += (
                f"\n[bold]Recommended next step:[/bold]\n"
                f"  • Return to active task\n"
                f"  • Or initiate goal recalibration via 'recalibrate' command\n"
            )
            console.print(Panel(alert_body, border_style="red", title="[bold red]INTERRUPT[/bold red]"))
            return True
        
        authority_ok = bool(checkpoint_hash)
        authority_state = "LOCKED" if authority_ok else "UNSEALED"
        authority_spin = self._authority_glyph(turn_index)
        halt_state = "MANDATORY_HALT" if mandatory_halt else "FLOW"
        border_style = "red" if (mandatory_halt or drift) else "cyan"

        header_body = (
            f"[bold]ZERO-G // TURN {turn_index}[/bold]\n"
            f"trust {self._meter(trust)} {trust:.2f}   "
            f"alignment {self._meter(overlap)} {overlap:.2f}   "
            f"authority {authority_spin} {authority_state}"
        )
        evidence_lines = self._evidence_chain_preview(reflection_summary, ux_feedback)
        if evidence_lines:
            header_body += "\n[dim]evidence:[/dim] " + "  |  ".join(evidence_lines)
        footer_body = (
            f"topic={dominant_topic}  drift_streak={streak}  budget_pressure={pressure:.2f}  "
            f"state={halt_state}"
        )

        console.print(Panel(header_body, border_style=border_style, title="header"))
        console.print(Panel(footer_body, border_style=border_style, title="footer"))
        return True

    def _render_audit_hud(self) -> None:
        """Render a per-turn Zero-G HUD for CLI users."""
        ctx = getattr(self.bot, "_last_turn_context", None)
        state = dict(getattr(ctx, "state", {}) or {}) if ctx is not None else {}

        relational = dict(state.get("relational_state") or {})
        temporal = dict(state.get("temporal_budget") or {})
        if not relational and not temporal:
            return

        no_color = str(getattr(self.bot, "NO_COLOR", "") or "").strip().lower() in {"1", "true", "yes", "on"}
        enabled = not no_color
        reset = self._ansi(enabled, "0")
        dim = self._ansi(enabled, "2")
        cyan = self._ansi(enabled, "36")
        green = self._ansi(enabled, "32")
        yellow = self._ansi(enabled, "33")
        red = self._ansi(enabled, "31")

        drift = bool(relational.get("topic_drift_detected"))
        overlap = float(relational.get("topic_overlap_ratio") or 0.0)
        trust = float(relational.get("trust_credit") or 0.0)
        turn_index = int(temporal.get("turn_index") or 0)
        streak = int(temporal.get("topic_drift_streak") or 0)
        pressure = float(temporal.get("budget_pressure") or 0.0)
        dominant_topic = str(relational.get("dominant_topic") or "general")
        mandatory_halt = bool(state.get("goal_alignment_mandatory_halt", False))
        checkpoint_hash = str(getattr(ctx, "last_checkpoint_hash", "") or "") if ctx is not None else ""
        reflection_summary = dict(state.get("reflection_summary") or {}) if state else {}
        ux_feedback = dict(state.get("ux_feedback") or {}) if state else {}

        # If in halt mode, render alert first
        if mandatory_halt:
            self._render_halt_mode_hud(
                turn_index=turn_index,
                reflection_summary=reflection_summary or None,
                ux_feedback=ux_feedback or None,
            )

        if self._render_rich_hud(
            turn_index=turn_index,
            trust=trust,
            overlap=overlap,
            drift=drift,
            streak=streak,
            pressure=pressure,
            dominant_topic=dominant_topic,
            checkpoint_hash=checkpoint_hash,
            mandatory_halt=mandatory_halt,
            reflection_summary=reflection_summary or None,
            ux_feedback=ux_feedback or None,
        ):
            return

        drift_text = "TOPIC DRIFT" if drift else "ON TRACK"
        drift_color = red if drift else green
        authority_state = "LOCKED" if checkpoint_hash else "UNSEALED"
        authority_spin = self._authority_glyph(turn_index)
        halt_state = "MANDATORY_HALT" if mandatory_halt else "FLOW"

        print(
            f"{dim}[ZERO-G HEADER]{reset} {cyan}turn#{turn_index}{reset} "
            f"trust={self._meter(trust)} {trust:.2f} alignment={self._meter(overlap)} {overlap:.2f} "
            f"authority={authority_spin}:{authority_state}",
        )
        print(
            f"{dim}[ZERO-G FOOTER]{reset} topic={dominant_topic} {drift_color}{drift_text}{reset} "
            f"drift_streak={streak} budget_pressure={pressure:.2f} state={halt_state}",
        )

        evidence_lines = self._evidence_chain_preview(reflection_summary, ux_feedback)
        if evidence_lines:
            print(f"{dim}[EVIDENCE]{reset} " + " | ".join(evidence_lines))

    def chat_loop(self):
        self._render_session_header("--- Dad is online a\u2764\ufe0f ---")

        if not self.bot.ensure_ollama_ready():
            return

        self.bot.reset_session_state()
        opening = self.bot.opening_message(default_message="")
        if opening:
            self.bot.print_speaker_message("Dad", opening)

        while True:
            try:
                user_input = input("Tony: ").strip()
                if not user_input:
                    continue
                if self._handle_ui_command(user_input):
                    continue

                response = self.bot.execute_turn(
                    live_turn_request(
                        user_input,
                        delivery=TurnDelivery.SYNC,
                        session_id=str(getattr(self.bot, "active_thread_id", "") or "default"),
                    ),
                )
                dad_reply, should_end = cast(TurnResponse, response).as_result()
                if dad_reply:
                    chat_turn = self._turn_from_context(str(dad_reply), getattr(self.bot, "_last_turn_context", None))
                    self._render_turn(chat_turn)
                    self._trigger_contextual_learning(user_input)

                if should_end:
                    break
            except KeyboardInterrupt:
                self.bot.persist_conversation()
                self.bot.print_speaker_message(
                    "Dad",
                    self.bot.reply_finalization.append_signoff(
                        "Hey Tony, you okay? I'm right here.",
                    ),
                )
                break
            except EOFError:
                self.bot.persist_conversation()
                break
            except Exception:
                self.bot.print_speaker_message(
                    "Dad",
                    self.bot.reply_finalization.append_signoff(
                        "Sorry buddy, something went a bit wonky there. Try again?",
                    ),
                )

    def chat_loop_via_service(self, service_client, session_id=None):
        self._render_session_header("--- Dad is online via Dad Bot API a\u2764\ufe0f ---")

        session_key = str(session_id or f"cli-{uuid.uuid4().hex}")
        service_client.ensure_service_running(preferred_model=self.bot.MODEL_NAME)
        self.bot.reset_session_state()
        opening = self.bot.opening_message(default_message="")
        if opening:
            self.bot.print_speaker_message("Dad", opening)

        last_chat_turn: ChatTurn | None = None

        while True:
            try:
                user_input = input("Tony: ").strip()
                if not user_input:
                    continue
                if self._handle_ui_command(user_input, chat_turn=last_chat_turn):
                    continue

                result = service_client.chat(
                    session_key,
                    user_input=user_input,
                    requested_model=self.bot.MODEL_NAME,
                )
                if result.session_state:
                    self.bot.load_session_state_snapshot(result.session_state)
                if result.active_model:
                    self.bot.ACTIVE_MODEL = result.active_model

                if result.reply:
                    chat_turn = self._turn_from_context(str(result.reply), dict(result.session_state or {}))
                    self._render_turn(chat_turn)
                    last_chat_turn = chat_turn
                    self._trigger_contextual_learning(user_input)

                if result.should_end:
                    self.bot.persist_conversation()
                    break
            except KeyboardInterrupt:
                self.bot.persist_conversation()
                self.bot.print_speaker_message(
                    "Dad",
                    self.bot.reply_finalization.append_signoff(
                        "Hey Tony, you okay? I'm right here.",
                    ),
                )
                break
            except EOFError:
                self.bot.persist_conversation()
                break
            except Exception:
                self.bot.print_speaker_message(
                    "Dad",
                    self.bot.reply_finalization.append_signoff(
                        "Sorry buddy, something went a bit wonky there. Try again?",
                    ),
                )


__all__ = ["RuntimeInterfaceManager"]
