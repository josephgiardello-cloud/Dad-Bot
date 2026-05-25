"""Email drafting and optional SMTP delivery."""

from __future__ import annotations

import os


class EmailManager:
    """Owns email drafting via tool registry and optional live SMTP delivery."""

    def __init__(self, bot):
        self.bot = bot

    def send_email(
        self,
        recipient: str,
        subject: str = "",
        body: str = "",
    ) -> dict | None:
        """Draft and save an email. Returns draft metadata or None on failure.
        To actually deliver mail, configure DADBOT_SMTP_* env vars; without them
        the draft is saved to email_drafts/ as an .eml file for manual sending.
        """
        # Use the new tool registry for drafting emails
        tool_registry = getattr(self.bot, "tool_registry", None)
        if tool_registry is None:
            return None
        invocation = {
            "tool_spec": tool_registry.get_spec("draft_email"),
            "arguments": {"recipient": recipient, "subject": subject, "body": body},
            "invocation_id": None  # Not needed for direct call
        }
        executor = tool_registry.get_executor("draft_email")
        if executor is None:
            return None
        result = executor(type("ToolInvocation", (), invocation)())
        draft = result.payload if result and hasattr(result, "payload") else None
        if not draft:
            return None
        if not draft:
            return None

        smtp_host = os.environ.get("DADBOT_SMTP_HOST", "").strip()
        smtp_port = int(os.environ.get("DADBOT_SMTP_PORT", "587") or 587)
        smtp_user = os.environ.get("DADBOT_SMTP_USER", "").strip()
        smtp_pass = os.environ.get("DADBOT_SMTP_PASS", "").strip()
        from_addr = os.environ.get(
            "DADBOT_SMTP_FROM",
            smtp_user or "dad@local.dadbot",
        ).strip()

        if smtp_host and smtp_user and smtp_pass:
            import smtplib
            from email.message import EmailMessage as _EmailMsg

            msg = _EmailMsg()
            msg["To"] = recipient
            msg["From"] = from_addr
            msg["Subject"] = subject or "Quick note"
            msg.set_content(body or "")
            try:
                with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                    server.ehlo()
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                    server.send_message(msg)
                draft["delivered"] = True
                draft["smtp_host"] = smtp_host
            except Exception as exc:
                draft["delivered"] = False
                draft["smtp_error"] = str(exc)
        else:
            draft["delivered"] = False
            draft["note"] = "Draft saved. Set DADBOT_SMTP_* env vars to enable live delivery."

        return draft
