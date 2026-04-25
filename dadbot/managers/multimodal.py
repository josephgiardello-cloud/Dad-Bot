from __future__ import annotations

import logging

from dadbot.contracts import AttachmentList, DadBotContext

logger = logging.getLogger(__name__)


class MultimodalManager:
    """Owns attachment normalization and text-model image fallback analysis."""

    def __init__(self, bot: DadBotContext | object):
        self.context = DadBotContext.from_runtime(bot)
        self.bot = self.context.bot

    @staticmethod
    def normalize_chat_attachment(attachment):
        if not isinstance(attachment, dict):
            return None

        attachment_type = str(attachment.get("type") or "").strip().lower()
        if attachment_type not in {"image", "audio", "document"}:
            return None

        normalized = {
            "type": attachment_type,
            "name": str(attachment.get("name") or "").strip(),
            "mime_type": str(attachment.get("mime_type") or "").strip(),
        }

        if attachment_type == "image":
            image_b64 = str(attachment.get("image_b64") or attachment.get("data_b64") or "").strip()
            if not image_b64:
                return None
            normalized["image_b64"] = image_b64
            note = str(attachment.get("note") or "").strip()
            analysis = str(attachment.get("analysis") or "").strip()
            if note:
                normalized["note"] = note
            if analysis:
                normalized["analysis"] = analysis
            return normalized

        if attachment_type == "audio":
            transcript = str(attachment.get("transcript") or "").strip()
            note = str(attachment.get("note") or "").strip()
            if not transcript and not note:
                return None
            if transcript:
                normalized["transcript"] = transcript
            if note:
                normalized["note"] = note
            return normalized

        note = str(attachment.get("note") or "").strip()
        text_content = str(attachment.get("text") or attachment.get("content") or "").strip()
        if not note and not text_content:
            return None
        if note:
            normalized["note"] = note
        if text_content:
            normalized["text"] = text_content
        return normalized

    def normalize_chat_attachments(self, attachments: AttachmentList | None = None):
        normalized = []
        for attachment in attachments or []:
            cleaned = self.normalize_chat_attachment(attachment)
            if cleaned is not None:
                normalized.append(cleaned)
        return normalized[:6]

    @staticmethod
    def model_supports_image_input(model_name):
        lowered = str(model_name or "").strip().lower()
        if not lowered:
            return False

        multimodal_hints = (
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
        return any(hint in lowered for hint in multimodal_hints)

    def compose_user_turn_text(self, user_input, attachments: AttachmentList | None = None):
        sections = []
        primary_text = str(user_input or "").strip()
        if primary_text:
            sections.append(primary_text)

        for attachment in self.bot.normalize_chat_attachments(attachments):
            if attachment.get("type") == "image":
                note = str(attachment.get("note") or "").strip()
                analysis = str(attachment.get("analysis") or "").strip()
                sections.append(f"Photo note: {note}" if note else "Tony shared a photo in this turn.")
                if analysis:
                    sections.append(f"Photo analysis: {analysis}")
                continue

            if attachment.get("type") == "document":
                note = str(attachment.get("note") or "").strip()
                text_content = str(attachment.get("text") or "").strip()
                if note:
                    sections.append(f"Document note: {note}")
                if text_content:
                    sections.append(f"Document excerpt: {text_content}")
                continue

            transcript = str(attachment.get("transcript") or "").strip()
            note = str(attachment.get("note") or "").strip()
            if transcript:
                sections.append(f"Voice note transcript: {transcript}")
            elif note:
                sections.append(f"Voice note summary: {note}")

        return "\n\n".join(section for section in sections if section).strip()

    def build_user_request_message(self, user_input, attachments: AttachmentList | None = None):
        request_message = {"role": "user", "content": user_input}
        if self.bot.model_supports_image_input(self.bot.ACTIVE_MODEL):
            image_payload = [
                attachment["image_b64"]
                for attachment in self.bot.normalize_chat_attachments(attachments)
                if attachment.get("type") == "image" and attachment.get("image_b64")
            ]
            if image_payload:
                request_message["images"] = image_payload
        return request_message

    @staticmethod
    def history_attachment_metadata(attachment):
        metadata = {
            "type": attachment.get("type"),
            "name": attachment.get("name", ""),
            "mime_type": attachment.get("mime_type", ""),
        }
        if attachment.get("type") == "image" and attachment.get("note"):
            metadata["note"] = attachment["note"]
        if attachment.get("type") == "image" and attachment.get("analysis"):
            metadata["analysis"] = attachment["analysis"]
        if attachment.get("type") == "audio":
            if attachment.get("transcript"):
                metadata["transcript"] = attachment["transcript"]
            if attachment.get("note"):
                metadata["note"] = attachment["note"]
        if attachment.get("type") == "document":
            if attachment.get("note"):
                metadata["note"] = attachment["note"]
            if attachment.get("text"):
                metadata["text"] = attachment["text"]
        return metadata

    def build_image_analysis_prompt(self, note: str = "", user_input: str = "", attachment: dict | None = None) -> str:
        return self.bot.prompt_assembly.build_image_analysis_prompt(note=note, user_input=user_input, attachment=attachment)

    def describe_image_attachment(self, attachment, user_input=""):
        if not isinstance(attachment, dict) or attachment.get("type") != "image":
            return ""
        if self.bot.model_supports_image_input(self.bot.ACTIVE_MODEL):
            return ""
        image_b64 = str(attachment.get("image_b64") or "").strip()
        if not image_b64:
            return ""
        vision_model = self.bot.find_available_vision_model()
        if not vision_model:
            return ""

        try:
            response = self.bot.call_ollama_chat_with_model(
                vision_model,
                messages=[{
                    "role": "user",
                    "content": self.bot.build_image_analysis_prompt(
                        str(attachment.get("note") or ""),
                        user_input=user_input,
                        attachment=attachment,
                    ),
                    "images": [image_b64],
                }],
                options={"temperature": 0.1},
                purpose="image analysis",
            )
        except RuntimeError as exc:
            logger.warning("Image analysis failed: %s", exc)
            return ""

        return self.bot.extract_ollama_message_content(response).strip().strip('"')

    def enrich_multimodal_attachments(self, attachments: AttachmentList | None = None, user_input=""):
        enriched = []
        vision_ready, vision_message = self.bot.vision_fallback_status()
        for attachment in self.bot.normalize_chat_attachments(attachments):
            updated = dict(attachment)
            if updated.get("type") == "image" and not updated.get("analysis"):
                analysis = self.bot.describe_image_attachment(updated, user_input=user_input)
                if analysis:
                    updated["analysis"] = analysis
                elif not vision_ready:
                    updated["analysis"] = (
                        "Dad could not inspect the image directly. "
                        f"{vision_message}"
                    )
            enriched.append(updated)
        return enriched


__all__ = ["MultimodalManager"]
