from __future__ import annotations

import json
import mailbox
import tempfile
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

try:
    from PIL import ExifTags, Image  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    ExifTags = None
    Image = None


def _safe_text(value) -> str:
    return str(value or "").strip()


def _clip(value: str, limit: int = 220) -> str:
    text = _safe_text(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _normalize_uploaded_bytes(uploaded_file):
    if uploaded_file is None:
        return b""
    if hasattr(uploaded_file, "getvalue"):
        try:
            return uploaded_file.getvalue() or b""
        except Exception:
            return b""
    if hasattr(uploaded_file, "read"):
        try:
            return uploaded_file.read() or b""
        except Exception:
            return b""
    return b""


def _parse_json_memories(name: str, payload: object, limit: int):
    entries = []
    flattened = []
    if isinstance(payload, list):
        flattened = payload
    elif isinstance(payload, dict):
        # Prefer common export roots first.
        for key in ("messages", "chat", "conversations", "items", "memories"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                flattened = candidate
                break
        if not flattened:
            flattened = [payload]

    for item in flattened[: max(1, int(limit or 1))]:
        if isinstance(item, dict):
            summary = _safe_text(item.get("summary") or item.get("text") or item.get("content") or item.get("message"))
            category = _safe_text(item.get("category") or "heritage") or "heritage"
            if summary:
                entries.append({"summary": _clip(f"Heritage note from {name}: {summary}"), "category": category})
        elif isinstance(item, str):
            cleaned = _safe_text(item)
            if cleaned:
                entries.append({"summary": _clip(f"Heritage note from {name}: {cleaned}"), "category": "heritage"})
    return entries


def _parse_text_memories(name: str, text: str, limit: int):
    entries = []
    lines = [line.strip() for line in str(text or "").splitlines()]
    chunks = [line for line in lines if line]
    for line in chunks[: max(1, int(limit or 1))]:
        entries.append({"summary": _clip(f"Heritage journal from {name}: {line}"), "category": "heritage"})
    return entries


def _decode_email_snippet(message_obj):
    if message_obj is None:
        return ""
    if message_obj.is_multipart():
        for part in message_obj.walk():
            content_type = str(part.get_content_type() or "").lower()
            if content_type != "text/plain":
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            charset = part.get_content_charset() or "utf-8"
            try:
                return payload.decode(charset, errors="replace").strip()
            except Exception:
                return payload.decode("utf-8", errors="replace").strip()
        return ""

    payload = message_obj.get_payload(decode=True)
    if payload is None:
        return ""
    charset = message_obj.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace").strip()
    except Exception:
        return payload.decode("utf-8", errors="replace").strip()


def _parse_mbox_memories(name: str, content_bytes: bytes, limit: int):
    if not content_bytes:
        return []
    entries = []
    with tempfile.TemporaryDirectory() as temp_dir:
        mbox_path = Path(temp_dir) / "heritage_import.mbox"
        mbox_path.write_bytes(content_bytes)
        box = mailbox.mbox(mbox_path)
        try:
            for index, message_obj in enumerate(box):
                if index >= max(1, int(limit or 1)):
                    break
                subject = _safe_text(message_obj.get("subject") or "(no subject)")
                sender = _safe_text(message_obj.get("from") or "unknown sender")
                sent_at = _safe_text(message_obj.get("date"))
                sent_label = ""
                if sent_at:
                    try:
                        sent_label = parsedate_to_datetime(sent_at).strftime("%Y-%m-%d")
                    except Exception:
                        sent_label = sent_at[:16]
                snippet = _decode_email_snippet(message_obj)
                summary = f"Email import from {name}: {subject}"
                details = f"Sender: {sender}."
                if sent_label:
                    details += f" Date: {sent_label}."
                if snippet:
                    details += f" Note: {_clip(snippet, limit=120)}"
                entries.append({"summary": _clip(f"{summary}. {details}"), "category": "heritage_email"})
        finally:
            box.close()
    return entries


def _parse_photo_memory(name: str, content_bytes: bytes):
    if not content_bytes:
        return []
    if Image is None:
        return [{"summary": _clip(f"Photo added to heritage import: {name}."), "category": "heritage_photo"}]

    created = ""
    width = 0
    height = 0

    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix or ".img", delete=True) as handle:
        handle.write(content_bytes)
        handle.flush()
        with Image.open(handle.name) as image:
            width, height = image.size
            try:
                exif = image.getexif() or {}
            except Exception:
                exif = {}
            if exif and ExifTags is not None:
                reverse = {value: key for key, value in ExifTags.TAGS.items()}
                for exif_key in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
                    exif_value = exif.get(reverse.get(exif_key))
                    if exif_value:
                        created = _safe_text(exif_value)
                        break

    detail = f"Photo added to heritage import: {name} ({width}x{height})."
    if created:
        detail += f" Captured: {created}."
    return [{"summary": _clip(detail), "category": "heritage_photo"}]


def build_heritage_memories(uploaded_files, *, notes: str = "", max_items_per_file: int = 40):
    memories = []
    stats = {
        "files": 0,
        "memories": 0,
        "failed": 0,
        "by_type": {},
    }
    extra_notes = _safe_text(notes)

    for uploaded_file in list(uploaded_files or []):
        file_name = _safe_text(getattr(uploaded_file, "name", "upload")) or "upload"
        suffix = Path(file_name).suffix.lower()
        content = _normalize_uploaded_bytes(uploaded_file)
        if not content:
            continue
        stats["files"] += 1
        stats["by_type"][suffix or "(no-ext)"] = int(stats["by_type"].get(suffix or "(no-ext)", 0)) + 1

        try:
            if suffix in {".json"}:
                payload = json.loads(content.decode("utf-8", errors="replace"))
                memories.extend(_parse_json_memories(file_name, payload, max_items_per_file))
            elif suffix in {".txt", ".md", ".csv", ".log"}:
                text = content.decode("utf-8", errors="replace")
                memories.extend(_parse_text_memories(file_name, text, max_items_per_file))
            elif suffix in {".mbox"}:
                memories.extend(_parse_mbox_memories(file_name, content, max_items_per_file))
            elif suffix in {".jpg", ".jpeg", ".png", ".webp"}:
                memories.extend(_parse_photo_memory(file_name, content))
            else:
                text = content.decode("utf-8", errors="replace")
                memories.extend(_parse_text_memories(file_name, text, max_items_per_file // 2 or 1))
        except Exception:
            stats["failed"] = int(stats.get("failed", 0)) + 1

    if extra_notes:
        memories.append(
            {
                "summary": _clip(f"Heritage onboarding note: {extra_notes}"),
                "category": "heritage",
            }
        )

    dedup = []
    seen = set()
    for entry in memories:
        summary = _safe_text(entry.get("summary"))
        category = _safe_text(entry.get("category") or "heritage") or "heritage"
        if not summary:
            continue
        key = (summary.lower(), category.lower())
        if key in seen:
            continue
        seen.add(key)
        dedup.append({"summary": summary, "category": category})

    stats["memories"] = len(dedup)
    return {
        "memories": dedup,
        "stats": stats,
    }


__all__ = ["build_heritage_memories"]
