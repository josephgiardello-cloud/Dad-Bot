from __future__ import annotations

import base64
import mimetypes

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def summarize_document_attachment(uploaded_file, max_chars: int = 420):
    if uploaded_file is None:
        return None
    raw_bytes = b""
    try:
        raw_bytes = uploaded_file.getvalue() or b""
    except Exception:
        raw_bytes = b""
    if not raw_bytes:
        return None

    name = str(getattr(uploaded_file, "name", "document")).strip() or "document"
    mime_type = str(getattr(uploaded_file, "type", "")).strip() or (mimetypes.guess_type(name)[0] or "")
    text_excerpt = ""
    if mime_type.startswith("text/") or name.lower().endswith((".txt", ".md", ".csv", ".json", ".py", ".log")):
        text_excerpt = raw_bytes.decode("utf-8", errors="replace").strip()
    elif name.lower().endswith(".pdf"):
        text_excerpt = f"PDF uploaded ({len(raw_bytes)} bytes)."
    else:
        text_excerpt = f"File uploaded ({len(raw_bytes)} bytes)."

    cleaned = " ".join(text_excerpt.split())
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 3].rstrip() + "..."
    if not cleaned:
        cleaned = f"File uploaded ({len(raw_bytes)} bytes)."

    return {
        "type": "document",
        "name": name,
        "mime_type": mime_type,
        "note": f"Tony shared a document: {name}",
        "text": cleaned,
    }


def build_chat_attachments_from_uploads(uploaded_files, *, max_upload_bytes: int = MAX_UPLOAD_BYTES):
    attachments = []
    issues = []
    for uploaded_file in list(uploaded_files or []):
        if uploaded_file is None:
            continue
        file_name = str(getattr(uploaded_file, "name", "upload")).strip().lower()
        mime_type = str(getattr(uploaded_file, "type", "")).strip().lower()
        raw_bytes = b""
        try:
            raw_bytes = uploaded_file.getvalue() or b""
        except Exception as exc:
            raw_bytes = b""
            issues.append(f"{getattr(uploaded_file, 'name', 'upload')}: could not read file ({exc}).")
        if not raw_bytes:
            issues.append(f"{getattr(uploaded_file, 'name', 'upload')}: file was empty.")
            continue
        if len(raw_bytes) > max_upload_bytes:
            issues.append(
                f"{getattr(uploaded_file, 'name', 'upload')}: file is larger than {max_upload_bytes // (1024 * 1024)}MB limit."
            )
            continue

        is_image = mime_type.startswith("image/") or file_name.endswith((".png", ".jpg", ".jpeg", ".webp"))
        if is_image:
            attachments.append(
                {
                    "type": "image",
                    "name": str(getattr(uploaded_file, "name", "image")),
                    "mime_type": mime_type,
                    "image_b64": base64.b64encode(raw_bytes).decode("utf-8"),
                    "note": f"Tony uploaded {str(getattr(uploaded_file, 'name', 'an image'))}",
                }
            )
            continue

        summarized = summarize_document_attachment(uploaded_file)
        if summarized is not None:
            attachments.append(summarized)
        else:
            issues.append(f"{getattr(uploaded_file, 'name', 'upload')}: unsupported or unreadable document.")

    return attachments[:6], issues