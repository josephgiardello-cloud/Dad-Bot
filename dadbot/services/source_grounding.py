from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dadbot.utils import SIGNIFICANT_TOKEN_STOPWORDS, tokenize_text

SOURCE_REFUSAL = (
    "I don't have a relevant approved source for that. "
    "Jor-El can only quote the profile and explicitly approved memories."
)
SOURCE_SAFETY_REFUSAL = "I can't show that source excerpt safely."
_PROFILE_SECTIONS = frozenset({
    "family", "education", "sports_preferences", "facts", "biography",
    "background", "interests", "hobbies", "values", "stories", "work",
})
_PRIVATE_KEYS = frozenset({
    "security", "password", "secret", "token", "api_key", "credentials",
    "pin", "streamlit_pin", "streamlit_pin_hash",
})


class GroundingViolation(RuntimeError):
    """A source-mode response no longer matches the approved evidence."""


@dataclass(frozen=True)
class EvidenceSource:
    source_id: str
    label: str
    text: str
    digest: str

    @classmethod
    def create(cls, kind: str, identity: str, text: str) -> EvidenceSource:
        digest = hashlib.sha256(
            json.dumps([kind, identity, text], ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        return cls(f"{kind}-{digest[:16]}", identity, text, digest)


class MemoryCitationApprovals:
    """Human citation consent, separate from model-writable memory content."""

    def __init__(self, path: Path, *, tenant_id: str) -> None:
        self.path = path
        self.tenant_id = tenant_id

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0)
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS citation_approvals "
                "(tenant_id TEXT NOT NULL, source_id TEXT NOT NULL, digest TEXT NOT NULL, "
                "PRIMARY KEY (tenant_id, source_id))"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS citation_approval_revisions "
                "(tenant_id TEXT PRIMARY KEY, revision INTEGER NOT NULL)"
            )
            connection.commit()
        except sqlite3.Error:
            connection.close()
            raise
        return connection

    def approved_digests(self) -> dict[str, str]:
        with closing(self._connect()) as connection:
            return dict(connection.execute(
                "SELECT source_id, digest FROM citation_approvals WHERE tenant_id = ?",
                (self.tenant_id,),
            ))

    def set_approval(self, source: EvidenceSource, *, approved: bool) -> None:
        with closing(self._connect()) as connection, connection:
            if approved:
                connection.execute(
                    "INSERT OR REPLACE INTO citation_approvals VALUES (?, ?, ?)",
                    (self.tenant_id, source.source_id, source.digest),
                )
            else:
                connection.execute(
                    "DELETE FROM citation_approvals WHERE tenant_id = ? AND source_id = ?",
                    (self.tenant_id, source.source_id),
                )
            connection.execute(
                "INSERT INTO citation_approval_revisions VALUES (?, 1) "
                "ON CONFLICT(tenant_id) DO UPDATE SET revision = revision + 1",
                (self.tenant_id,),
            )

    def revision(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT revision FROM citation_approval_revisions WHERE tenant_id = ?",
                (self.tenant_id,),
            ).fetchone()
            return int(row[0]) if row is not None else 0


class SourceGroundingService:
    """Deterministic, extractive retrieval; never calls a model or a tool."""

    def __init__(self, bot: Any) -> None:
        self.bot = bot
        memory_path = Path(bot.memory_path)
        self.approvals = MemoryCitationApprovals(
            memory_path.with_name(f"{memory_path.stem}.citation-approvals.sqlite3"),
            tenant_id=str(getattr(getattr(bot, "config", None), "tenant_id", "default")),
        )

    @staticmethod
    def _profile_leaves(value: Any, path: str) -> list[EvidenceSource]:
        if isinstance(value, dict):
            sources = []
            for key, child in sorted(value.items()):
                normalized = str(key).lower().replace("-", "_")
                if normalized in _PRIVATE_KEYS or any(
                    marker in normalized for marker in ("password", "secret", "token", "api_key")
                ):
                    continue
                pointer = str(key).replace("~", "~0").replace("/", "~1")
                sources.extend(SourceGroundingService._profile_leaves(child, f"{path}/{pointer}"))
            return sources
        if isinstance(value, list):
            return [
                source
                for index, child in enumerate(value)
                for source in SourceGroundingService._profile_leaves(child, f"{path}/{index}")
            ]
        if value is None or value == "":
            return []
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return [EvidenceSource.create("profile", path, text)]

    def profile_sources(self) -> list[EvidenceSource]:
        profile = self.bot.PROFILE
        sources = []
        for section in sorted(_PROFILE_SECTIONS):
            if section in profile:
                sources.extend(self._profile_leaves(profile[section], f"/{section}"))
        style = profile.get("style") or {}
        for field in ("name", "listener_name", "tone", "signoff"):
            if field in style:
                sources.extend(self._profile_leaves(style[field], f"/style/{field}"))
        return sources

    def memory_sources(self) -> list[EvidenceSource]:
        sources = []
        store = self.bot.memory_manager.memory_projection()
        for section in ("memories", "consolidated_memories"):
            for record in store.get(section) or []:
                if not isinstance(record, dict) or record.get("superseded"):
                    continue
                text = record.get("summary")
                if not isinstance(text, str) or not text.strip():
                    continue
                identity = json.dumps(
                    {key: record.get(key) for key in ("id", "category", "created_at", "version")},
                    sort_keys=True,
                    ensure_ascii=True,
                    default=str,
                )
                sources.append(EvidenceSource.create(section, f"{section}:{identity}", text))
        return list({source.source_id: source for source in sources}.values())

    def set_memory_approval(self, source: EvidenceSource, *, approved: bool) -> None:
        current = {item.source_id: item for item in self.memory_sources()}.get(source.source_id)
        if current is None or current != source:
            raise GroundingViolation("Memory changed since review; reload and review it again.")
        self.approvals.set_approval(current, approved=approved)

    def approved_sources(self) -> list[EvidenceSource]:
        approved = self.approvals.approved_digests()
        return self.profile_sources() + [
            source for source in self.memory_sources()
            if approved.get(source.source_id) == source.digest
        ]

    def corpus_fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps({
                "sources": sorted(
                    source.digest for source in self.profile_sources() + self.memory_sources()
                ),
                "approval_revision": self.approvals.revision(),
            }, sort_keys=True).encode("ascii")
        ).hexdigest()

    @staticmethod
    def _tokens(text: str) -> frozenset[str]:
        return frozenset(
            token for token in tokenize_text(text.replace("_", " "))
            if len(token) > 1 and token not in SIGNIFICANT_TOKEN_STOPWORDS
        )

    def answer(self, query: str, *, attachments: Any = None) -> str:
        tokens = self._tokens(query)
        if attachments or not tokens or not any(token.isalpha() for token in tokens):
            return SOURCE_REFUSAL
        scored = [
            (len(tokens & self._tokens(f"{source.label} {source.text}")) / len(tokens), source)
            for source in self.approved_sources()
        ]
        best = max((score for score, _ in scored), default=0.0)
        if best < 0.5:
            return SOURCE_REFUSAL
        matches = sorted(
            ((score, source) for score, source in scored if score == best),
            key=lambda item: item[1].source_id,
        )[:3]
        blocks = []
        for score, source in matches:
            label = "partial match" if score < 1.0 else "full token match"
            excerpt = f"Source: {source.label}\nQuote:\n{source.text}"
            blocks.append(
                f"[{source.source_id}] - Retrieval match strength: {score:.0%} ({label})\n\n"
                + "\n".join(f"    {line}" for line in excerpt.splitlines())
            )
        return (
            "Source excerpts:\n\n" + "\n\n".join(blocks)
            + "\n\nMatch strength measures query-token overlap, not probability of truth."
        )

    def validate_reply(self, query: str, reply: str, *, attachments: Any = None) -> None:
        if reply == SOURCE_SAFETY_REFUSAL:
            return
        if reply != self.answer(query, attachments=attachments):
            raise GroundingViolation("Source reply does not match the current approved quotations.")
