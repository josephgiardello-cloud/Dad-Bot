from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import OrderedDict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from dadbot.memory.oral_history_evidence import verified_reviewed_evidence
from dadbot.memory.reranker import get_local_reranker, select_candidates
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
    provenance: str = ""

    @classmethod
    def create(cls, kind: str, identity: str, text: str, *, provenance: str = "") -> EvidenceSource:
        content = [kind, identity, text]
        if provenance:
            content.append(provenance)
        digest = hashlib.sha256(
            json.dumps(content, ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        return cls(f"{kind}-{digest[:16]}", identity, text, digest, provenance)


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
    """Approved extractive retrieval with optional local relevance reranking."""

    def __init__(self, bot: Any) -> None:
        self.bot = bot
        memory_path = Path(bot.memory_path)
        self.approvals = MemoryCitationApprovals(
            memory_path.with_name(f"{memory_path.stem}.citation-approvals.sqlite3"),
            tenant_id=str(getattr(getattr(bot, "config", None), "tenant_id", "default")),
        )
        self._answer_cache: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._answer_cache_fingerprint = ""
        self._answer_cache_lock = RLock()

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
                provenance = ""
                if str(record.get("category") or "").strip().lower() == "oral_history":
                    evidence = verified_reviewed_evidence(record)
                    if evidence is None:
                        continue
                    provenance = (
                        f"Reviewed oral-history transcript; evidence ID: {evidence['id']}; "
                        f"reviewed at: {evidence['reviewed_at']}; SHA-256: {evidence['sha256']}. "
                        "Original audio not retained."
                    )
                identity = json.dumps(
                    {key: record.get(key) for key in ("id", "category", "created_at", "version")},
                    sort_keys=True,
                    ensure_ascii=True,
                    default=str,
                )
                sources.append(EvidenceSource.create(section, f"{section}:{identity}", text, provenance=provenance))
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
        # Always read current eligible evidence before consulting cached answers.
        sources = self.approved_sources()
        fingerprint = hashlib.sha256(
            json.dumps({
                "sources": sorted(source.digest for source in sources),
                "vault": self.bot.memory_manager.memory_projection(),
                "write_revision": getattr(self.bot.memory_manager, "cache_invalidation_revision", 0),
                "approval_revision": self.approvals.revision(),
                "reranker_enabled": getattr(
                    getattr(self.bot, "runtime_config", None), "memory_reranker_enabled", False,
                ),
                "reranker_path": getattr(getattr(self.bot, "runtime_config", None), "memory_reranker_path", ""),
            }, sort_keys=True, ensure_ascii=True).encode("ascii")
        ).hexdigest()
        query_hash = hashlib.sha256(" ".join(query.casefold().split()).encode("utf-8")).hexdigest()
        key = (self.approvals.tenant_id, query_hash)
        with self._answer_cache_lock:
            if fingerprint != self._answer_cache_fingerprint:
                self._answer_cache.clear()
                self._answer_cache_fingerprint = fingerprint
            cached = self._answer_cache.get(key)
            if cached is not None:
                self._answer_cache.move_to_end(key)
                return cached
            reply = self._render_query_answer(query, tokens, sources)
            self._answer_cache[key] = reply
            if len(self._answer_cache) > 64:
                self._answer_cache.popitem(last=False)
            return reply

    def _render_query_answer(self, query: str, tokens: frozenset[str], sources: list[EvidenceSource]) -> str:
        config = getattr(self.bot, "runtime_config", None)
        if not getattr(config, "memory_reranker_enabled", False):
            return self._render_answer(tokens, sources)
        query = " ".join(query.casefold().split())
        scored = sorted(
            ((len(tokens & self._tokens(f"{source.label} {source.text}")) / len(tokens), source)
             for source in sources),
            key=lambda item: (-item[0], item[1].source_id),
        )
        eligible = [item for item in scored if item[0] >= 0.5]
        if not eligible:
            return SOURCE_REFUSAL
        fired, _, candidates = select_candidates(eligible, lambda source: source.source_id)
        if not fired:
            return self._render_answer(tokens, sources)
        model = get_local_reranker(config.memory_reranker_path)
        probabilities = model.score(query, [source.text for _, source in candidates])
        ranked = sorted(
            zip(probabilities, candidates, strict=True),
            key=lambda item: (-item[0], -item[1][0], item[1][1].source_id),
        )
        return self._format_matches([item for _, item in ranked[:3]])

    def _render_answer(self, tokens: frozenset[str], sources: list[EvidenceSource]) -> str:
        scored = [
            (len(tokens & self._tokens(f"{source.label} {source.text}")) / len(tokens), source)
            for source in sources
        ]
        best = max((score for score, _ in scored), default=0.0)
        if best < 0.5:
            return SOURCE_REFUSAL
        matches = sorted(
            ((score, source) for score, source in scored if score == best),
            key=lambda item: item[1].source_id,
        )[:3]
        return self._format_matches(matches)

    @staticmethod
    def _format_matches(matches: list[tuple[float, EvidenceSource]]) -> str:
        blocks = []
        for score, source in matches:
            label = "partial match" if score < 1.0 else "full token match"
            excerpt = f"Source: {source.label}\nQuote:\n{source.text}"
            if source.provenance:
                excerpt += f"\nProvenance: {source.provenance}"
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
        tokens = self._tokens(query)
        expected = SOURCE_REFUSAL
        if not attachments and tokens and any(token.isalpha() for token in tokens):
            expected = self._render_query_answer(query, tokens, self.approved_sources())
        if reply != expected:
            raise GroundingViolation("Source reply does not match the current approved quotations.")
