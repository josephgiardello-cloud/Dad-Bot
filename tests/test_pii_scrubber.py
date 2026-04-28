"""Tests for dadbot.pii_scrubber — PII detection and redaction."""
from __future__ import annotations

import pytest
pytestmark = pytest.mark.unit
from dadbot.pii_scrubber import contains_pii, scrub_memory_entry, scrub_memory_list, scrub_text


# ─── scrub_text ───────────────────────────────────────────────────────────────

class TestScrubText:
    def test_clean_text_passes_through_unchanged(self):
        text = "Tony loves hiking and building robots."
        result, found = scrub_text(text)
        assert result == text
        assert found == []

    def test_detects_and_redacts_email(self):
        text = "Reach me at tony@example.com if you need anything."
        result, found = scrub_text(text)
        assert "tony@example.com" not in result
        assert "[EMAIL REDACTED]" in result
        assert "email" in found

    def test_detects_and_redacts_us_phone(self):
        text = "Call me at 555-867-5309 anytime."
        result, found = scrub_text(text)
        assert "555-867-5309" not in result
        assert "[PHONE REDACTED]" in result
        assert "phone" in found

    def test_detects_and_redacts_ssn(self):
        text = "My SSN is 123-45-6789."
        result, found = scrub_text(text)
        assert "123-45-6789" not in result
        assert "[SSN REDACTED]" in result
        assert "ssn" in found

    def test_detects_and_redacts_credit_card(self):
        text = "Card number: 4111 1111 1111 1111 please charge it."
        result, found = scrub_text(text)
        assert "4111" not in result
        assert "[CARD REDACTED]" in result
        assert "credit_card" in found

    def test_detects_multiple_pii_types_in_one_string(self):
        text = "Email tony@test.com or call 555-123-4567."
        result, found = scrub_text(text)
        assert "tony@test.com" not in result
        assert "555-123-4567" not in result
        assert "email" in found
        assert "phone" in found

    def test_empty_string_returns_unchanged(self):
        result, found = scrub_text("")
        assert result == ""
        assert found == []

    def test_none_like_empty_returns_unchanged(self):
        # scrub_text expects str; ensure it handles falsy gracefully
        result, found = scrub_text("")
        assert found == []


# ─── scrub_memory_entry ───────────────────────────────────────────────────────

class TestScrubMemoryEntry:
    def test_clean_entry_unchanged(self):
        entry = {"summary": "Tony enjoys cooking pasta.", "category": "hobbies"}
        result = scrub_memory_entry(entry)
        assert result["summary"] == "Tony enjoys cooking pasta."
        assert "_pii_scrubbed" not in result

    def test_email_in_summary_is_redacted(self):
        entry = {"summary": "Contact at user@domain.com for updates.", "category": "contacts"}
        result = scrub_memory_entry(entry)
        assert "user@domain.com" not in result["summary"]
        assert "email" in result.get("_pii_scrubbed", [])

    def test_non_dict_entry_returned_unchanged(self):
        assert scrub_memory_entry("not a dict") == "not a dict"
        assert scrub_memory_entry(None) is None

    def test_scrub_detail_field(self):
        entry = {"summary": "nothing here", "detail": "ssn 321-54-9876 on file"}
        result = scrub_memory_entry(entry)
        assert "321-54-9876" not in result["detail"]
        assert "ssn" in result.get("_pii_scrubbed", [])

    def test_pii_scrubbed_tag_accumulates_across_fields(self):
        entry = {
            "summary": "email: a@b.com",
            "detail": "phone: 555-000-1234",
        }
        result = scrub_memory_entry(entry)
        tags = result.get("_pii_scrubbed", [])
        assert "email" in tags
        assert "phone" in tags


# ─── scrub_memory_list ────────────────────────────────────────────────────────

class TestScrubMemoryList:
    def test_empty_list(self):
        assert scrub_memory_list([]) == []

    def test_list_of_clean_entries(self):
        entries = [{"summary": "likes cats"}, {"summary": "plays guitar"}]
        result = scrub_memory_list(entries)
        assert all("_pii_scrubbed" not in r for r in result)

    def test_list_scrubs_affected_entries_only(self):
        entries = [
            {"summary": "likes pizza"},
            {"summary": "email: x@y.com"},
        ]
        result = scrub_memory_list(entries)
        assert "_pii_scrubbed" not in result[0]
        assert "email" in result[1].get("_pii_scrubbed", [])


# ─── contains_pii ─────────────────────────────────────────────────────────────

class TestContainsPii:
    def test_clean_text_false(self):
        assert contains_pii("Tony likes basketball.") is False

    def test_email_detected(self):
        assert contains_pii("hit me at user@example.org") is True

    def test_phone_detected(self):
        assert contains_pii("ring me at (800) 555-1212") is True
