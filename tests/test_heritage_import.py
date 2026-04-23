from dadbot.heritage_import import build_heritage_memories


class UploadedStub:
    def __init__(self, name, payload):
        self.name = name
        self._payload = payload

    def getvalue(self):
        return self._payload


def test_build_heritage_memories_extracts_json_and_text_entries():
    files = [
        UploadedStub("journal.txt", b"I felt stressed at work\nI slept better after a walk"),
        UploadedStub("history.json", b"[{\"summary\": \"I moved to a new city\", \"category\": \"life\"}]"),
    ]

    result = build_heritage_memories(files, notes="Focus on patterns", max_items_per_file=10)

    memories = result["memories"]
    assert len(memories) >= 3
    assert any("journal" in item["summary"].lower() for item in memories)
    assert any(item["category"] == "life" for item in memories)
    assert any("onboarding note" in item["summary"].lower() for item in memories)


def test_build_heritage_memories_handles_mbox_payload():
    mbox_payload = (
        b"From sender@example.com Fri Jan 05 09:12:00 2024\n"
        b"Subject: Catching up\n"
        b"From: sender@example.com\n"
        b"Date: Fri, 05 Jan 2024 09:12:00 -0500\n"
        b"\n"
        b"Hey Tony, proud of your progress this week.\n"
    )

    result = build_heritage_memories([UploadedStub("mailbox.mbox", mbox_payload)], max_items_per_file=5)

    memories = result["memories"]
    assert memories
    assert memories[0]["category"] == "heritage_email"
    assert "catching up" in memories[0]["summary"].lower()
