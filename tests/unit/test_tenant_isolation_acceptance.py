"""Application-level acceptance checks for local tenant isolation."""

import json
import subprocess
import sys

from tests.unit.test_local_tenant_establishment_application import _environment


def test_fresh_tenant_must_not_import_another_tenants_local_memory(tmp_path):
    marker = "synthetic tenant a isolation marker"
    env = _environment(tmp_path)
    tenant_a_setup = r"""
from dadbot.core.dadbot import DadBot
bot = DadBot()
assert bot._execution_ledger.replay_tenant_memory(bot.config.tenant_id).revision == 1
bot.shutdown()
"""
    env["DADBOT_TENANT_ID"] = "audit-tenant-a"
    result = subprocess.run(
        [sys.executable, "-c", tenant_a_setup],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    (tmp_path / "memory.json").write_text(
        json.dumps({"memories": [{"summary": marker}]}), encoding="utf-8"
    )
    script = r"""
from dadbot.memory.semantic_manager import SemanticIndexManager
SemanticIndexManager.embed_texts = lambda self, texts, purpose=None: [[1.0] * 12 for _ in texts]
from dadbot.core.dadbot import DadBot
bot = DadBot()
present = any("synthetic tenant a isolation marker" in str(m.get("summary", ""))
              for m in bot.MEMORY_STORE.get("memories", []))
print("ISOLATION_MARKER_PRESENT=" + str(present))
bot.shutdown()
"""
    env["DADBOT_TENANT_ID"] = "audit-tenant-b"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    matches = [line for line in result.stdout.splitlines() if line.startswith("ISOLATION_MARKER_PRESENT=")]
    assert matches == ["ISOLATION_MARKER_PRESENT=False"]
    persisted_marker = json.loads((tmp_path / "memory.json").read_text(encoding="utf-8"))["memories"][0]["summary"]
    assert marker in persisted_marker
