import json
import logging
import time
from pathlib import Path
from tempfile import TemporaryDirectory

logging.disable(logging.CRITICAL)

from tests.stress.phase4_certification_gate import Phase4CertificationGate, build_bot

start = time.time()
with TemporaryDirectory() as tmp:
    bot = build_bot(Path(tmp))
    gate = Phase4CertificationGate(bot)
    result = gate.run_concurrency(num_threads=50)

payload = {
    "passed": bool(result.passed),
    "score": int(result.score),
    "max_score": int(result.max_score),
    "metrics": dict(result.metrics or {}),
    "failures": list(result.failures or []),
    "risk_flags": list(result.risk_flags or []),
    "duration_s": float(result.duration_s),
    "wall_s": round(time.time() - start, 2),
}
Path("session_logs").mkdir(parents=True, exist_ok=True)
Path("session_logs/concurrency_50_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload))
