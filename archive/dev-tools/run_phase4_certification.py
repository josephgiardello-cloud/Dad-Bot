"""Phase 4 Certification Gate — entry point.

Run from the project root:

    python run_phase4_certification.py

Optional environment variables:
    PHASE4_LONG_HORIZON_TURNS  (default 200)
    PHASE4_MEMORY_GROWTH_TURNS (default 200)
    PHASE4_REPLAY_TURNS        (default 100)
    PHASE4_CONCURRENCY_THREADS (default 20)
    PHASE4_REPORT_PATH         (default session_logs/phase4_certification_report.json)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase4_certification")


def main() -> int:
    from tests.stress.phase4_certification_gate import Phase4CertificationGate, build_bot

    long_horizon_turns = int(os.environ.get("PHASE4_LONG_HORIZON_TURNS", "200") or 200)
    memory_growth_turns = int(os.environ.get("PHASE4_MEMORY_GROWTH_TURNS", "200") or 200)
    replay_turns = int(os.environ.get("PHASE4_REPLAY_TURNS", "100") or 100)
    concurrency_threads = int(os.environ.get("PHASE4_CONCURRENCY_THREADS", "20") or 20)
    report_path_str = os.environ.get("PHASE4_REPORT_PATH", "session_logs/phase4_certification_report.json")
    report_path = ROOT / report_path_str

    report_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 64)
    logger.info("Phase 4 Certification Gate")
    logger.info("  long_horizon_turns   = %d", long_horizon_turns)
    logger.info("  memory_growth_turns  = %d", memory_growth_turns)
    logger.info("  replay_turns         = %d", replay_turns)
    logger.info("  concurrency_threads  = %d", concurrency_threads)
    logger.info("=" * 64)

    t0 = time.monotonic()

    with TemporaryDirectory() as tmp:
        logger.info("Building bot in isolated temp directory: %s", tmp)
        bot = build_bot(Path(tmp))
        try:
            gate = Phase4CertificationGate(bot)
            report = gate.run_all(
                long_horizon_turns=long_horizon_turns,
                memory_growth_turns=memory_growth_turns,
                replay_turns=replay_turns,
                concurrency_threads=concurrency_threads,
            )
        finally:
            try:
                bot.shutdown()
            except Exception as exc:
                logger.warning("Bot shutdown raised: %s", exc)

    elapsed = time.monotonic() - t0
    report["wall_time_s"] = round(elapsed, 2)

    # Write report
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Report written to: %s", report_path)

    # Print summary
    logger.info("=" * 64)
    logger.info("CERTIFICATION: %s", report["phase4_certification"])
    logger.info("SCORE:         %d / 100  (threshold %d)", report["score"], report["pass_threshold"])
    logger.info("WALL TIME:     %.1fs", elapsed)
    logger.info("")

    module_results = report.get("results", {})
    for mod_name, mod in module_results.items():
        status = "PASS" if mod["passed"] else "FAIL"
        logger.info("  %-20s %s  %2d/%-2d  %.1fs", mod_name, status, mod["score"], mod["max_score"], mod["duration_s"])

    if report["failures"]:
        logger.info("")
        logger.info("Failures (%d):", len(report["failures"]))
        for f in report["failures"][:20]:
            logger.info("  ✗ %s", f)
        if len(report["failures"]) > 20:
            logger.info("  ... and %d more (see report)", len(report["failures"]) - 20)

    if report["risk_flags"]:
        logger.info("")
        logger.info("Risk flags (%d):", len(report["risk_flags"]))
        for rf in report["risk_flags"][:10]:
            logger.info("  ⚠ %s", rf)

    logger.info("=" * 64)

    return 0 if report["phase4_certification"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
