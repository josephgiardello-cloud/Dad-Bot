"""
Benchmark certification validator: anti-inflation gate for benchmark artifacts.

Validates:
1. Artifact schema strictness (cert mode)
2. Hash drift detection (scenario/script SHA unchanged)
3. Entrant roster lock (same entrants across comparable runs)
4. Offline stub prohibition (cert runs cannot use offline LLM patch)
5. Required field presence (completed, error, response always present)

Usage:
  python tools/benchmark_cert_validator.py --baseline <scorecard.json> --candidate <scorecard.json>
  python tools/benchmark_cert_validator.py --check-manifest <run_manifest.json>
  python tools/benchmark_cert_validator.py --validate-entry <entrant_artifact.json>

Exit codes:
  0 — all validations passed (VALID)
  1 — one or more validations failed (INVALID)
  2 — validation error (bad args or file not found)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkCertError(RuntimeError):
    """Raised when a benchmark validation fails."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BenchmarkCertError(f"Failed to load JSON from {path}: {exc}") from exc


def validate_scorecard(scorecard: dict[str, Any]) -> list[str]:
    """Validate a comparative scorecard for inflation/drift markers.
    
    Returns list of validation errors (empty = valid).
    """
    errors: list[str] = []
    
    # Check for cert_mode flag.
    cert_mode = scorecard.get("cert_mode", False)
    if not isinstance(cert_mode, bool):
        errors.append(f"cert_mode must be bool, got {type(cert_mode).__name__}")
    
    # Check scenario count.
    scenario_count = scorecard.get("scenario_count", 0)
    if not isinstance(scenario_count, int) or scenario_count < 1:
        errors.append(f"scenario_count missing or invalid: {scenario_count}")
    
    # Check scenario hash presence.
    scenarios_sha = scorecard.get("scenarios_sha256", "")
    if not scenarios_sha:
        errors.append("scenarios_sha256 missing or empty")
    
    # Check script hash presence.
    script_sha = scorecard.get("script_sha256", "")
    if not script_sha:
        errors.append("script_sha256 missing or empty")
    
    # Check entrant roster for cohort lock.
    entrant_roster = scorecard.get("entrant_roster", [])
    if not isinstance(entrant_roster, list):
        errors.append(f"entrant_roster must be list, got {type(entrant_roster).__name__}")
    elif not entrant_roster:
        errors.append("entrant_roster is empty")
    
    entrant_roster_hash = scorecard.get("entrant_roster_hash", "")
    if not entrant_roster_hash:
        errors.append("entrant_roster_hash missing or empty")
    
    # Verify roster hash.
    expected_hash_str = ":".join(sorted(entrant_roster))
    import hashlib
    expected_hash = hashlib.sha256(expected_hash_str.encode()).hexdigest()
    if entrant_roster_hash != expected_hash:
        errors.append(f"entrant_roster_hash mismatch: expected {expected_hash}, got {entrant_roster_hash}")
    
    # Check entrants structure.
    entrants = scorecard.get("entrants", [])
    if not isinstance(entrants, list):
        errors.append(f"entrants must be list, got {type(entrants).__name__}")
    elif len(entrants) == 0:
        errors.append("No entrants in scorecard")
    else:
        for idx, entrant in enumerate(entrants):
            if not isinstance(entrant, dict):
                errors.append(f"Entrant {idx} is not a dict")
                continue
            if not entrant.get("name"):
                errors.append(f"Entrant {idx} missing 'name'")
            if "overall_average" not in entrant:
                errors.append(f"Entrant {idx} missing 'overall_average'")
            if "pass_rate" not in entrant:
                errors.append(f"Entrant {idx} missing 'pass_rate'")
    
    return errors


def validate_entry_artifact(artifact: dict[str, Any]) -> list[str]:
    """Validate an entrant artifact for required fields and proper structure.
    
    Returns list of validation errors (empty = valid).
    """
    errors: list[str] = []
    
    # Check cert_mode and offline_llm_stub conflict.
    cert_mode = artifact.get("cert_mode", False)
    offline_stub = artifact.get("offline_llm_stub", False)
    if cert_mode and offline_stub:
        errors.append("cert_mode and offline_llm_stub cannot both be true")
    
    # In cert mode, require git_commit_sha.
    if cert_mode and not artifact.get("git_commit_sha"):
        errors.append("cert_mode requires git_commit_sha")
    
    # Check responses list.
    responses = artifact.get("responses", [])
    if not isinstance(responses, list):
        errors.append(f"responses must be list, got {type(responses).__name__}")
    elif not responses:
        errors.append("responses list is empty")
    else:
        for idx, resp in enumerate(responses):
            if not isinstance(resp, dict):
                errors.append(f"Response {idx} is not a dict")
                continue
            
            scenario = resp.get("scenario", "")
            if not scenario:
                errors.append(f"Response {idx} missing 'scenario'")
                continue
            
            # Check required fields.
            if "completed" not in resp:
                errors.append(f"Response '{scenario}' missing 'completed' field")
            if "error" not in resp:
                errors.append(f"Response '{scenario}' missing 'error' field")
            if "response" not in resp and "final_response" not in resp:
                errors.append(f"Response '{scenario}' missing 'response' or 'final_response'")
    
    return errors


def compare_scorecards(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """Check for drift between two scorecard runs.
    
    Returns list of warnings/errors about drift (empty = no drift detected).
    """
    issues: list[str] = []
    
    # Check scenario hash consistency.
    baseline_scenarios_sha = baseline.get("scenarios_sha256", "")
    candidate_scenarios_sha = candidate.get("scenarios_sha256", "")
    if baseline_scenarios_sha and candidate_scenarios_sha and baseline_scenarios_sha != candidate_scenarios_sha:
        issues.append(
            f"Scenario suite hash drift: baseline={baseline_scenarios_sha[:8]}... "
            f"vs candidate={candidate_scenarios_sha[:8]}..."
        )
    
    # Check script hash consistency.
    baseline_script_sha = baseline.get("script_sha256", "")
    candidate_script_sha = candidate.get("script_sha256", "")
    if baseline_script_sha and candidate_script_sha and baseline_script_sha != candidate_script_sha:
        issues.append(
            f"Scorer script hash drift: baseline={baseline_script_sha[:8]}... "
            f"vs candidate={candidate_script_sha[:8]}..."
        )
    
    # Check entrant roster consistency (cohort lock).
    baseline_roster = set(baseline.get("entrant_roster", []))
    candidate_roster = set(candidate.get("entrant_roster", []))
    if baseline_roster and candidate_roster and baseline_roster != candidate_roster:
        missing = baseline_roster - candidate_roster
        extra = candidate_roster - baseline_roster
        if missing:
            issues.append(f"Entrant roster drift: missing {missing}")
        if extra:
            issues.append(f"Entrant roster drift: added {extra}")
    
    # Check offline stub consistency.
    baseline_offline = baseline.get("cert_mode", False)
    candidate_offline = candidate.get("cert_mode", False)
    if baseline_offline != candidate_offline:
        issues.append(f"cert_mode mismatch: baseline={baseline_offline} vs candidate={candidate_offline}")
    
    return issues


def check_manifest(manifest: dict[str, Any]) -> list[str]:
    """Validate a run manifest for consistency and provenance.
    
    Returns list of validation errors (empty = valid).
    """
    errors: list[str] = []
    
    # Check script hash.
    if not manifest.get("script_sha256"):
        errors.append("Manifest missing script_sha256")
    
    # Check scenarios hash.
    if not manifest.get("scenarios_sha256"):
        errors.append("Manifest missing scenarios_sha256")
    
    # Check entrant roster hash.
    if not manifest.get("entrant_roster_hash"):
        errors.append("Manifest missing entrant_roster_hash")
    
    # Check entries structure.
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        errors.append(f"entries must be list, got {type(entries).__name__}")
    else:
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                errors.append(f"Entry {idx} is not a dict")
                continue
            if not entry.get("name"):
                errors.append(f"Entry {idx} missing 'name'")
            if not entry.get("sha256"):
                errors.append(f"Entry {idx} missing 'sha256' (artifact hash)")
    
    # Check cert_mode flag.
    if "cert_mode" in manifest:
        cert = manifest["cert_mode"]
        if not isinstance(cert, bool):
            errors.append(f"cert_mode must be bool, got {type(cert).__name__}")
    
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark artifact certification validator.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--validate-entry",
        type=Path,
        help="Validate a single entrant artifact for required fields and schema.",
    )
    group.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        help="Compare two scorecard runs for hash/roster drift.",
    )
    group.add_argument(
        "--check-manifest",
        type=Path,
        help="Validate a run manifest for consistency and provenance.",
    )
    group.add_argument(
        "--validate-scorecard",
        type=Path,
        help="Validate a comparative scorecard for inflation markers.",
    )
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_args()
        
        if args.validate_entry:
            artifact = _load_json(args.validate_entry)
            errors = validate_entry_artifact(artifact)
            if errors:
                print(f"INVALID: {args.validate_entry}")
                for err in errors:
                    print(f"  ERROR: {err}")
                return 1
            print(f"VALID: {args.validate_entry} (entry artifact)")
            return 0
        
        elif args.compare:
            baseline = _load_json(Path(args.compare[0]))
            candidate = _load_json(Path(args.compare[1]))
            issues = compare_scorecards(baseline, candidate)
            if issues:
                print(f"DRIFT DETECTED comparing {args.compare[0]} vs {args.compare[1]}")
                for issue in issues:
                    print(f"  {issue}")
                return 1
            print(f"NO DRIFT: {args.compare[0]} vs {args.compare[1]} are cohesive")
            return 0
        
        elif args.check_manifest:
            manifest = _load_json(args.check_manifest)
            errors = check_manifest(manifest)
            if errors:
                print(f"INVALID: {args.check_manifest}")
                for err in errors:
                    print(f"  ERROR: {err}")
                return 1
            print(f"VALID: {args.check_manifest} (run manifest)")
            return 0
        
        elif args.validate_scorecard:
            scorecard = _load_json(args.validate_scorecard)
            errors = validate_scorecard(scorecard)
            if errors:
                print(f"INVALID: {args.validate_scorecard}")
                for err in errors:
                    print(f"  ERROR: {err}")
                return 1
            print(f"VALID: {args.validate_scorecard} (comparative scorecard)")
            return 0
        
        return 0
    
    except BenchmarkCertError as exc:
        print(f"CERT_ERROR: {exc}")
        return 2
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
