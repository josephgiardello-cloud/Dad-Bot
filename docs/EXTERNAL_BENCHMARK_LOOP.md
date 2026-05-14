# External Benchmark Loop

This lane compares Dad-Bot and external competitors on the same scenario suite and writes reproducible artifacts (JSON + optional Markdown) with input/output hashes.

## Command

```powershell
python tools/external_benchmark_loop.py \
  --entry dadbot=path/to/dadbot_outputs.json \
  --entry competitor_a=path/to/competitor_a_outputs.json \
  --entry competitor_b=path/to/competitor_b_outputs.json \
  --include-stack all \
  --write-markdown
```

Optional:

- `--output-dir artifacts/external_benchmark/my_run`
- `--include-stack swebench`
- `--include-stack bfcl`
- `--include-stack osworld`
- `--include-stack all`

Default output directory:

- `artifacts/external_benchmark/external_benchmark_<UTC timestamp>/`

## Input Artifact Schema

Each `--entry` points to a JSON file with this shape:

```json
{
  "agent": "dadbot",
  "model": "llama3.2:latest",
  "responses": [
    {
      "scenario": "multi_step_task_decomposition",
      "response": "...",
      "completed": true,
      "error": null,
      "planner_output": {"steps": 4},
      "tools_executed": ["time"],
      "memory_accessed": ["semantic"]
    }
  ]
}
```

Accepted aliases:

- `final_response` instead of `response`
- `tool_calls` instead of `tools_executed`
- `memory_reads` instead of `memory_accessed`

If an entry is missing a scenario, the run is still scored and the scenario is marked missing for that entrant.

## Outputs

- `comparative_scorecard.json`: full ranking + per-entrant scores + hashes
- `run_manifest.json`: immutable run metadata and input/output hashes
- `comparative_scorecard.md` (when `--write-markdown`): human-readable summary

When `--include-stack ...` is used, additional files are written to:

- `stack_includes/swebench_like_bundle.json`
- `stack_includes/bfcl_like_bundle.json`
- `stack_includes/osworld_ready_manifest.json`

These are lane-targeted export bundles designed to keep one benchmark run reusable across your coding-agent, tool-use, and computer-use submission pipelines.

## Included Examples

- `artifacts/external_benchmark_examples/dadbot.sample.json`
- `artifacts/external_benchmark_examples/competitor.sample.json`

Example smoke run:

```powershell
python tools/external_benchmark_loop.py \
  --entry dadbot=artifacts/external_benchmark_examples/dadbot.sample.json \
  --entry competitor=artifacts/external_benchmark_examples/competitor.sample.json \
  --write-markdown
```
