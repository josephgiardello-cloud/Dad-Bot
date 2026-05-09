from pathlib import Path

from tools.external_benchmark_loop import ArtifactSource, ExternalBenchmarkRunner, build_stack_includes


def test_external_benchmark_runner_produces_ranked_payload(tmp_path: Path) -> None:
    dadbot_artifact = tmp_path / "dadbot.json"
    competitor_artifact = tmp_path / "competitor.json"

    dadbot_artifact.write_text(
        """
{
  "responses": [
    {
      "scenario": "multi_step_task_decomposition",
      "response": "Define scope, budget, dependencies, then execute in sequence.",
      "completed": true,
      "tools_executed": [],
      "memory_accessed": []
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    competitor_artifact.write_text(
        """
{
  "responses": [
    {
      "scenario": "multi_step_task_decomposition",
      "response": "Just do it quickly.",
      "completed": true,
      "tools_executed": [],
      "memory_accessed": []
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    runner = ExternalBenchmarkRunner()
    payload = runner.run(
        [
            ArtifactSource(name="dadbot", path=dadbot_artifact),
            ArtifactSource(name="competitor", path=competitor_artifact),
        ]
    )

    assert "ranking" in payload
    assert len(payload["ranking"]) == 2
    assert payload["ranking"][0]["overall_average"] >= payload["ranking"][1]["overall_average"]

    entrants = payload["entrants"]
    assert len(entrants) == 2
    for entrant in entrants:
        assert entrant["scenario_count_expected"] >= entrant["scenario_count_scored"]
        assert isinstance(entrant["missing_scenarios"], list)


def test_build_stack_includes_writes_all_formats(tmp_path: Path) -> None:
    dadbot_artifact = tmp_path / "dadbot.json"
    competitor_artifact = tmp_path / "competitor.json"

    dadbot_artifact.write_text(
        """
{
  "responses": [
    {
      "scenario": "multi_step_task_decomposition",
      "response": "Plan then execute in phases.",
      "completed": true,
      "tools_executed": ["time"],
      "memory_accessed": ["semantic"]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    competitor_artifact.write_text(
        """
{
  "responses": [
    {
      "scenario": "multi_step_task_decomposition",
      "response": "Do it now.",
      "completed": false,
      "error": "timeout",
      "tools_executed": [],
      "memory_accessed": []
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    runner = ExternalBenchmarkRunner()
    payload = runner.run(
        [
            ArtifactSource(name="dadbot", path=dadbot_artifact),
            ArtifactSource(name="competitor", path=competitor_artifact),
        ]
    )

    responses = {
        "dadbot": [
            {
                "scenario": "multi_step_task_decomposition",
                "response": "Plan then execute in phases.",
                "completed": True,
                "error": None,
                "planner_output": None,
                "tools_executed": ["time"],
                "memory_accessed": ["semantic"],
            }
        ],
        "competitor": [
            {
                "scenario": "multi_step_task_decomposition",
                "response": "Do it now.",
                "completed": False,
                "error": "timeout",
                "planner_output": None,
                "tools_executed": [],
                "memory_accessed": [],
            }
        ],
    }

    bundles = build_stack_includes(payload, responses, {"swebench", "bfcl", "osworld"})

    assert "swebench_like_bundle.json" in bundles
    assert "bfcl_like_bundle.json" in bundles
    assert "osworld_ready_manifest.json" in bundles

    swebench = bundles["swebench_like_bundle.json"]
    assert swebench["format"] == "swebench-like-v1"
    assert len(swebench["leaderboard"]) == 2

    bfcl = bundles["bfcl_like_bundle.json"]
    assert bfcl["format"] == "bfcl-like-v1"
    assert isinstance(bfcl["tool_name_histogram"], dict)

    osworld = bundles["osworld_ready_manifest.json"]
    assert osworld["format"] == "osworld-ready-v1"
    assert isinstance(osworld["tasks"], list)


def test_external_benchmark_runner_populates_tool_category_profile(tmp_path: Path) -> None:
    entrant_artifact = tmp_path / "entrant.json"
    entrant_artifact.write_text(
        """
{
  "responses": [
    {
      "scenario": "correct_tool_selection",
      "response": "The current time in Tokyo is 18:32 JST.",
      "completed": true,
      "tools_executed": ["time"],
      "memory_accessed": []
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    runner = ExternalBenchmarkRunner()
    payload = runner.run([ArtifactSource(name="entrant", path=entrant_artifact)])

    entrants = payload["entrants"]
    assert len(entrants) == 1
    profile = entrants[0]["category_profile"]
    assert "tool" in profile
    assert profile["tool"] > 0.0
