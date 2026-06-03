"""Tests for benchmarks/render_readme.py — auto-generated README results section.

The README's results tables and charts are GENERATED from the benchmark result
JSONs (single source of truth), so the README can never drift from the data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))


@pytest.fixture()
def results_dirs(tmp_path):
    """Minimal fixture result files mirroring the real layout."""
    rebuild = tmp_path / "rebuild" / "results"
    orchestration = tmp_path / "orchestration" / "results"
    phase0 = tmp_path / "phase0" / "results"
    for d in (rebuild, orchestration, phase0):
        d.mkdir(parents=True)

    (rebuild / "feasibility_after_rebuild.json").write_text(
        json.dumps(
            {
                "measurements": [
                    {
                        "planner": "mock",
                        "critic_survivors": 6,
                        "physically_feasible": 6,
                        "feasibility_of_survivors": 1.0,
                        "gpu_wall_s": 12.4,
                    },
                    {
                        "planner": "claude",
                        "critic_survivors": 8,
                        "physically_feasible": 8,
                        "feasibility_of_survivors": 1.0,
                        "gpu_wall_s": 26.6,
                    },
                ]
            }
        )
    )
    (orchestration / "orchestration_eval.json").write_text(
        json.dumps(
            {
                "reports": [
                    {
                        "planner_name": "mock",
                        "metrics": {
                            "end_to_end_success_rate": 0.6,
                            "infeasible_recognition_rate": 1.0,
                            "recovery_rate": 1.0,
                            "decomposition_validity_rate": 1.0,
                            "invalid_plan_catch_rate": 1.0,
                            "orchestrator_score": 0.92,
                        },
                    },
                    {
                        "planner_name": "llm",
                        "metrics": {
                            "end_to_end_success_rate": 1.0,
                            "infeasible_recognition_rate": 1.0,
                            "recovery_rate": 1.0,
                            "decomposition_validity_rate": 1.0,
                            "invalid_plan_catch_rate": 1.0,
                            "orchestrator_score": 1.0,
                        },
                    },
                ]
            }
        )
    )
    (phase0 / "benchmark_3_gpu_saturation.json").write_text(
        json.dumps(
            {
                "results": [
                    {"num_envs": 1, "env_steps_per_s": 281.4, "scaling_efficiency": 1.0},
                    {"num_envs": 1024, "env_steps_per_s": 250227.2, "scaling_efficiency": 0.868},
                ],
                "saturation_knee_envs": None,
            }
        )
    )
    return tmp_path


def test_results_section_contains_data_from_jsons(results_dirs):
    from render_readme import build_results_section

    section = build_results_section(results_dirs)
    # feasibility numbers
    assert "6/6" in section and "8/8" in section
    # orchestration scores
    assert "0.92" in section
    # GPU saturation
    assert "1024" in section
    # mermaid charts render dynamically on GitHub
    assert "```mermaid" in section
    assert "xychart-beta" in section


def test_results_section_shows_pre_rebuild_artifact_contrast(results_dirs):
    """The before/after framing must be present: 17% (artifact) vs 100%."""
    from render_readme import build_results_section

    section = build_results_section(results_dirs)
    assert "17%" in section
    assert "100%" in section


def test_inject_replaces_marked_block_idempotently(results_dirs, tmp_path):
    from render_readme import RESULTS_BEGIN, RESULTS_END, inject_section

    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\nintro\n\n{RESULTS_BEGIN}\nold content\n{RESULTS_END}\n\nfooter\n")

    inject_section(readme, "NEW CONTENT", RESULTS_BEGIN, RESULTS_END)
    first = readme.read_text()
    assert "NEW CONTENT" in first
    assert "old content" not in first
    assert "intro" in first and "footer" in first

    # idempotent: running again with the same content changes nothing
    inject_section(readme, "NEW CONTENT", RESULTS_BEGIN, RESULTS_END)
    assert readme.read_text() == first


def test_inject_fails_loudly_when_markers_missing(tmp_path):
    from render_readme import RESULTS_BEGIN, RESULTS_END, inject_section

    readme = tmp_path / "README.md"
    readme.write_text("# Title\nno markers here\n")
    with pytest.raises(ValueError, match="marker"):
        inject_section(readme, "content", RESULTS_BEGIN, RESULTS_END)


def test_missing_result_file_produces_explicit_placeholder(tmp_path):
    """A missing benchmark file must not crash or silently vanish — the section
    says explicitly that the result is not available."""
    from render_readme import build_results_section

    empty = tmp_path / "nothing"
    empty.mkdir()
    section = build_results_section(empty)
    assert "not available" in section.lower()
