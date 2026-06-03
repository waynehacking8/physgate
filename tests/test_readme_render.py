"""Tests for benchmarks/render_readme.py — auto-generated README results section.

The README's results section is GENERATED from the benchmark result JSONs
(single source of truth): publication-quality PNG charts (via physgate.viz.charts)
+ markdown tables. The README can never drift from the data.
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
    eval_v2 = tmp_path / "eval_v2" / "results"
    for d in (rebuild, orchestration, phase0, eval_v2):
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
    (eval_v2 / "ablation.json").write_text(
        json.dumps(
            {
                "conditions": {
                    "A0_no_validation": {
                        "success_rate": 0.43,
                        "success_ci_95": [0.38, 0.48],
                        "false_execution_rate": 1.0,
                        "rejection_rate": 0.0,
                    },
                    "A3_nav_aware_gate": {
                        "success_rate": 1.0,
                        "success_ci_95": [0.84, 1.0],
                        "false_execution_rate": 0.0,
                        "rejection_rate": 1.0,
                    },
                },
                "config": {"n_feasible_instances": 20, "n_infeasible_instances": 8},
            }
        )
    )
    (eval_v2 / "gate_classifier.json").write_text(
        json.dumps(
            {
                "layers": {
                    "critic": {
                        "report": {
                            "precision": 1.0,
                            "recall": 0.25,
                            "f1": 0.4,
                            "false_positive_rate": 0.0,
                        },
                        "per_defect_rejection_rate": {"D0_clean": 0.0},
                        "wall_s": 0.0,
                    },
                    "physics_gate": {
                        "report": {
                            "precision": 1.0,
                            "recall": 1.0,
                            "f1": 1.0,
                            "false_positive_rate": 0.0,
                        },
                        "per_defect_rejection_rate": {"D0_clean": 0.0},
                        "wall_s": 76.4,
                    },
                },
            }
        )
    )
    return tmp_path


def test_results_section_leads_with_ablation_and_classifier(results_dirs, tmp_path):
    """E1 (ablation) and E2 (classifier) are the headline evidence, before the
    legacy feasibility/orchestration sections."""
    from render_readme import build_results_section

    section = build_results_section(results_dirs, charts_dir=tmp_path / "charts")
    e1_pos = section.find("Pipeline ablation")
    e2_pos = section.find("gate as a classifier")
    feasibility_pos = section.find("artifact is eliminated")
    assert e1_pos != -1 and e2_pos != -1
    assert e1_pos < feasibility_pos, "E1 must come before the legacy feasibility section"


def test_results_section_embeds_chart_images_not_mermaid(results_dirs, tmp_path):
    """Charts are publication-quality PNGs (matplotlib), not Mermaid blocks."""
    from render_readme import build_results_section

    charts_dir = tmp_path / "charts"
    section = build_results_section(results_dirs, charts_dir=charts_dir)
    assert "```mermaid" not in section
    assert "![" in section  # markdown image embeds
    # the chart PNGs were actually generated
    assert (charts_dir / "e1_ablation.png").exists()
    assert (charts_dir / "e2_gate_classifier.png").exists()


def test_results_section_contains_key_numbers(results_dirs, tmp_path):
    from render_readme import build_results_section

    section = build_results_section(results_dirs, charts_dir=tmp_path / "charts")
    # E1 numbers
    assert "0.43" in section and "1.00" in section
    # E2 recall progression
    assert "0.25" in section
    # legacy sections still present
    assert "6/6" in section and "8/8" in section


def test_inject_replaces_marked_block_idempotently(results_dirs, tmp_path):
    from render_readme import RESULTS_BEGIN, RESULTS_END, inject_section

    readme = tmp_path / "README.md"
    readme.write_text(
        f"# Title\n\nintro\n\n{RESULTS_BEGIN}\nold content\n{RESULTS_END}\n\nfooter\n"
    )

    inject_section(readme, "NEW CONTENT", RESULTS_BEGIN, RESULTS_END)
    first = readme.read_text()
    assert "NEW CONTENT" in first
    assert "old content" not in first
    assert "intro" in first and "footer" in first

    inject_section(readme, "NEW CONTENT", RESULTS_BEGIN, RESULTS_END)
    assert readme.read_text() == first


def test_inject_fails_loudly_when_markers_missing(tmp_path):
    from render_readme import RESULTS_BEGIN, RESULTS_END, inject_section

    readme = tmp_path / "README.md"
    readme.write_text("# Title\nno markers here\n")
    with pytest.raises(ValueError, match="marker"):
        inject_section(readme, "content", RESULTS_BEGIN, RESULTS_END)


def test_missing_result_file_produces_explicit_placeholder(tmp_path):
    from render_readme import build_results_section

    empty = tmp_path / "nothing"
    empty.mkdir()
    section = build_results_section(empty, charts_dir=tmp_path / "charts")
    assert "not available" in section.lower()
