"""Tests for physgate.viz.charts — publication-quality matplotlib figures.

The charts are generated from benchmark result JSONs (single source of truth)
at 300 DPI with proper axes, units, legends, and error bars when repeat-run
statistics are available.
"""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")


@pytest.fixture()
def feasibility_data():
    return {
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


@pytest.fixture()
def orchestration_reports():
    metrics = {
        "end_to_end_success_rate": 0.6,
        "infeasible_recognition_rate": 1.0,
        "recovery_rate": 1.0,
        "decomposition_validity_rate": 1.0,
        "invalid_plan_catch_rate": 1.0,
        "orchestrator_score": 0.92,
    }
    llm_metrics = {**metrics, "end_to_end_success_rate": 1.0, "orchestrator_score": 1.0}
    return [
        {"planner_name": "mock", "metrics": metrics},
        {"planner_name": "llm", "metrics": llm_metrics},
    ]


@pytest.fixture()
def gpu_data():
    return {
        "results": [
            {"num_envs": 1, "env_steps_per_s": 281.4, "scaling_efficiency": 1.0},
            {"num_envs": 16, "env_steps_per_s": 4360.2, "scaling_efficiency": 0.968},
            {"num_envs": 1024, "env_steps_per_s": 250227.2, "scaling_efficiency": 0.868},
        ],
        "saturation_knee_envs": None,
    }


def _assert_valid_png(path, min_bytes: int = 5000):
    """A real chart PNG is at least a few KB and starts with the PNG magic."""
    assert path.exists(), f"{path} not created"
    data = path.read_bytes()
    assert len(data) > min_bytes, f"{path} suspiciously small ({len(data)} bytes)"
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG file"


def test_feasibility_chart_renders(tmp_path, feasibility_data):
    from physgate.viz.charts import feasibility_chart

    out = feasibility_chart(feasibility_data, tmp_path / "feasibility.png")
    _assert_valid_png(out)


def test_orchestrator_chart_renders_without_error_bars(tmp_path, orchestration_reports):
    from physgate.viz.charts import orchestrator_chart

    out = orchestrator_chart(orchestration_reports, tmp_path / "orchestrator.png")
    _assert_valid_png(out)


def test_orchestrator_chart_renders_with_repeat_statistics(tmp_path, orchestration_reports):
    """When repeat runs exist, the chart shows mean +/- std error bars."""
    from physgate.viz.charts import orchestrator_chart

    # three repeat runs with slight variation in the llm scores
    repeats = []
    for delta in (0.0, -0.04, 0.0):
        run = [
            {"planner_name": r["planner_name"], "metrics": dict(r["metrics"])}
            for r in orchestration_reports
        ]
        run[1]["metrics"]["orchestrator_score"] = 1.0 + delta
        run[1]["metrics"]["end_to_end_success_rate"] = 1.0 + delta
        repeats.append(run)

    out = orchestrator_chart(
        orchestration_reports, tmp_path / "orchestrator_stats.png", repeats=repeats
    )
    _assert_valid_png(out)


def test_gpu_scaling_chart_renders(tmp_path, gpu_data):
    from physgate.viz.charts import gpu_scaling_chart

    out = gpu_scaling_chart(gpu_data, tmp_path / "gpu.png")
    _assert_valid_png(out)


def test_aggregate_repeats_computes_mean_and_std():
    from physgate.viz.charts import aggregate_metric_repeats

    repeats = [
        [{"planner_name": "llm", "metrics": {"orchestrator_score": 1.0}}],
        [{"planner_name": "llm", "metrics": {"orchestrator_score": 0.96}}],
        [{"planner_name": "llm", "metrics": {"orchestrator_score": 0.92}}],
    ]
    stats = aggregate_metric_repeats(repeats)
    assert stats["llm"]["orchestrator_score"]["n"] == 3
    assert abs(stats["llm"]["orchestrator_score"]["mean"] - 0.96) < 1e-9
    expected_std = float(np.std([1.0, 0.96, 0.92], ddof=1))
    assert abs(stats["llm"]["orchestrator_score"]["std"] - expected_std) < 1e-9


@pytest.fixture()
def ablation_data():
    return {
        "conditions": {
            "A0_no_validation": {
                "success_rate": 0.43,
                "success_ci_95": [0.38, 0.48],
                "false_execution_rate": 1.0,
                "rejection_rate": 0.0,
            },
            "A1_critic_only": {
                "success_rate": 0.50,
                "success_ci_95": [0.45, 0.55],
                "false_execution_rate": 1.0,
                "rejection_rate": 0.0,
            },
            "A2_symbolic_gate": {
                "success_rate": 1.0,
                "success_ci_95": [0.84, 1.0],
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


@pytest.fixture()
def classifier_data():
    report = {"precision": 1.0, "recall": 0.25, "f1": 0.4, "false_positive_rate": 0.0}
    return {
        "layers": {
            "critic": {
                "report": report,
                "per_defect_rejection_rate": {"D0_clean": 0.0, "D3_hallucinated_target": 1.0},
            },
            "l1_l3": {
                "report": {**report, "recall": 0.75, "f1": 0.86},
                "per_defect_rejection_rate": {"D0_clean": 0.0, "D1_step_inversion": 1.0},
            },
            "symbolic_gate": {
                "report": {**report, "recall": 1.0, "f1": 1.0},
                "per_defect_rejection_rate": {"D0_clean": 0.0, "D4_wrong_placement": 1.0},
            },
            "physics_gate": {
                "report": {**report, "recall": 1.0, "f1": 1.0},
                "per_defect_rejection_rate": {"D0_clean": 0.0, "D4_wrong_placement": 1.0},
            },
        }
    }


def test_ablation_chart_renders(tmp_path, ablation_data):
    from physgate.viz.charts import ablation_chart

    out = ablation_chart(ablation_data, tmp_path / "ablation.png")
    _assert_valid_png(out)


def test_gate_classifier_chart_renders(tmp_path, classifier_data):
    from physgate.viz.charts import gate_classifier_chart

    out = gate_classifier_chart(classifier_data, tmp_path / "classifier.png")
    _assert_valid_png(out)


def test_charts_are_publication_styled(tmp_path, feasibility_data):
    """Charts must be high-resolution (300 DPI -> wide pixel dimensions)."""
    from PIL import Image

    from physgate.viz.charts import feasibility_chart

    out = feasibility_chart(feasibility_data, tmp_path / "f.png")
    img = Image.open(out)
    # 300 DPI rendering (even after tight-bbox cropping) is >1500 px wide;
    # a default 100 DPI render of the same figure would be ~700 px
    assert img.width >= 1500, f"chart too low-resolution: {img.width}px wide"
