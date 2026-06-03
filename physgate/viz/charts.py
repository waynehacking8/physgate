"""Publication-quality matplotlib figures for the benchmark results.

Single source of truth: the result JSONs under ``benchmarks/*/results/``.
Every figure is rendered at 300 DPI with explicit axes, units, legends, and
(when repeat-run data exists) mean ± std error bars.

Style follows common robotics-paper conventions: colorblind-safe palette,
no chartjunk, annotated values on bars.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering — no display required
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------- style

DPI = 300
#: Colorblind-safe palette (Okabe-Ito)
COLORS = {
    "mock": "#999999",
    "claude": "#0072B2",
    "llm": "#0072B2",
    "accent": "#D55E00",
    "pre": "#CC79A7",
    "post": "#009E73",
}

PLANNER_LABELS = {"mock": "Mock planner", "claude": "Real Claude", "llm": "Real Claude"}

METRIC_LABELS = [
    ("end_to_end_success_rate", "End-to-end\nsuccess"),
    ("infeasible_recognition_rate", "Infeasible\nrecognition"),
    ("recovery_rate", "Failure\nrecovery"),
    ("decomposition_validity_rate", "Decomposition\nvalidity"),
    ("invalid_plan_catch_rate", "Invalid-plan\ncatch"),
    ("orchestrator_score", "Orchestrator\nscore"),
]

#: Pre-rebuild (milestone 2) feasibility — the artifact the rebuild eliminated.
PRE_REBUILD = {"mock": (2, 8), "claude": (1, 7)}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def _finish(fig, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


# ----------------------------------------------------------------- statistics


def aggregate_metric_repeats(repeats: list[list[dict]]) -> dict[str, dict[str, dict]]:
    """Aggregate repeat-run reports into per-planner per-metric statistics.

    Args:
        repeats: list of runs; each run is a list of report dicts
            ({"planner_name", "metrics"}).

    Returns:
        planner -> metric -> {"mean", "std", "n", "values"}.
        std is the sample standard deviation (ddof=1); 0.0 when n == 1.
    """
    collected: dict[str, dict[str, list[float]]] = {}
    for run in repeats:
        for report in run:
            planner = report["planner_name"]
            for metric, value in report["metrics"].items():
                collected.setdefault(planner, {}).setdefault(metric, []).append(float(value))

    stats: dict[str, dict[str, dict]] = {}
    for planner, metrics in collected.items():
        stats[planner] = {}
        for metric, values in metrics.items():
            arr = np.asarray(values, dtype=float)
            stats[planner][metric] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "n": int(len(arr)),
                "values": [float(v) for v in arr],
            }
    return stats


# -------------------------------------------------------------------- figures


def feasibility_chart(
    data: dict, output: str | Path, repeats: list[dict] | None = None
) -> Path:
    """Pre- vs post-rebuild plan feasibility, grouped by planner.

    The figure that documents the artifact elimination: feasibility was a
    property of the broken low level (straight-line driver), not of the plans.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    planners = [m["planner"] for m in data["measurements"]]
    x = np.arange(len(planners))
    width = 0.36

    pre_vals, pre_labels = [], []
    post_vals, post_labels = [], []
    for m in data["measurements"]:
        pre_n, pre_total = PRE_REBUILD.get(m["planner"], (0, 1))
        pre_vals.append(100.0 * pre_n / pre_total)
        pre_labels.append(f"{pre_n}/{pre_total}")
        post_vals.append(100.0 * m["feasibility_of_survivors"])
        post_labels.append(f"{m['physically_feasible']}/{m['critic_survivors']}")

    bars_pre = ax.bar(
        x - width / 2,
        pre_vals,
        width,
        label="Pre-rebuild (routing artifact)",
        color=COLORS["pre"],
        alpha=0.9,
    )
    bars_post = ax.bar(
        x + width / 2,
        post_vals,
        width,
        label="Post-rebuild (deterministic A* navigation)",
        color=COLORS["post"],
        alpha=0.9,
    )

    for bars, labels in ((bars_pre, pre_labels), (bars_post, post_labels)):
        for bar, label in zip(bars, labels):
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([PLANNER_LABELS.get(p, p) for p in planners])
    ax.set_ylabel("Physically feasible plans (%)")
    ax.set_ylim(0, 112)
    ax.set_title(
        "Plan feasibility: critic-surviving plans that succeed in Isaac Lab physics\n"
        "(pre-rebuild feasibility measured the broken low level, not the plans)"
    )
    ax.legend(loc="upper left")
    return _finish(fig, output)


def orchestrator_chart(
    reports: list[dict], output: str | Path, repeats: list[list[dict]] | None = None
) -> Path:
    """Agent-orchestrator metrics, mock vs real Claude, with error bars when
    repeat-run statistics are available."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8.0, 3.8))

    stats = aggregate_metric_repeats(repeats) if repeats else None

    metric_keys = [k for k, _ in METRIC_LABELS]
    x = np.arange(len(metric_keys))
    n_planners = len(reports)
    width = 0.76 / n_planners

    for i, report in enumerate(reports):
        planner = report["planner_name"]
        color = COLORS.get(planner, COLORS["accent"])
        offsets = x + (i - (n_planners - 1) / 2) * width

        if stats and planner in stats:
            means = [stats[planner][k]["mean"] for k in metric_keys]
            stds = [stats[planner][k]["std"] for k in metric_keys]
            ns = {stats[planner][k]["n"] for k in metric_keys}
            n_runs = max(ns)
            label = f"{PLANNER_LABELS.get(planner, planner)} (n={n_runs} runs)"
            bars = ax.bar(
                offsets,
                means,
                width,
                yerr=stds,
                capsize=3,
                error_kw={"linewidth": 1.0},
                label=label,
                color=color,
                alpha=0.9,
            )
            annotate_vals = means
        else:
            vals = [report["metrics"].get(k, 0.0) for k in metric_keys]
            bars = ax.bar(
                offsets,
                vals,
                width,
                label=PLANNER_LABELS.get(planner, planner),
                color=color,
                alpha=0.9,
            )
            annotate_vals = vals

        for bar, val in zip(bars, annotate_vals):
            ax.annotate(
                f"{val:.2f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04),
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METRIC_LABELS])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.22)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    ax.set_title(
        "Agent-orchestrator evaluation: 7 scenarios "
        "(ordering / preconditions / recovery / multi-step / infeasible), "
        "real LangGraph orchestrator + fault injection"
    )
    ax.legend(loc="lower right")
    return _finish(fig, output)


def gpu_scaling_chart(data: dict, output: str | Path) -> Path:
    """Parallel-env throughput and scaling efficiency on one figure."""
    _setup_style()
    fig, ax_throughput = plt.subplots(figsize=(7.0, 3.6))

    results = data["results"]
    envs = [r["num_envs"] for r in results]
    throughput = [r["env_steps_per_s"] for r in results]
    efficiency = [r["scaling_efficiency"] for r in results]

    # throughput: log-log line
    ax_throughput.loglog(
        envs,
        throughput,
        marker="o",
        markersize=5,
        linewidth=1.6,
        color=COLORS["claude"],
        label="Throughput (env-steps/s)",
    )
    ax_throughput.set_xlabel("Parallel environments")
    ax_throughput.set_ylabel("Throughput (env-steps/s)", color=COLORS["claude"])
    ax_throughput.tick_params(axis="y", labelcolor=COLORS["claude"])
    ax_throughput.set_xticks(envs)
    ax_throughput.set_xticklabels([str(e) for e in envs])
    ax_throughput.minorticks_off()

    # efficiency: right axis, linear
    ax_eff = ax_throughput.twinx()
    ax_eff.plot(
        envs,
        efficiency,
        marker="s",
        markersize=5,
        linewidth=1.6,
        linestyle="--",
        color=COLORS["accent"],
        label="Scaling efficiency",
    )
    ax_eff.set_ylabel("Scaling efficiency", color=COLORS["accent"])
    ax_eff.tick_params(axis="y", labelcolor=COLORS["accent"])
    ax_eff.set_ylim(0.5, 1.05)
    ax_eff.set_xscale("log")
    ax_eff.spines["right"].set_visible(True)
    ax_eff.grid(False)

    for env_count, eff in zip(envs, efficiency):
        ax_eff.annotate(
            f"{eff:.2f}",
            (env_count, eff),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=7.5,
            color=COLORS["accent"],
        )

    knee = data.get("saturation_knee_envs")
    knee_text = (
        f"saturation knee: {knee} envs"
        if knee
        else "saturation knee NOT reached in measured range"
    )
    ax_throughput.set_title(
        f"GPU parallel-validation scaling — RTX PRO 6000 Blackwell, 300 W\n({knee_text})"
    )

    # combined legend
    lines_a, labels_a = ax_throughput.get_legend_handles_labels()
    lines_b, labels_b = ax_eff.get_legend_handles_labels()
    ax_throughput.legend(lines_a + lines_b, labels_a + labels_b, loc="upper left")
    return _finish(fig, output)


# ---------------------------------------------------------- eval_v2 figures

CONDITION_LABELS = {
    "A0_no_validation": "A0\nNo validation",
    "A1_critic_only": "A1\n+ LLM critic",
    "A2_symbolic_gate": "A2\n+ Symbolic gate",
    "A3_nav_aware_gate": "A3\n+ Physics gate\n(nav + goal)",
}

LAYER_LABELS = {
    "critic": "LLM critic",
    "l1_l3": "+ L1/L3\ndeterministic",
    "symbolic_gate": "+ Symbolic L2\n(full gate)",
    "physics_gate": "Isaac physics L2",
}

DEFECT_LABELS = {
    "D0_clean": "clean\n(control)",
    "D1_step_inversion": "step\ninversion",
    "D2_missing_pick": "missing\npick",
    "D3_hallucinated_target": "hallucinated\ntarget",
    "D4_wrong_placement": "wrong\nplacement",
    "D6_speed_violation": "speed\nviolation*",
    "D7_redundant_step": "redundant\nstep*",
}


def ablation_chart(data: dict, output: str | Path) -> Path:
    """E1 pipeline ablation: success rate on feasible tasks (with 95% CI) and
    false-execution rate on infeasible tasks, per pipeline condition."""
    _setup_style()
    fig, (ax_success, ax_reject) = plt.subplots(
        1, 2, figsize=(11.0, 4.0), gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.25}
    )

    conditions = list(data["conditions"].keys())
    x = np.arange(len(conditions))
    config = data.get("config", {})

    # ---- left panel: success on feasible instances ----
    rates = [data["conditions"][c]["success_rate"] for c in conditions]
    err_low, err_high = [], []
    for c in conditions:
        ci = data["conditions"][c].get("success_ci_95", [None, None])
        rate = data["conditions"][c]["success_rate"]
        err_low.append(rate - ci[0] if ci[0] is not None else 0)
        err_high.append(ci[1] - rate if ci[1] is not None else 0)

    bars = ax_success.bar(
        x,
        rates,
        0.62,
        yerr=[err_low, err_high],
        capsize=4,
        color=[COLORS["pre"], COLORS["accent"], COLORS["claude"], COLORS["post"]],
        alpha=0.9,
        error_kw={"linewidth": 1.2},
    )
    for bar, rate in zip(bars, rates):
        ax_success.annotate(
            f"{rate:.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax_success.set_xticks(x)
    ax_success.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions], fontsize=8)
    ax_success.set_ylabel("Task success rate")
    ax_success.set_ylim(0, 1.2)
    ax_success.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    n_feasible = config.get("n_feasible_instances", "?")
    ax_success.set_title(
        f"Feasible tasks (n={n_feasible} generated layouts)\nsuccess rate, 95% Wilson CI"
    )

    # ---- right panel: infeasible-task handling (pre-execution rejection rate) ----
    rejection = [data["conditions"][c]["rejection_rate"] for c in conditions]
    bar_colors = [COLORS["post"] if r > 0.5 else COLORS["pre"] for r in rejection]
    bars_r = ax_reject.bar(x, rejection, 0.62, color=bar_colors, alpha=0.9)
    for bar, rate in zip(bars_r, rejection):
        ax_reject.annotate(
            f"{rate:.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.06),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax_reject.set_xticks(x)
    ax_reject.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions], fontsize=8)
    ax_reject.set_ylabel("Pre-execution rejection rate")
    ax_reject.set_ylim(0, 1.2)
    ax_reject.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    n_infeasible = config.get("n_infeasible_instances", "?")
    ax_reject.set_title(
        f"Infeasible tasks (n={n_infeasible} unreachable layouts)\n"
        "higher is better: impossible tasks rejected before wasting execution"
    )

    fig.suptitle(
        "E1 — Pipeline ablation: what each validation layer adds",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    return _finish(fig, output)


def gate_classifier_chart(data: dict, output: str | Path) -> Path:
    """E2 gate-as-classifier: recall (defect catch rate) per validation layer,
    with the per-defect breakdown that substantiates the layered-defense claim."""
    _setup_style()
    fig, (ax_recall, ax_matrix) = plt.subplots(
        1, 2, figsize=(12.0, 4.0), gridspec_kw={"width_ratios": [0.8, 1.5], "wspace": 0.3}
    )

    layers = list(data["layers"].keys())

    # ---- left: recall + FPR per layer ----
    recalls = [data["layers"][layer]["report"]["recall"] for layer in layers]
    fprs = [data["layers"][layer]["report"]["false_positive_rate"] for layer in layers]
    x = np.arange(len(layers))
    width = 0.38
    ax_recall.bar(
        x - width / 2,
        recalls,
        width,
        label="Recall (defects caught)",
        color=COLORS["claude"],
        alpha=0.9,
    )
    ax_recall.bar(
        x + width / 2,
        fprs,
        width,
        label="False-positive rate\n(good plans killed)",
        color=COLORS["pre"],
        alpha=0.9,
    )
    for xi, recall in zip(x, recalls):
        ax_recall.annotate(
            f"{recall:.2f}",
            (xi - width / 2, recall + 0.03),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax_recall.set_xticks(x)
    ax_recall.set_xticklabels(
        [LAYER_LABELS.get(layer, layer).replace("\n", " ") for layer in layers],
        fontsize=7,
        rotation=20,
        ha="right",
    )
    ax_recall.set_ylabel("Rate")
    ax_recall.set_ylim(0, 1.3)
    ax_recall.set_title("Defect catch rate by validation layer")
    ax_recall.legend(fontsize=7.5, loc="upper left", ncols=1)

    # ---- right: layer x defect-class catch matrix ----
    defect_ids = sorted(
        {d for layer in layers for d in data["layers"][layer]["per_defect_rejection_rate"]}
    )
    matrix = np.array(
        [
            [
                data["layers"][layer]["per_defect_rejection_rate"].get(d, np.nan)
                for d in defect_ids
            ]
            for layer in layers
        ]
    )
    # color encodes CORRECTNESS, not raw rejection: for invalid-plan classes the
    # correct behavior is rejection; for valid-plan classes (D0, D6, D7) the
    # correct behavior is acceptance. Green always = the layer did the right thing.
    from physgate.eval_v2.defects import DEFECT_CLASSES

    correctness = np.empty_like(matrix)
    for j, defect in enumerate(defect_ids):
        should_reject = DEFECT_CLASSES.get(defect, True)
        correctness[:, j] = matrix[:, j] if should_reject else 1.0 - matrix[:, j]
    im = ax_matrix.imshow(correctness, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax_matrix.set_xticks(range(len(defect_ids)))
    ax_matrix.set_xticklabels([DEFECT_LABELS.get(d, d) for d in defect_ids], fontsize=7.5)
    ax_matrix.set_yticks(range(len(layers)))
    ax_matrix.set_yticklabels([LAYER_LABELS.get(layer, layer) for layer in layers], fontsize=8)
    ax_matrix.grid(False)
    for i in range(len(layers)):
        for j in range(len(defect_ids)):
            value = matrix[i, j]
            if not np.isnan(value):
                ax_matrix.text(
                    j,
                    i,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color="white" if correctness[i, j] < 0.3 or correctness[i, j] > 0.7 else "black",
                )
    ax_matrix.set_title(
        "Rejection rate per defect class — green = correct behavior\n"
        "(invalid plans: reject; *valid plans (control/clamped/wasteful): accept)"
    )
    colorbar = fig.colorbar(im, ax=ax_matrix, fraction=0.04, pad=0.02)
    colorbar.set_label("correctness", fontsize=7.5)

    fig.suptitle(
        "E2 — The gate as a classifier (defect-injection corpus)",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    return _finish(fig, output)
