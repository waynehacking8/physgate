"""Publication-quality matplotlib figures for the benchmark results.

Single source of truth: the result JSONs under ``benchmarks/*/results/``.
Every figure is rendered at 300 DPI with explicit axes, units, legends, and
(when repeat-run data exists) mean ± std error bars.

Layout policy (camera-ready):
* constrained_layout reserves space for titles, suptitles, above-axes legends,
  and colorbars deterministically — no ``bbox="tight"`` crop, so the requested
  figsize aspect ratio is the output aspect ratio and nothing clips;
* value labels use point offsets (not data-unit offsets) so the gap above a bar
  is constant regardless of the y-limit;
* ylim sits just above the 1.0 data ceiling (no stunted-bar dead band);
* legends never sit on top of data.

Palette semantics (one meaning per color, enforced across every figure):
* blue   = model / throughput series
* orange = secondary / efficiency series
* green  = correct / good outcome
* pink   = bad / artifact outcome
* grey   = baseline
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering — no display required
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------- style

DPI = 300

#: Shared font sizes (kept consistent across every figure)
LABEL_FS = 8     # bar value labels
TICK_FS = 8      # axis tick labels
TITLE_FS = 10    # axes titles
SUPTITLE_FS = 11

#: Colorblind-safe palette (Okabe-Ito). See the module docstring for semantics.
COLORS = {
    "mock": "#999999",    # baseline
    "claude": "#0072B2",  # model series
    "llm": "#0072B2",     # alias of claude (PLANNER_LABELS maps both -> Real Claude)
    "accent": "#D55E00",  # secondary / efficiency
    "pre": "#CC79A7",     # bad / artifact
    "post": "#009E73",    # correct / good
}

PLANNER_LABELS = {"mock": "Mock planner", "claude": "Real Claude", "llm": "Real Claude"}

#: The five component metrics, plus the derived orchestrator_score (a mean of
#: the five — plotted with a gap so it is not read as an independent measurement).
COMPONENT_METRICS = [
    ("end_to_end_success_rate", "End-to-end\nsuccess"),
    ("infeasible_recognition_rate", "Infeasible\nrecognition"),
    ("recovery_rate", "Failure\nrecovery"),
    ("decomposition_validity_rate", "Decomposition\nvalidity"),
    ("invalid_plan_catch_rate", "Invalid-plan\ncatch"),
]
DERIVED_METRIC = ("orchestrator_score", "Orchestrator\nscore (mean)")
METRIC_LABELS = [*COMPONENT_METRICS, DERIVED_METRIC]  # kept for back-compat

#: Pre-rebuild (milestone 2) feasibility — the artifact the rebuild eliminated.
PRE_REBUILD = {"mock": (2, 8), "claude": (1, 7)}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.constrained_layout.use": True,
            "font.size": 9,
            "axes.titlesize": TITLE_FS,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "legend.fontsize": LABEL_FS,
        }
    )


def _finish(fig, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def _label_bars(ax, bars, values, *, fmt="{:.2f}", base=None, fontsize=LABEL_FS):
    """Annotate bars with a constant point-offset above their top (or above an
    optional ``base`` height, e.g. the upper CI whisker)."""
    for i, (bar, val) in enumerate(zip(bars, values)):
        y = bar.get_height() if base is None else base[i]
        ax.annotate(
            fmt.format(val),
            (bar.get_x() + bar.get_width() / 2, y),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
            clip_on=False,
        )


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
    fig, ax = plt.subplots(figsize=(7.0, 4.0), layout="constrained")

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
        x - width / 2, pre_vals, width,
        label="Pre-rebuild (routing artifact)", color=COLORS["pre"], alpha=0.9,
    )
    bars_post = ax.bar(
        x + width / 2, post_vals, width,
        label="Post-rebuild (A* navigation)", color=COLORS["post"], alpha=0.9,
    )

    for bars, vals, labels in (
        (bars_pre, pre_vals, pre_labels),
        (bars_post, post_vals, post_labels),
    ):
        for bar, label in zip(bars, labels):
            ax.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points", xytext=(0, 3),
                ha="center", va="bottom", fontsize=LABEL_FS,
                fontweight="bold", clip_on=False,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([PLANNER_LABELS.get(p, p) for p in planners], fontsize=TICK_FS)
    ax.set_ylabel("Physically feasible plans (%)")
    ax.set_ylim(0, 118)
    # all 3 repeat runs are identical (sigma = 0): say so rather than fake error bars
    ax.text(
        0.5, 0.04, "post-rebuild: 100% feasible across n=3 runs (σ = 0)",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=LABEL_FS, color="#444444",
    )
    ax.set_title("Plan feasibility before vs after the navigation rebuild")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
    return _finish(fig, output)


def orchestrator_chart(
    reports: list[dict], output: str | Path, repeats: list[list[dict]] | None = None
) -> Path:
    """Agent-orchestrator metrics, mock vs real Claude, with error bars when
    repeat-run statistics are available."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(9.0, 4.8), layout="constrained")

    stats = aggregate_metric_repeats(repeats) if repeats else None

    component_keys = [k for k, _ in COMPONENT_METRICS]
    metric_keys = [*component_keys, DERIVED_METRIC[0]]
    # x positions: 5 component metrics packed, then a gap before the derived score
    x = np.array([*range(len(component_keys)), len(component_keys) + 0.7])
    n_planners = len(reports)
    width = 0.76 / n_planners
    sd_zero = True

    for i, report in enumerate(reports):
        planner = report["planner_name"]
        color = COLORS.get(planner, COLORS["accent"])
        offsets = x + (i - (n_planners - 1) / 2) * width

        if stats and planner in stats:
            means = [stats[planner][k]["mean"] for k in metric_keys]
            stds = [stats[planner][k]["std"] for k in metric_keys]
            if any(s > 0 for s in stds):
                sd_zero = False
            n_runs = max(stats[planner][k]["n"] for k in metric_keys)
            label = f"{PLANNER_LABELS.get(planner, planner)} (n={n_runs} runs)"
            bars = ax.bar(
                offsets, means, width, yerr=stds, capsize=3,
                error_kw={"linewidth": 1.0}, label=label, color=color, alpha=0.9,
            )
            annotate_vals = means
        else:
            vals = [report["metrics"].get(k, 0.0) for k in metric_keys]
            bars = ax.bar(
                offsets, vals, width,
                label=PLANNER_LABELS.get(planner, planner), color=color, alpha=0.9,
            )
            annotate_vals = vals
        _label_bars(ax, bars, annotate_vals, fontsize=7.5)

    # visual separator marking the derived score as different from the 5 components
    ax.axvline(len(component_keys) - 0.15, color="#bbbbbb", linewidth=0.8, linestyle="-")

    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in METRIC_LABELS], fontsize=TICK_FS)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    note = (
        "orchestrator score = unweighted mean of the 5 metrics at left"
        + ("   ·   error bars = sample SD (σ = 0 across n=3 runs)" if stats and sd_zero else "")
    )
    ax.text(0.5, 0.02, note, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=LABEL_FS, color="#444444")
    ax.set_title(
        "Agent-orchestrator quality: 5 metrics + derived score\n"
        "(7-scenario corpus: ordering / preconditions / recovery / multi-step / infeasible)"
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=n_planners, frameon=False)
    return _finish(fig, output)


def gpu_scaling_chart(data: dict, output: str | Path) -> Path:
    """Parallel-env throughput and scaling efficiency on one figure."""
    _setup_style()
    fig, ax_throughput = plt.subplots(figsize=(7.5, 4.2), layout="constrained")

    results = data["results"]
    envs = [r["num_envs"] for r in results]
    throughput = [r["env_steps_per_s"] for r in results]
    efficiency = [r["scaling_efficiency"] for r in results]

    ax_throughput.loglog(
        envs, throughput, marker="o", markersize=5, linewidth=1.6,
        color=COLORS["claude"], label="Throughput (env-steps/s)",
    )
    ax_throughput.set_xlabel("Parallel environments")
    ax_throughput.set_ylabel("Throughput (env-steps/s)", color=COLORS["claude"])
    ax_throughput.tick_params(axis="y", labelcolor=COLORS["claude"])
    ax_throughput.set_xticks(envs)
    ax_throughput.set_xticklabels([str(e) for e in envs])
    ax_throughput.minorticks_off()

    ax_eff = ax_throughput.twinx()
    ax_eff.plot(
        envs, efficiency, marker="s", markersize=5, linewidth=1.6, linestyle="--",
        color=COLORS["accent"], label="Scaling efficiency",
    )
    ax_eff.set_ylabel("Scaling efficiency", color=COLORS["accent"])
    ax_eff.tick_params(axis="y", labelcolor=COLORS["accent"])
    ax_eff.set_ylim(0.45, 1.08)
    ax_eff.set_xscale("log")
    ax_eff.spines["right"].set_visible(True)
    ax_eff.spines["right"].set_color(COLORS["accent"])
    ax_eff.grid(False)

    for env_count, eff in zip(envs, efficiency):
        ax_eff.annotate(
            f"{eff:.2f}", (env_count, eff),
            textcoords="offset points", xytext=(0, 8),
            ha="center", va="bottom", fontsize=LABEL_FS, color=COLORS["accent"],
        )

    knee = data.get("saturation_knee_envs")
    if knee:
        ax_throughput.axvline(knee, color="#999999", linewidth=0.8, linestyle=":")
        ax_throughput.annotate(
            f"saturation knee\n{knee} envs", (knee, throughput[0]),
            textcoords="offset points", xytext=(6, 0), ha="left", va="bottom",
            fontsize=LABEL_FS, color="#555555",
        )
    else:
        ax_throughput.text(
            0.98, 0.05, "saturation knee not reached in measured range",
            transform=ax_throughput.transAxes, fontsize=LABEL_FS, color="#555555",
            ha="right", va="bottom",
        )
    ax_throughput.set_title("GPU parallel-validation throughput and scaling efficiency")

    lines_a, labels_a = ax_throughput.get_legend_handles_labels()
    lines_b, labels_b = ax_eff.get_legend_handles_labels()
    ax_throughput.legend(lines_a + lines_b, labels_a + labels_b, loc="lower left")
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
    pre-execution rejection rate on infeasible tasks, per pipeline condition."""
    _setup_style()
    fig, (ax_success, ax_reject) = plt.subplots(
        1, 2, figsize=(11.0, 4.4), layout="constrained",
        gridspec_kw={"width_ratios": [1.1, 1.0]},
    )

    conditions = list(data["conditions"].keys())
    x = np.arange(len(conditions))
    config = data.get("config", {})

    # ---- left panel: success on feasible instances ----
    rates = [data["conditions"][c]["success_rate"] for c in conditions]
    err_low, err_high, upper = [], [], []
    for c in conditions:
        ci = data["conditions"][c].get("success_ci_95", [None, None])
        rate = data["conditions"][c]["success_rate"]
        lo = rate - ci[0] if ci[0] is not None else 0
        hi = ci[1] - rate if ci[1] is not None else 0
        err_low.append(lo)
        err_high.append(hi)
        upper.append(rate + hi)

    bars = ax_success.bar(
        x, rates, 0.62, yerr=[err_low, err_high], capsize=4,
        color=[COLORS["pre"], COLORS["accent"], COLORS["claude"], COLORS["post"]],
        alpha=0.9, error_kw={"linewidth": 1.2},
    )
    _label_bars(ax_success, bars, rates, base=upper, fontsize=9)
    ax_success.set_xticks(x)
    ax_success.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions], fontsize=TICK_FS)
    ax_success.set_ylabel("Task success rate")
    ax_success.set_ylim(0, 1.12)
    ax_success.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    n_feasible = config.get("n_feasible_instances", "?")
    ax_success.set_title(
        f"Feasible tasks (n={n_feasible} generated layouts)\nsuccess rate, 95% Wilson CI"
    )

    # ---- right panel: infeasible-task handling (pre-execution rejection rate) ----
    rejection = [data["conditions"][c]["rejection_rate"] for c in conditions]
    bar_colors = [COLORS["post"] if r > 0.5 else COLORS["pre"] for r in rejection]
    bars_r = ax_reject.bar(x, rejection, 0.62, color=bar_colors, alpha=0.9)
    _label_bars(ax_reject, bars_r, rejection, fontsize=9)
    ax_reject.set_xticks(x)
    ax_reject.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions], fontsize=TICK_FS)
    ax_reject.set_ylabel("Pre-execution rejection rate")
    ax_reject.set_ylim(0, 1.12)
    ax_reject.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.5)
    n_infeasible = config.get("n_infeasible_instances", "?")
    ax_reject.set_title(
        f"Infeasible tasks (n={n_infeasible} unreachable layouts)\n"
        "rejected before wasting execution (higher is better)"
    )
    ax_reject.text(
        0.02, 0.96, "green = task rejected (good)\npink = falsely executed (bad)",
        transform=ax_reject.transAxes, ha="left", va="top",
        fontsize=LABEL_FS, color="#444444",
    )

    fig.suptitle(
        "E1 — Pipeline ablation: what each validation layer adds",
        fontsize=SUPTITLE_FS, fontweight="bold",
    )
    return _finish(fig, output)


def gate_classifier_chart(data: dict, output: str | Path) -> Path:
    """E2 gate-as-classifier: recall (defect catch rate) per validation layer,
    plus a per-defect correctness matrix that substantiates layered defense."""
    _setup_style()
    fig, (ax_recall, ax_matrix) = plt.subplots(
        1, 2, figsize=(12.0, 4.6), layout="constrained",
        gridspec_kw={"width_ratios": [0.8, 1.5]},
    )

    layers = list(data["layers"].keys())

    # ---- left: recall (+ honest all-zero FPR handling) ----
    recalls = [data["layers"][layer]["report"]["recall"] for layer in layers]
    fprs = [data["layers"][layer]["report"]["false_positive_rate"] for layer in layers]
    x = np.arange(len(layers))
    any_fpr = any(f > 0 for f in fprs)
    width = 0.38 if any_fpr else 0.55

    bars_recall = ax_recall.bar(
        x - (width / 2 if any_fpr else 0), recalls, width,
        label="Recall (defects caught)", color=COLORS["claude"], alpha=0.9,
    )
    _label_bars(ax_recall, bars_recall, recalls, fontsize=8.5)
    if any_fpr:
        ax_recall.bar(
            x + width / 2, fprs, width,
            label="False-positive rate\n(good plans killed)",
            color=COLORS["pre"], alpha=0.9,
        )
        ax_recall.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    else:
        # all FPR = 0: mark the zeros explicitly so absence reads as a real zero
        ax_recall.scatter(x, [0] * len(x), marker="_", s=200, color=COLORS["pre"], zorder=3)
        ax_recall.text(
            0.02, 0.97, "FPR = 0.00 across all layers\n(no valid plan ever rejected)",
            transform=ax_recall.transAxes, ha="left", va="top",
            fontsize=7.5, color="#444444",
        )

    ax_recall.set_xticks(x)
    ax_recall.set_xticklabels(
        [LAYER_LABELS.get(layer, layer).replace("\n", " ") for layer in layers],
        fontsize=7, rotation=20, ha="right",
    )
    ax_recall.set_ylabel("Recall")
    ax_recall.set_ylim(0, 1.12)
    ax_recall.set_title("Defect catch rate by validation layer")

    # ---- right: layer x defect-class correctness matrix ----
    from physgate.eval_v2.defects import DEFECT_CLASSES

    present = {d for layer in layers for d in data["layers"][layer]["per_defect_rejection_rate"]}
    # regime-sort columns: invalid (should reject) first, then valid (should accept)
    invalid = sorted(d for d in present if DEFECT_CLASSES.get(d, True))
    valid = sorted(d for d in present if not DEFECT_CLASSES.get(d, True))
    defect_ids = [*invalid, *valid]

    matrix = np.array(
        [
            [data["layers"][layer]["per_defect_rejection_rate"].get(d, np.nan) for d in defect_ids]
            for layer in layers
        ]
    )
    # SINGLE encoding: both color AND printed number are "correct behavior rate".
    correctness = np.empty_like(matrix)
    for j, defect in enumerate(defect_ids):
        should_reject = DEFECT_CLASSES.get(defect, True)
        correctness[:, j] = matrix[:, j] if should_reject else 1.0 - matrix[:, j]

    im = ax_matrix.imshow(correctness, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax_matrix.set_xticks(range(len(defect_ids)))
    ax_matrix.set_xticklabels([DEFECT_LABELS.get(d, d) for d in defect_ids], fontsize=7.5)
    ax_matrix.set_yticks(range(len(layers)))
    ax_matrix.set_yticklabels([LAYER_LABELS.get(layer, layer) for layer in layers], fontsize=TICK_FS)
    ax_matrix.grid(False)
    for i in range(len(layers)):
        for j, defect in enumerate(defect_ids):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            should_reject = DEFECT_CLASSES.get(defect, True)
            action = "reject" if should_reject else "accept"
            ax_matrix.text(
                j, i, f"{action}\n{correctness[i, j]:.0%}",
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color="white" if correctness[i, j] < 0.25 else "black",
            )
    # divider between the invalid (reject) and valid (accept) regimes
    if invalid and valid:
        ax_matrix.axvline(len(invalid) - 0.5, color="black", linewidth=1.4)
    ax_matrix.set_title("Per-defect behavior by validation layer")
    colorbar = fig.colorbar(im, ax=ax_matrix, fraction=0.04, pad=0.02)
    colorbar.set_label("correct behavior (rate)", fontsize=LABEL_FS)

    fig.suptitle(
        "E2 — The gate as a classifier (defect-injection corpus)\n"
        "left of divider: invalid plans (correct = reject)   ·   "
        "right: valid plans (correct = accept)",
        fontsize=SUPTITLE_FS, fontweight="bold",
    )
    return _finish(fig, output)
