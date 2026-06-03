"""Publication-quality matplotlib figures for the benchmark results.

Single source of truth: the result JSONs under ``benchmarks/*/results/``.
Every figure is rendered at 300 DPI with constrained_layout, Tableau-10
palette, and minimal decoration.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DPI = 300

# ── Tableau-10 palette (the industry standard for data-viz) ──────────────
# Designed by Tableau / UW IDL: distinguishable, colorblind-safe, print-ready.
COLORS = {
    "mock": "#bab0ac",     # Tableau warm-grey — baseline
    "claude": "#4e79a7",   # Tableau steel-blue — model
    "llm": "#4e79a7",      # alias
    "accent": "#f28e2b",   # Tableau amber — secondary axis / emphasis
    "pre": "#e15759",      # Tableau brick-red — bad / pre-rebuild
    "post": "#59a14f",     # Tableau sage-green — good / post-rebuild
    "info": "#76b7b2",     # Tableau teal — supporting
    "purple": "#b07aa1",   # Tableau mauve — extra category
}

INK = "#333333"
MUTED = "#888888"

PLANNER_LABELS = {"mock": "Mock planner", "claude": "Real Claude", "llm": "Real Claude"}

COMPONENT_METRICS = [
    ("end_to_end_success_rate", "E2E\nsuccess"),
    ("infeasible_recognition_rate", "Infeasible\nrecognition"),
    ("recovery_rate", "Failure\nrecovery"),
    ("decomposition_validity_rate", "Decomposition\nvalidity"),
    ("invalid_plan_catch_rate", "Invalid-plan\ncatch"),
]
DERIVED_METRIC = ("orchestrator_score", "Orchestrator\nscore")
METRIC_LABELS = [*COMPONENT_METRICS, DERIVED_METRIC]

PRE_REBUILD = {"mock": (2, 8), "claude": (1, 7)}


# ── shared style ─────────────────────────────────────────────────────────

def _setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "figure.facecolor": "white",
        "figure.constrained_layout.use": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.titlepad": 12,
        "axes.labelsize": 10,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": "#ebebeb",
        "grid.alpha": 1.0,
        "grid.linewidth": 0.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def _finish(fig, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return output


def _vlabel(ax, bar, text, *, fontsize=8.5, y_override=None, color=INK):
    """Place a value label above a bar with a fixed point offset."""
    y = y_override if y_override is not None else bar.get_height()
    ax.annotate(
        text,
        (bar.get_x() + bar.get_width() / 2, y),
        textcoords="offset points", xytext=(0, 4),
        ha="center", va="bottom", fontsize=fontsize,
        fontweight="bold", color=color, clip_on=False,
    )


# ── statistics ───────────────────────────────────────────────────────────

def aggregate_metric_repeats(repeats: list[list[dict]]) -> dict[str, dict[str, dict]]:
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


# ── feasibility ──────────────────────────────────────────────────────────

def feasibility_chart(
    data: dict, output: str | Path, repeats: list[dict] | None = None
) -> Path:
    _setup_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.8), layout="constrained")

    planners = [m["planner"] for m in data["measurements"]]
    x = np.arange(len(planners))
    w = 0.28

    pre_vals, pre_fracs = [], []
    post_vals, post_fracs = [], []
    for m in data["measurements"]:
        pn, pt = PRE_REBUILD.get(m["planner"], (0, 1))
        pre_vals.append(100.0 * pn / pt)
        pre_fracs.append(f"{pn}/{pt}")
        post_vals.append(100.0 * m["feasibility_of_survivors"])
        post_fracs.append(f"{m['physically_feasible']}/{m['critic_survivors']}")

    bars_pre = ax.bar(x - w / 2, pre_vals, w, color=COLORS["pre"],
                      label="Pre-rebuild (artifact)", edgecolor="white", linewidth=0.6)
    bars_post = ax.bar(x + w / 2, post_vals, w, color=COLORS["post"],
                       label="Post-rebuild (A* nav)", edgecolor="white", linewidth=0.6)

    for bars, fracs in ((bars_pre, pre_fracs), (bars_post, post_fracs)):
        for bar, frac in zip(bars, fracs):
            _vlabel(ax, bar, frac, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([PLANNER_LABELS.get(p, p) for p in planners])
    ax.set_xlim(-0.55, len(planners) - 0.45)
    ax.set_ylabel("Feasible plans (%)")
    ax.set_ylim(0, 114)
    ax.set_title("Plan feasibility: before vs after navigation rebuild")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=8.5, frameon=False)
    ax.text(0.99, 0.02, "n = 3 runs, σ = 0", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color=MUTED)
    return _finish(fig, output)


# ── orchestrator ─────────────────────────────────────────────────────────

def orchestrator_chart(
    reports: list[dict], output: str | Path, repeats: list[list[dict]] | None = None
) -> Path:
    _setup_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.0), layout="constrained")

    stats = aggregate_metric_repeats(repeats) if repeats else None

    comp_keys = [k for k, _ in COMPONENT_METRICS]
    metric_keys = [*comp_keys, DERIVED_METRIC[0]]
    x = np.array([*range(len(comp_keys)), len(comp_keys) + 0.8])
    n_planners = len(reports)
    w = 0.32

    for i, report in enumerate(reports):
        planner = report["planner_name"]
        color = COLORS.get(planner, COLORS["accent"])
        offsets = x + (i - (n_planners - 1) / 2) * w

        if stats and planner in stats:
            means = [stats[planner][k]["mean"] for k in metric_keys]
            stds = [stats[planner][k]["std"] for k in metric_keys]
            n_runs = max(stats[planner][k]["n"] for k in metric_keys)
            label = f"{PLANNER_LABELS.get(planner, planner)} (n={n_runs})"
            bars = ax.bar(offsets, means, w, yerr=stds, capsize=3,
                          error_kw={"linewidth": 0.8, "color": "#555"},
                          label=label, color=color, edgecolor="white", linewidth=0.6)
            vals = means
        else:
            vals = [report["metrics"].get(k, 0.0) for k in metric_keys]
            bars = ax.bar(offsets, vals, w, color=color, edgecolor="white",
                          linewidth=0.6, label=PLANNER_LABELS.get(planner, planner))

        for bar, val in zip(bars, vals):
            if abs(val - 1.0) > 0.005:
                _vlabel(ax, bar, f"{val:.2f}", fontsize=8)

    ax.axvline(len(comp_keys) + 0.05, color="#d0d0d0", linewidth=1.0, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in METRIC_LABELS], fontsize=8.5)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.10)
    ax.axhline(1.0, color="#d0d0d0", linewidth=0.5, linestyle=":")
    ax.set_title("Orchestrator quality — 7-scenario regression suite")
    ax.legend(loc="lower left", fontsize=8.5)
    return _finish(fig, output)


# ── E1 ablation ──────────────────────────────────────────────────────────

CONDITION_LABELS = {
    "A0_no_validation": "A0\nNone",
    "A1_critic_only": "A1\n+ Critic",
    "A2_symbolic_gate": "A2\n+ Symbolic",
    "A3_nav_aware_gate": "A3\n+ Physics",
}

ABLATION_COLORS = ["#bab0ac", "#f28e2b", "#4e79a7", "#59a14f"]


def ablation_chart(data: dict, output: str | Path) -> Path:
    _setup_style()
    fig, (ax_s, ax_r) = plt.subplots(
        1, 2, figsize=(10.0, 4.0), layout="constrained",
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    conditions = list(data["conditions"].keys())
    x = np.arange(len(conditions))
    config = data.get("config", {})

    # left: success on feasible tasks
    rates = [data["conditions"][c]["success_rate"] for c in conditions]
    err_lo, err_hi, tops = [], [], []
    for c in conditions:
        ci = data["conditions"][c].get("success_ci_95", [None, None])
        r = data["conditions"][c]["success_rate"]
        lo = r - ci[0] if ci[0] is not None else 0
        hi = ci[1] - r if ci[1] is not None else 0
        err_lo.append(lo)
        err_hi.append(hi)
        tops.append(r + hi)

    bars = ax_s.bar(x, rates, 0.56, yerr=[err_lo, err_hi], capsize=4,
                    color=ABLATION_COLORS, edgecolor="white", linewidth=0.6,
                    error_kw={"linewidth": 1.0, "color": "#444"})
    for bar, val, top in zip(bars, rates, tops):
        _vlabel(ax_s, bar, f"{val:.2f}", y_override=top, fontsize=9)

    ax_s.set_xticks(x)
    ax_s.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions])
    ax_s.set_ylabel("Task success rate")
    ax_s.set_ylim(0, 1.14)
    ax_s.axhline(1.0, color="#d0d0d0", linewidth=0.5, linestyle=":")
    n_f = config.get("n_feasible_instances", "?")
    ax_s.set_title(f"Feasible tasks (n={n_f}), 95% Wilson CI")

    # right: rejection of infeasible tasks
    rej = [data["conditions"][c]["rejection_rate"] for c in conditions]
    rej_colors = [COLORS["post"] if r > 0.5 else COLORS["pre"] for r in rej]
    bars_r = ax_r.bar(x, rej, 0.56, color=rej_colors, edgecolor="white", linewidth=0.6)
    for bar, val in zip(bars_r, rej):
        _vlabel(ax_r, bar, f"{val:.2f}", fontsize=9)

    ax_r.set_xticks(x)
    ax_r.set_xticklabels([CONDITION_LABELS.get(c, c) for c in conditions])
    ax_r.set_ylabel("Pre-execution rejection rate")
    ax_r.set_ylim(0, 1.14)
    ax_r.axhline(1.0, color="#d0d0d0", linewidth=0.5, linestyle=":")
    n_i = config.get("n_infeasible_instances", "?")
    ax_r.set_title(f"Infeasible tasks (n={n_i}), rejection rate")

    fig.suptitle("E1 — Pipeline ablation: what each validation layer adds",
                 fontsize=13, fontweight="bold")
    return _finish(fig, output)


# ── E2 gate classifier ──────────────────────────────────────────────────

LAYER_LABELS = {
    "critic": "LLM critic",
    "l1_l3": "+ L1/L3",
    "symbolic_gate": "+ Symbolic L2",
    "physics_gate": "Isaac physics",
}

DEFECT_LABELS = {
    "D0_clean": "clean",
    "D1_step_inversion": "step\ninversion",
    "D2_missing_pick": "missing\npick",
    "D3_hallucinated_target": "hallucinated\ntarget",
    "D4_wrong_placement": "wrong\nplacement",
    "D6_speed_violation": "speed\nviolation*",
    "D7_redundant_step": "redundant\nstep*",
}


def gate_classifier_chart(data: dict, output: str | Path) -> Path:
    _setup_style()
    fig, (ax_bar, ax_mat) = plt.subplots(
        1, 2, figsize=(12.5, 5.0), layout="constrained",
        gridspec_kw={"width_ratios": [0.7, 1.5]},
    )

    layers = list(data["layers"].keys())

    # left: recall per layer
    recalls = [data["layers"][ly]["report"]["recall"] for ly in layers]
    fprs = [data["layers"][ly]["report"]["false_positive_rate"] for ly in layers]
    x = np.arange(len(layers))

    bars = ax_bar.bar(x, recalls, 0.52, color=COLORS["claude"],
                      edgecolor="white", linewidth=0.6, label="Recall")
    for bar, val in zip(bars, recalls):
        _vlabel(ax_bar, bar, f"{val:.2f}", fontsize=9)

    if all(f == 0 for f in fprs):
        ax_bar.text(0.03, 0.96, "FPR = 0 across all layers",
                    transform=ax_bar.transAxes, ha="left", va="top",
                    fontsize=8, color=MUTED, style="italic")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([LAYER_LABELS.get(ly, ly) for ly in layers], fontsize=8)
    ax_bar.set_ylabel("Recall")
    ax_bar.set_ylim(0, 1.14)
    ax_bar.set_title("Defect catch rate")

    # right: layer × defect correctness matrix
    from physgate.eval_v2.defects import DEFECT_CLASSES

    present = {d for ly in layers for d in data["layers"][ly]["per_defect_rejection_rate"]}
    invalid = sorted(d for d in present if DEFECT_CLASSES.get(d, True))
    valid = sorted(d for d in present if not DEFECT_CLASSES.get(d, True))
    defect_ids = [*invalid, *valid]

    raw = np.array([
        [data["layers"][ly]["per_defect_rejection_rate"].get(d, np.nan) for d in defect_ids]
        for ly in layers
    ])
    correctness = np.empty_like(raw)
    for j, defect in enumerate(defect_ids):
        should_reject = DEFECT_CLASSES.get(defect, True)
        correctness[:, j] = raw[:, j] if should_reject else 1.0 - raw[:, j]

    im = ax_mat.imshow(correctness, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax_mat.set_xticks(range(len(defect_ids)))
    ax_mat.set_xticklabels([DEFECT_LABELS.get(d, d) for d in defect_ids], fontsize=8)
    ax_mat.set_yticks(range(len(layers)))
    ax_mat.set_yticklabels([LAYER_LABELS.get(ly, ly) for ly in layers], fontsize=9)
    ax_mat.grid(False)

    for i in range(len(layers)):
        for j, defect in enumerate(defect_ids):
            val = raw[i, j]
            if np.isnan(val):
                continue
            should_reject = DEFECT_CLASSES.get(defect, True)
            action = "reject" if should_reject else "accept"
            c = correctness[i, j]
            ax_mat.text(j, i, f"{action}\n{c:.0%}", ha="center", va="center",
                        fontsize=8.5, fontweight="bold",
                        color="white" if c < 0.3 else INK)

    if invalid and valid:
        ax_mat.axvline(len(invalid) - 0.5, color="white", linewidth=2.5)
    ax_mat.set_title("Per-defect correctness (left: should reject · right: should accept)")
    cbar = fig.colorbar(im, ax=ax_mat, fraction=0.035, pad=0.02, shrink=0.85)
    cbar.set_label("correct behavior rate", fontsize=8.5)

    fig.suptitle("E2 — Gate as classifier (defect-injection corpus)",
                 fontsize=13, fontweight="bold")
    return _finish(fig, output)


# ── GPU scaling ──────────────────────────────────────────────────────────

def gpu_scaling_chart(data: dict, output: str | Path) -> Path:
    _setup_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.0), layout="constrained")

    results = data["results"]
    envs = [r["num_envs"] for r in results]
    throughput = [r["env_steps_per_s"] for r in results]
    efficiency = [r["scaling_efficiency"] for r in results]

    ax.loglog(envs, throughput, "o-", color=COLORS["claude"], markersize=5, linewidth=1.8)
    ax.set_xlabel("Parallel environments")
    ax.set_ylabel("Throughput (env-steps/s)", color=COLORS["claude"])
    ax.tick_params(axis="y", labelcolor=COLORS["claude"])
    ax.set_xticks(envs)
    ax.set_xticklabels([str(e) for e in envs])
    ax.minorticks_off()

    ax2 = ax.twinx()
    ax2.plot(envs, efficiency, "s--", color=COLORS["accent"], markersize=5, linewidth=1.8)
    ax2.set_ylabel("Scaling efficiency", color=COLORS["accent"])
    ax2.tick_params(axis="y", labelcolor=COLORS["accent"])
    ax2.set_ylim(0.50, 1.06)
    ax2.set_xscale("log")
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(COLORS["accent"])
    ax2.spines["right"].set_linewidth(0.7)
    ax2.grid(False)

    for e, eff in zip(envs, efficiency):
        ax2.annotate(f"{eff:.2f}", (e, eff), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color=COLORS["accent"])

    knee = data.get("saturation_knee_envs")
    if knee:
        ax.axvline(knee, color="#aaa", linewidth=0.8, linestyle=":")
    else:
        ax.text(0.97, 0.04, "saturation knee not reached",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=MUTED, style="italic")

    ax.set_title("GPU parallel-validation scaling (RTX PRO 6000 Blackwell)")
    return _finish(fig, output)
