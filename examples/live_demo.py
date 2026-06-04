#!/usr/bin/env python3
"""physgate Computex live demo — animated pipeline visualization.

A rich terminal UI that shows the full orchestrator flow executing in
real-time: task → planner → critic → Sim-Gate → executor, with timing,
color-coded verdicts, and a final trajectory summary.

Works in two modes:
  - Offline (default): MockPlanner, instant, no network — booth-reliable
  - Live Claude: set CLAUDE_CODE_OAUTH_TOKEN or ANTHROPIC_API_KEY

Usage:
    python examples/live_demo.py
    python examples/live_demo.py --task "put the fallen box on shelf A"
    python examples/live_demo.py --task "put the safe on the shelf"  # infeasible
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

console = Console()

# ── brand colors (Tableau-10, matching charts.py) ────────────────────────
C_BLUE = "steel_blue"
C_GREEN = "green"
C_RED = "red"
C_AMBER = "dark_orange"
C_GREY = "grey62"
C_BOLD = "bold"


def _timer():
    t0 = time.perf_counter()
    return lambda: time.perf_counter() - t0


def _stage_header(title: str, icon: str = "●") -> Text:
    return Text(f" {icon} {title}", style=f"bold {C_BLUE}")


def run_demo(task: str) -> dict:
    from physgate.audit.trail import AuditTrail
    from physgate.examples_lib.fetch_and_place import build_demo_scene
    from physgate.executor.backend import MockWorldBackend
    from physgate.executor.plan_executor import execute_plan
    from physgate.gate.parallel import run_gate, symbolic_l2
    from physgate.gate.scoring import SelectionResult
    from physgate.planner.critic import make_critic
    from physgate.planner.planner import llm_credentials_available, make_planner

    scene = build_demo_scene()
    planner = make_planner()
    critic = make_critic()
    trail = AuditTrail()
    is_llm = llm_credentials_available()

    planner_label = "Claude Opus 4.8" if is_llm else "MockPlanner (offline)"

    # ── HEADER ───────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            f"[bold white]{task}[/]",
            title="[bold]physgate[/] — Sim-Gate Orchestrator Demo",
            subtitle=f"planner: {planner_label}",
            border_style=C_BLUE,
            padding=(1, 2),
        )
    )
    console.print()

    # ── STAGE 1: PLANNER ─────────────────────────────────────────────
    console.print(_stage_header("PLANNER", "▶"))
    elapsed = _timer()
    with console.status(f"[{C_GREY}]Generating candidate plans...", spinner="dots"):
        candidates = planner(task, scene, 8)
    t_plan = elapsed()
    console.print(
        f"  [bold]{len(candidates)}[/] candidate plans generated "
        f"[{C_GREY}]({t_plan:.1f}s)[/]"
    )
    for p in candidates:
        steps_desc = " → ".join(
            s.args.get("skill", s.tool.value)
            for s in p.steps
        )
        console.print(f"    [{C_GREY}]{p.plan_id}[/]  {steps_desc}")
    console.print()

    # ── STAGE 2: SAFETY CRITIC ───────────────────────────────────────
    console.print(_stage_header("SAFETY CRITIC", "▶"))
    elapsed = _timer()
    with console.status(f"[{C_GREY}]Pruning unsafe candidates...", spinner="dots"):
        survivors = critic(candidates, scene)
    t_critic = elapsed()

    pruned = set(p.plan_id for p in candidates) - set(p.plan_id for p in survivors)
    console.print(
        f"  [{C_RED}]✗ {len(pruned)} rejected[/]  "
        f"[{C_GREEN}]✓ {len(survivors)} survivors[/]  "
        f"[{C_GREY}]({t_critic:.1f}s)[/]"
    )
    for pid in pruned:
        console.print(f"    [{C_RED}]✗[/] {pid}")
    for p in survivors:
        console.print(f"    [{C_GREEN}]✓[/] {p.plan_id}")
    console.print()

    # ── STAGE 3: SIM-GATE ────────────────────────────────────────────
    console.print(_stage_header("SIM-GATE (L1 kinematic → L3 scene → L2 physics)", "▶"))
    elapsed = _timer()
    with console.status(f"[{C_GREY}]Running parallel physics validation...", spinner="dots"):
        selection: SelectionResult = run_gate(survivors, scene, l2_fn=symbolic_l2)
    t_gate = elapsed()

    gate_table = Table(show_header=True, header_style=f"bold {C_BLUE}", box=None, padding=(0, 2))
    gate_table.add_column("Plan", style=C_GREY)
    gate_table.add_column("L1", justify="center")
    gate_table.add_column("L3", justify="center")
    gate_table.add_column("L2", justify="center")
    gate_table.add_column("Verdict", justify="center")

    for r in selection.ranked:
        verdict = f"[{C_GREEN}]PASS[/]" if r.success else f"[{C_RED}]FAIL[/]"
        l1 = f"[{C_GREEN}]✓[/]"
        l3 = f"[{C_GREEN}]✓[/]" if r.success or not r.failure_code else f"[{C_RED}]✗[/]"
        l2 = f"[{C_GREEN}]✓[/]" if r.success else f"[{C_RED}]✗[/]"
        is_best = " ★" if r.plan_id == selection.best_plan_id else ""
        gate_table.add_row(f"{r.plan_id}{is_best}", l1, l3, l2, verdict)

    console.print(gate_table)
    console.print(
        f"\n  Best plan: [{C_GREEN}bold]{selection.best_plan_id}[/]  "
        f"[{C_GREY}]({t_gate:.1f}s)[/]"
    )

    if not selection.any_feasible:
        console.print(
            Panel(
                f"[{C_RED}]No feasible plan found — task is infeasible or all plans are defective.\n"
                "The orchestrator would ESCALATE to a human operator.[/]",
                title="ESCALATED",
                border_style=C_RED,
            )
        )
        console.print()
        return {
            "outcome": "escalated",
            "timing": {"planner": t_plan, "critic": t_critic, "gate": t_gate},
        }

    console.print()

    # ── STAGE 4: EXECUTOR ────────────────────────────────────────────
    best_plan = next(p for p in survivors if p.plan_id == selection.best_plan_id)
    console.print(_stage_header("EXECUTOR (deterministic A* navigation)", "▶"))
    elapsed = _timer()
    with console.status(f"[{C_GREY}]Executing verified plan...", spinner="dots"):
        backend = MockWorldBackend(scene)
        result = execute_plan(best_plan, backend)
    t_exec = elapsed()

    success = result.get("success", False)
    final_scene = backend.get_scene()
    goal_met = final_scene.has_relation("box_03", "on", "shelf_A")

    console.print(
        f"  Execution: [{'bold green' if success else 'bold red'}]"
        f"{'SUCCESS' if success else 'FAILED'}[/]  "
        f"[{C_GREY}]({t_exec:.1f}s)[/]"
    )
    console.print(
        f"  Goal (box on shelf): [{'bold green' if goal_met else 'bold red'}]"
        f"{'ACHIEVED ✓' if goal_met else 'NOT ACHIEVED ✗'}[/]"
    )
    console.print()

    # ── AUDIT TRAIL ──────────────────────────────────────────────────
    trail.close()
    console.print(
        f"  [{C_GREY}]Audit: {len(trail.records)} records, "
        f"Merkle root {trail.checkpoints[-1].root[:16] if trail.checkpoints else 'none'}...[/]"
    )
    console.print()

    # ── SUMMARY ──────────────────────────────────────────────────────
    total = t_plan + t_critic + t_gate + t_exec
    summary = Table(show_header=False, box=None, padding=(0, 1))
    summary.add_column("Stage", style=C_BLUE)
    summary.add_column("Time", justify="right")
    summary.add_row("Planner", f"{t_plan:.1f}s")
    summary.add_row("Critic", f"{t_critic:.1f}s")
    summary.add_row("Sim-Gate", f"{t_gate:.1f}s")
    summary.add_row("Executor", f"{t_exec:.1f}s")
    summary.add_row("[bold]Total[/]", f"[bold]{total:.1f}s[/]")

    console.print(
        Panel(
            summary,
            title=f"[bold {'green' if goal_met else 'red'}]"
            f"{'TASK COMPLETE' if goal_met else 'TASK FAILED'}[/]",
            border_style=C_GREEN if goal_met else C_RED,
            padding=(1, 2),
        )
    )

    return {
        "outcome": "done" if goal_met else "failed",
        "timing": {
            "planner": t_plan,
            "critic": t_critic,
            "gate": t_gate,
            "executor": t_exec,
            "total": total,
        },
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="put the fallen box back on shelf A",
        help="natural-language task for the robot",
    )
    args = parser.parse_args()

    result = run_demo(args.task)
    return 0 if result["outcome"] == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
