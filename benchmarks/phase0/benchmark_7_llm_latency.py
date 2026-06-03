#!/usr/bin/env python3
"""Phase 0 benchmark #7: LLM planning latency P50/P95.

Measures the real planning bottleneck (architecture doc §6 estimates the LLM at
75-85% of pipeline wall-clock). Two strategies are measured:

  A. single-call    — one Claude call that returns all N=8 candidate plans
                      (what ClaudePlanner does today)
  B. parallel-calls — 8 concurrent Claude calls, one candidate each
                      (the original design sketch in benchmarks/phase0/README.md)

plus the safety-critic call latency. Results go to
``benchmarks/phase0/results/benchmark_7_llm_latency.json`` (+ a markdown summary).

Requires ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN. Handles 429s with backoff.

Usage:
    source ~/.config/physgate/credentials.env
    python benchmarks/phase0/benchmark_7_llm_latency.py --trials 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from physgate.examples_lib.fetch_and_place import build_demo_scene
from physgate.planner.critic import ClaudeCritic
from physgate.planner.planner import (
    DEFAULT_PLANNER_MODEL,
    ClaudePlanner,
    llm_credentials_available,
    make_anthropic_client,
)
from physgate.planner.schemas import DEFAULT_NUM_CANDIDATES

TASK = "put the fallen box back on shelf A"
RESULTS_DIR = Path(__file__).parent / "results"


def _percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    k = (len(values) - 1) * pct / 100
    lower = int(k)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (k - lower)


def _with_retries(fn, max_attempts: int = 5, base_delay: float = 20.0):
    """Run fn(), retrying on rate limits with exponential backoff."""
    import anthropic

    for attempt in range(max_attempts):
        try:
            return fn()
        except anthropic.RateLimitError:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2**attempt)
            print(f"    [429 rate-limited; retrying in {delay:.0f}s "
                  f"(attempt {attempt + 1}/{max_attempts})]")
            time.sleep(delay)


# --------------------------------------------------------- strategy A: single call


def measure_single_call(trials: int, scene) -> dict:
    """One call returns all 8 candidate plans (current ClaudePlanner behaviour)."""
    planner = ClaudePlanner()
    latencies, plan_counts = [], []
    for trial in range(trials):
        print(f"  [A single-call] trial {trial + 1}/{trials}...")
        t0 = time.perf_counter()
        plans = _with_retries(lambda: planner(TASK, scene, DEFAULT_NUM_CANDIDATES, None))
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        plan_counts.append(len(plans))
        print(f"    {elapsed:.1f}s -> {len(plans)} valid plans")
    return {
        "strategy": "single_call_8_plans",
        "latencies_s": [round(v, 2) for v in latencies],
        "p50_s": round(_percentile(latencies, 50), 2),
        "p95_s": round(_percentile(latencies, 95), 2),
        "valid_plans_per_call": plan_counts,
    }


# ------------------------------------------------------ strategy B: parallel calls


async def _one_plan_call(async_client, scene_payload: str, plan_index: int) -> float:
    """One async call generating a single candidate plan. Returns latency."""
    from physgate.planner.planner import _SYSTEM_PROMPT

    t0 = time.perf_counter()
    await async_client.messages.create(
        model=DEFAULT_PLANNER_MODEL,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Task: {TASK}\n\nCurrent scene:\n{scene_payload}\n\n"
                    f"Generate exactly 1 candidate plan (variant #{plan_index}) as a JSON array."
                ),
            }
        ],
    )
    return time.perf_counter() - t0


async def _parallel_trial(scene_payload: str, n: int) -> tuple[float, list[float]]:
    import os

    import anthropic

    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if auth_token:
        async_client = anthropic.AsyncAnthropic(
            auth_token=auth_token,
            default_headers={"anthropic-beta": "oauth-2025-04-20"},
        )
    else:
        async_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    t0 = time.perf_counter()
    per_call = await asyncio.gather(
        *[_one_plan_call(async_client, scene_payload, i) for i in range(n)]
    )
    wall = time.perf_counter() - t0
    return wall, list(per_call)


def measure_parallel_calls(trials: int, scene) -> dict:
    """8 concurrent calls, one plan each (README design sketch)."""
    from physgate.world.scene_graph import to_query_scene_payload

    scene_payload = json.dumps(to_query_scene_payload(scene), indent=2)
    walls, all_calls = [], []
    for trial in range(trials):
        print(f"  [B parallel x{DEFAULT_NUM_CANDIDATES}] trial {trial + 1}/{trials}...")
        wall, per_call = _with_retries(
            lambda: asyncio.run(_parallel_trial(scene_payload, DEFAULT_NUM_CANDIDATES))
        )
        walls.append(wall)
        all_calls.extend(per_call)
        print(f"    wall {wall:.1f}s | per-call p50 {statistics.median(per_call):.1f}s")
    return {
        "strategy": f"parallel_{DEFAULT_NUM_CANDIDATES}_calls",
        "wall_latencies_s": [round(v, 2) for v in walls],
        "wall_p50_s": round(_percentile(walls, 50), 2),
        "wall_p95_s": round(_percentile(walls, 95), 2),
        "per_call_p50_s": round(_percentile(all_calls, 50), 2),
        "per_call_p95_s": round(_percentile(all_calls, 95), 2),
    }


# ------------------------------------------------------------------ critic latency


def measure_critic(trials: int, scene) -> dict:
    """Critic call latency over the 8 mock candidates."""
    from physgate.planner.planner import MockPlanner

    critic = ClaudeCritic()
    plans = MockPlanner()(TASK, scene, DEFAULT_NUM_CANDIDATES, None)
    latencies, survivor_counts = [], []
    for trial in range(trials):
        print(f"  [critic] trial {trial + 1}/{trials}...")
        t0 = time.perf_counter()
        survivors = _with_retries(lambda: critic(plans, scene))
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        survivor_counts.append(len(survivors))
        print(f"    {elapsed:.1f}s -> {len(survivors)}/{len(plans)} plans survived")
    return {
        "latencies_s": [round(v, 2) for v in latencies],
        "p50_s": round(_percentile(latencies, 50), 2),
        "p95_s": round(_percentile(latencies, 95), 2),
        "survivors_per_call": survivor_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=3, help="trials per strategy")
    parser.add_argument("--skip-parallel", action="store_true", help="skip strategy B")
    args = parser.parse_args()

    if not llm_credentials_available():
        print("ERROR: set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN first "
              "(source ~/.config/physgate/credentials.env)")
        return 1

    # quick credential sanity check
    make_anthropic_client()
    print(f"model: {DEFAULT_PLANNER_MODEL}; credentials OK; running {args.trials} trials per strategy\n")

    scene = build_demo_scene()
    results: dict = {
        "benchmark": "phase0 #7 - LLM planning latency",
        "model": DEFAULT_PLANNER_MODEL,
        "n_candidates": DEFAULT_NUM_CANDIDATES,
        "trials": args.trials,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=== strategy A: single call returning 8 plans ===")
    results["single_call"] = measure_single_call(args.trials, scene)

    if not args.skip_parallel:
        print("\n=== strategy B: 8 parallel calls, 1 plan each ===")
        results["parallel_calls"] = measure_parallel_calls(args.trials, scene)

    print("\n=== safety critic ===")
    results["critic"] = measure_critic(args.trials, scene)

    # ---- write results ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "benchmark_7_llm_latency.json"
    json_path.write_text(json.dumps(results, indent=2))

    # markdown summary
    a = results["single_call"]
    c = results["critic"]
    lines = [
        "# Phase 0 benchmark #7 — LLM planning latency",
        "",
        f"Model: `{results['model']}` · N={results['n_candidates']} candidates · "
        f"{args.trials} trials · {results['timestamp']}",
        "",
        "| Stage | P50 | P95 |",
        "|---|---|---|",
        f"| Planner (1 call, 8 plans) | {a['p50_s']} s | {a['p95_s']} s |",
    ]
    if "parallel_calls" in results:
        b = results["parallel_calls"]
        lines.append(f"| Planner (8 parallel calls) | {b['wall_p50_s']} s | {b['wall_p95_s']} s |")
    lines += [
        f"| Safety critic | {c['p50_s']} s | {c['p95_s']} s |",
        "",
        "**Gate criterion check:** the architecture estimates LLM planning at",
        "10-50 s and 75-85% of pipeline wall-clock. Compare with the L2 physics",
        "validation wall-clock from benchmark #4 to confirm the bottleneck.",
    ]
    md_path = RESULTS_DIR / "benchmark_7_llm_latency.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(f"\nresults written to {json_path} and {md_path}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
