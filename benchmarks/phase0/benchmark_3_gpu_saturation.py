#!/usr/bin/env python3
"""Phase 0 benchmark #3: GPU-utilization curve, Go2 envs 1 -> 1024.

Finds the env-count saturation knee on this GPU: total physics throughput
(env-steps/sec) should scale ~linearly with num_envs until the GPU saturates.
The knee tells us the *real* value of N for best-of-N validation (the
architecture warns N=8 is far below saturation — this measures by how much).

Each env count runs in a SUBPROCESS (Isaac's SimulationApp can only launch once
per process). Run inside the env_isaaclab venv:

    python benchmarks/phase0/benchmark_3_gpu_saturation.py
    python benchmarks/phase0/benchmark_3_gpu_saturation.py --worker 64   # (internal)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ENV_COUNTS = [1, 4, 16, 64, 256, 1024]
STEPS_PER_RUN = 200
RESULTS_DIR = Path(__file__).parent / "results"


def worker(num_envs: int) -> None:
    """Subprocess body: create the Go2 fetch scene with N envs, measure steps/sec."""
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=True).app  # noqa: F841

    import torch  # noqa: F401

    from physgate.gate.reset_workaround import reset_scene_to_identical_state
    from physgate.world.fetch_scene import FetchSimWorld

    t_create0 = time.perf_counter()
    world = FetchSimWorld(num_envs=num_envs)
    create_time = time.perf_counter() - t_create0

    reset_scene_to_identical_state(world.scene, world.sim)

    # warm up (JIT, GPU kernels)
    for _ in range(20):
        world.step()

    t0 = time.perf_counter()
    for _ in range(STEPS_PER_RUN):
        world.step()
    elapsed = time.perf_counter() - t0

    physics_steps_per_s = STEPS_PER_RUN / elapsed
    env_steps_per_s = physics_steps_per_s * num_envs

    print(
        "RESULT_JSON: "
        + json.dumps(
            {
                "num_envs": num_envs,
                "scene_creation_s": round(create_time, 2),
                "physics_steps_per_s": round(physics_steps_per_s, 1),
                "env_steps_per_s": round(env_steps_per_s, 1),
                "wall_s_for_200_steps": round(elapsed, 3),
            }
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=int, default=None, help="(internal) run as worker for N envs")
    parser.add_argument("--env-counts", type=int, nargs="+", default=ENV_COUNTS)
    args = parser.parse_args()

    if args.worker is not None:
        worker(args.worker)
        return 0

    results = []
    for n in args.env_counts:
        print(f"=== benchmarking {n} envs (subprocess) ===")
        proc = subprocess.run(
            [sys.executable, __file__, "--worker", str(n)],
            capture_output=True,
            text=True,
            timeout=900,
        )
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON: "):
                entry = json.loads(line[len("RESULT_JSON: "):])
                results.append(entry)
                print(f"    {entry['env_steps_per_s']:,.0f} env-steps/s "
                      f"({entry['physics_steps_per_s']:.0f} physics steps/s)")
                break
        else:
            print(f"    WORKER FAILED for {n} envs:\n{proc.stdout[-500:]}\n{proc.stderr[-500:]}")

    # find the knee: largest env count whose per-env efficiency >= 70% of the 1-env baseline
    knee = None
    if results and results[0]["num_envs"] == 1:
        baseline_eff = results[0]["env_steps_per_s"]  # 1 env => env_steps == physics_steps
        for entry in results:
            efficiency = entry["env_steps_per_s"] / (entry["num_envs"] * baseline_eff)
            entry["scaling_efficiency"] = round(efficiency, 3)
            if efficiency >= 0.7:
                knee = entry["num_envs"]

    output = {
        "benchmark": "phase0 #3 - GPU saturation curve (Go2 fetch scene)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": "RTX PRO 6000 Blackwell Max-Q (sm_120, 300W)",
        "results": results,
        "saturation_knee_envs": knee,
        "interpretation": (
            f"GPU scales near-linearly up to ~{knee} envs; best-of-N validation with "
            f"N=8 uses a small fraction of available parallelism." if knee else "incomplete"
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_3_gpu_saturation.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nresults -> {out_path}")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
