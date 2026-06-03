#!/usr/bin/env python3
"""Phase 0 benchmark #6: Warp/Blackwell first-run JIT time + kernel-cache persistence.

Go/No-Go criterion (benchmarks/phase0/README.md): if first-run JIT compilation
takes >1 h on sm_120 and the kernel cache does not persist, dev iteration on
the Newton physics engine is blocked.

SCOPE NOTE: the Newton engine itself is NOT in the current stack (Isaac Lab
2.3.2 ships PhysX; ``import newton`` fails in this venv). What we CAN measure
honestly today is **Warp** — the kernel framework Newton is built on, which IS
installed (warp-lang) and which exhibits the same sm_120 JIT/caching behaviour
Newton would. PhysX (the current engine) has no JIT issue; its startup cost is
covered by benchmark #5 (warm-start, 3.8 s app launch).

Method — three subprocess runs of the same representative kernel set:
  1. COLD   fresh kernel-cache dir  -> full sm_120 JIT compile time
  2. WARM   same cache dir, new process -> load-from-cache time
  3. WARM2  repeat to confirm stability

Run inside the env_isaaclab venv:

    python benchmarks/phase0/benchmark_6_jit_cache.py
    python benchmarks/phase0/benchmark_6_jit_cache.py --worker /tmp/cache_dir  # (internal)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
JIT_BLOCKER_THRESHOLD_S = 3600.0  # the ">1 h" go/no-go line


def worker(cache_dir: str) -> None:
    """Subprocess body: compile a representative kernel set with the given cache dir."""
    import os

    # the cache dir must be set BEFORE warp initializes
    os.environ["WARP_CACHE_PATH"] = cache_dir

    t_import0 = time.perf_counter()
    import warp as wp

    wp.config.kernel_cache_dir = cache_dir
    wp.init()
    import_init_s = time.perf_counter() - t_import0

    # ---- representative kernel set (sim-like workloads) ----
    @wp.kernel
    def integrate_bodies(
        x: wp.array(dtype=wp.vec3),
        v: wp.array(dtype=wp.vec3),
        f: wp.array(dtype=wp.vec3),
        inv_mass: wp.array(dtype=float),
        gravity: wp.vec3,
        dt: float,
    ):
        tid = wp.tid()
        v[tid] = v[tid] + (f[tid] * inv_mass[tid] + gravity) * dt
        x[tid] = x[tid] + v[tid] * dt

    @wp.kernel
    def transform_points(
        points: wp.array(dtype=wp.vec3),
        rotations: wp.array(dtype=wp.quat),
        translations: wp.array(dtype=wp.vec3),
        out: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()
        out[tid] = wp.quat_rotate(rotations[tid], points[tid]) + translations[tid]

    @wp.kernel
    def contact_reduction(
        positions: wp.array(dtype=wp.vec3),
        plane_height: float,
        contact_count: wp.array(dtype=wp.int32),
        penetration_sum: wp.array(dtype=float),
    ):
        tid = wp.tid()
        depth = plane_height - positions[tid][2]
        if depth > 0.0:
            wp.atomic_add(contact_count, 0, 1)
            wp.atomic_add(penetration_sum, 0, depth)

    @wp.kernel
    def joint_pd_control(
        q: wp.array(dtype=float),
        qd: wp.array(dtype=float),
        targets: wp.array(dtype=float),
        kp: float,
        kd: float,
        tau: wp.array(dtype=float),
    ):
        tid = wp.tid()
        tau[tid] = kp * (targets[tid] - q[tid]) - kd * qd[tid]

    # ---- force compilation by launching every kernel on the GPU ----
    n = 4096
    device = "cuda:0"
    x = wp.zeros(n, dtype=wp.vec3, device=device)
    v = wp.zeros(n, dtype=wp.vec3, device=device)
    f = wp.zeros(n, dtype=wp.vec3, device=device)
    inv_mass = wp.ones(n, dtype=float, device=device)
    quats = wp.zeros(n, dtype=wp.quat, device=device)
    out = wp.zeros(n, dtype=wp.vec3, device=device)
    contact_count = wp.zeros(1, dtype=wp.int32, device=device)
    penetration = wp.zeros(1, dtype=float, device=device)
    q = wp.zeros(n, dtype=float, device=device)
    qd = wp.zeros(n, dtype=float, device=device)
    targets = wp.ones(n, dtype=float, device=device)
    tau = wp.zeros(n, dtype=float, device=device)

    t_compile0 = time.perf_counter()
    wp.launch(
        integrate_bodies,
        dim=n,
        device=device,
        inputs=[x, v, f, inv_mass, wp.vec3(0.0, 0.0, -9.81), 0.005],
    )
    wp.launch(transform_points, dim=n, device=device, inputs=[x, quats, v, out])
    wp.launch(
        contact_reduction,
        dim=n,
        device=device,
        inputs=[x, 0.0, contact_count, penetration],
    )
    wp.launch(joint_pd_control, dim=n, device=device, inputs=[q, qd, targets, 25.0, 0.5, tau])
    wp.synchronize()
    compile_and_launch_s = time.perf_counter() - t_compile0

    cache_size_kb = sum(p.stat().st_size for p in Path(cache_dir).rglob("*") if p.is_file()) // 1024

    print(
        "RESULT_JSON: "
        + json.dumps(
            {
                "warp_version": wp.config.version,
                "device": wp.get_device(device).name,
                "arch": wp.get_device(device).arch,
                "import_init_s": round(import_init_s, 2),
                "compile_and_launch_s": round(compile_and_launch_s, 2),
                "kernel_cache_dir": cache_dir,
                "cache_size_kb": cache_size_kb,
            }
        )
    )


def run_worker(cache_dir: str) -> dict | None:
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", cache_dir],
        capture_output=True,
        text=True,
        timeout=7200,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON: "):
            return json.loads(line[len("RESULT_JSON: ") :])
    print(f"    WORKER FAILED:\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=str, default=None, help="(internal) cache dir to use")
    args = parser.parse_args()

    if args.worker is not None:
        worker(args.worker)
        return 0

    cache_dir = tempfile.mkdtemp(prefix="warp_jit_bench_")
    try:
        print(f"=== run 1: COLD JIT (fresh cache: {cache_dir}) ===")
        cold = run_worker(cache_dir)
        if cold:
            print(
                f"    compile+launch {cold['compile_and_launch_s']}s | cache {cold['cache_size_kb']} KB"
            )

        print("=== run 2: WARM (same cache, new process) ===")
        warm = run_worker(cache_dir)
        if warm:
            print(f"    compile+launch {warm['compile_and_launch_s']}s")

        print("=== run 3: WARM repeat (stability) ===")
        warm2 = run_worker(cache_dir)
        if warm2:
            print(f"    compile+launch {warm2['compile_and_launch_s']}s")
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    if not (cold and warm and warm2):
        print("benchmark incomplete — see worker failures above")
        return 1

    # cached loads are often below timing resolution — report the absolute number
    # and a bounded speedup statement, never a divide-by-near-zero ratio
    cached_s = warm["compile_and_launch_s"]
    cold_s = cold["compile_and_launch_s"]
    if cached_s < 0.01:
        cache_statement = f"cached load <0.01 s (below timing resolution; cold was {cold_s:.2f} s)"
        cache_speedup = None
    else:
        cache_speedup = round(cold_s / cached_s, 1)
        cache_statement = (
            f"{cache_speedup}x faster when cached ({cold_s:.2f} s -> {cached_s:.2f} s)"
        )
    cache_persists = cached_s < cold_s * 0.5
    jit_fast_enough = cold_s < JIT_BLOCKER_THRESHOLD_S

    output = {
        "benchmark": "phase0 #6 - Warp (Newton foundation) JIT + kernel-cache persistence on sm_120",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": "RTX PRO 6000 Blackwell Max-Q (sm_120, 300W)",
        "scope_note": (
            "Newton engine not in current stack (Isaac Lab 2.3.2 = PhysX); measured Warp "
            f"{cold['warp_version']}, the kernel framework Newton is built on. PhysX startup "
            "cost is covered by benchmark #5 (3.8 s app launch, no JIT)."
        ),
        "cold_run": cold,
        "warm_run": warm,
        "warm_run_repeat": warm2,
        "analysis": {
            "cold_jit_s": cold_s,
            "cached_s": cached_s,
            "cache_speedup_x": cache_speedup,
            "cache_statement": cache_statement,
            "cache_persists_across_processes": cache_persists,
            "jit_under_1h_threshold": jit_fast_enough,
        },
        "go_no_go": {
            "verdict": "GO" if (jit_fast_enough and cache_persists) else "NO-GO",
            "criterion": ">1 h JIT with no cache persistence would block dev iteration",
        },
        "interpretation": (
            f"sm_120 cold JIT for a representative sim-kernel set takes {cold_s:.1f} s "
            f"(threshold: 1 h) and the kernel cache persists across processes "
            f"({cache_statement}) — Newton/Warp dev iteration on Blackwell is not blocked. "
            f"Re-run against Newton itself when it enters the stack."
            if (jit_fast_enough and cache_persists)
            else "JIT or cache persistence failed the go/no-go line — see analysis."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_6_jit_cache.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"\nresults -> {out_path}")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
