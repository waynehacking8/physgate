# Phase 0 — Benchmark #1: Hardware Gate Report (gpu_burn chip-reset test)

**Date:** 2026-06-03
**Machine:** RTX PRO 6000 Blackwell workstation (sm_120)
**Operator:** Wayne (waynehacking8)

## Environment Audit

| Item | Value | Gate criterion | Verdict |
|---|---|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell, 97,887 MiB VRAM | — | ✅ |
| Driver | **580.159.03** | Must be 580.x (NOT 595.x) | ✅ PASS |
| CUDA (driver API) | 13.0 | — | ✅ |
| CUDA toolkit | 12.8 (`/usr/local/cuda`) ⚠️ `/usr/bin/nvcc` is stale 11.5 — always use `/usr/local/cuda/bin/nvcc` | 12.8 known-good | ✅ |
| OS | Ubuntu 22.04.5 LTS (jammy) | 22.04 known-good | ✅ |
| Python | 3.10.12 | ≥3.10 | ✅ |
| Power limit | **300 W** (default = max = 300 W, min 250 W) | — | ⚠️ see note |

**Important note — 300W Max-Q variant:** This card is the 300W Max-Q workstation
edition, *not* the 600W full-power variant. The hardware power ceiling is 300W, so
the `nvidia-smi -pl 400` mitigation documented in README.md does not apply (and is
not possible — max settable limit is 300W). The GB202 chip-reset bug is primarily
associated with sustained 600W operation; the 300W variant runs at half that power
envelope, substantially reducing (but not eliminating) the risk. The gpu_burn gate
below validates this empirically.

## Repo validation (no GPU)

- `gh` authenticated as waynehacking8, private repo cloned OK
- `python3 -m venv .venv` + `pip install -e ".[dev]"` — clean install
  (pydantic 2.13.4, numpy 2.2.6, pytest 9.0.3)
- **pytest: 6 passed in 0.07s** — all green ✅

## gpu_burn build

- Cloned wilicc/gpu-burn, built with `make CUDAPATH=/usr/local/cuda COMPUTE=120`
  (default Makefile picks up stale `/usr/bin/nvcc` 11.5 — must override)

## Short test — `./gpu_burn 300` (5 min)

| Metric | Value |
|---|---|
| Result | **PASS — GPU 0: OK, errors: 0** |
| Sustained throughput | ~50,300 GFLOP/s (FP32) |
| VRAM used by test | 76,615 MB of 85,128 MB available |
| Peak temperature | 88 °C |
| Steady-state temperature | 87 °C |
| Power draw | 300 W (pinned at cap) |
| SM clock under load | 1455–1477 MHz |
| ERR! / chip reset / bus drop | None |

## Long test — `./gpu_burn 3600` (curtailed at ~14 min by operator decision)

Started 11:14:13 in tmux session `gpuburn`; operator stopped the test at 11:28
(22.8% progress) after judging the cumulative evidence sufficient.

| Metric | Value |
|---|---|
| Result | **No anomalies — errors: 0 for the entire 13.8 min run** |
| Sustained throughput | ~50,310 GFLOP/s (FP32), flat the whole run |
| Peak temperature | 88 °C (steady-state 87 °C) |
| Power draw | 300 W (pinned at cap, no excursions) |
| SM clock under load | 1455–1477 MHz (stable, no throttle steps) |
| ERR! / chip reset / bus drop | None |

**Total sustained full-load time across both tests: ~19 minutes at 300 W,
zero compute errors, zero driver/bus anomalies.** GPU returned to idle cleanly
(64 °C / 16 W) and released all memory after the test.

## Gate verdict

**PASS** (operator-curtailed)

- No chip reset, no ERR!, no bus drop, no compute errors across ~19 min of
  sustained 300 W full load (5-min short test completed with `GPU 0: OK`;
  long test ran clean for 13.8 min before being stopped early).
- Thermals are healthy: 87–88 °C steady state with no clock instability.
- The 300W Max-Q power envelope (vs. the 600W variant the chip-reset bug is
  primarily associated with) further reduces residual risk.
- **Caveat:** the full 60-min criterion was not run to completion. If Isaac Lab
  training later shows instability, re-run `./gpu_burn 3600` to completion
  inside tmux before blaming the software stack.

## Recommended next steps

1. **Proceed to Isaac Sim 5.1.0 / Isaac Lab 2.3.2 installation** (next session)
   on this driver (580.159.03) — it is in the known-good 580.x family.
2. Before first sim run, apply the documented PhysX workaround:
   `sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so && sudo ldconfig`
   (the `nvidia-smi -pl 400` mitigation is N/A on this 300W-capped card).
3. Continue Phase 0 benchmarks #2–#8 (bug #2133 parallel-reset workaround,
   GPU-utilization knee, L2 latency, warm-start, JIT cache, LLM latency,
   step-count calibration).
4. Account for the ~11.5 GiB of VRAM permanently held by resident services
   (lightrag node process + desktop) when sizing env counts — usable VRAM is
   ~85 GiB, not 98 GiB.
5. Always build CUDA code with `/usr/local/cuda/bin/nvcc` (12.8); the default
   `/usr/bin/nvcc` is CUDA 11.5 and cannot target sm_120.
