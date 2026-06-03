# Phase 0 — go/no-go benchmarks

These eight checks must pass on the target RTX Pro 6000 Blackwell workstation
**before** any simulation components are integrated. They double as the data
basis for the project's paper. Run them in order; #1 is a hard hardware gate.

| # | Benchmark | Go / No-Go criterion |
|---|---|---|
| 1 | **`gpu_burn` 60-min stability** (chip-reset) | Reset → power-cap to 400W (`nvidia-smi -pl 400`) and retest; still resets → unit unreliable (RMA) |
| 2 | bug #2133 same-state parallel-reset workaround | Cannot reset N envs to identical state → C2 method must be redesigned |
| 3 | GPU-utilization curve, Go2 envs = 1→1024 | Find the saturation knee → sets the real value of `N` (not an assumed 64) |
| 4 | single-plan physics-validation wall-clock (20-step trajectory) | Establishes true L2 latency (likely 20–100 ms, not "~ms") |
| 5 | warm-start latency, 64 Go2 envs in a manipulation scene | Confirms the 10–30 s assumption or finds it longer |
| 6 | Newton/Blackwell first-run JIT time + kernel-cache persistence | >1 h and not cached → dev iteration blocked |
| 7 | LLM planning latency P50/P95 (8 parallel Opus 4.8 calls) | Confirms the real bottleneck (est. 10–50 s) |
| 8 | L2 step-count calibration (steps needed to reliably detect a collision) | Determines a 5× swing in L2 latency |

## Known-good stack (Blackwell sm_120, June 2026)

- Ubuntu 22.04.5 · driver **580.65.06** (NOT 595.x) · CUDA 12.8
- PyTorch **2.7.0+cu128** · Isaac Sim 5.1.0 · Isaac Lab 2.3.2 · ROS 2 Humble
- Planner LLM: Claude Opus 4.8 via cloud API (keeps inference off the local GPU)

## Required workarounds (apply before first sim run)

```bash
# PhysX GPU-pipeline fallback fix (driver 580.x)
sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
sudo ldconfig

# chip-reset mitigation
nvidia-smi -pl 400
```
