<h1 align="center">physgate</h1>

<p align="center">
  <b>Your LLM plans it. Physics proves it. Then the robot moves.</b><br>
  An open framework that validates LLM-generated robot plans in GPU physics
  simulation — generating <i>N</i> candidate plans, checking all of them, and
  executing only the most feasible one.
</p>

<p align="center">
  <code>LLM plan → best-of-N physics validation → execute</code>
</p>

---

> [!WARNING]
> **Research software — NOT a certified safety system.** physgate is a research
> prototype. It does **not** implement or replace any safety function under
> ISO 10218-1:2025, ISO 13849-1, IEC 61508, or ISO 3691-4, and it runs on a
> general-purpose Linux + GPU stack that cannot meet PL/SIL requirements. A
> simulation PASS is **not** a safety proof. See [Safety & Scope](#safety--scope)
> before any deployment.

## What this is

LLM agents that drive robots execute hazardous instructions ~95% of the time even
when capable of refusing (SafeAgentBench). And LLM-generated plans are frequently
physically infeasible — collisions, unreachable poses, violated preconditions.

physgate puts a **physics-verification gate** between the LLM planner and the robot:

1. An LLM (Claude) decomposes a natural-language task into **N candidate plans**.
2. A safety-critic agent adversarially prunes plans that violate safety contracts.
3. Surviving plans are validated in **parallel NVIDIA Isaac Lab physics simulation**
   — kinematic limits, collisions, and scene-graph preconditions.
4. The most feasible plan is selected and executed over ROS 2; the rest are discarded.

The core thesis: **physics simulation is the highest-quality verifier** for robot
plans — more accurate than a neural verifier or LLM self-critique — and best-of-N
selection against it measurably improves plan feasibility.

## What this is NOT

- **Not a certified safety component.** It is an application-layer *soft interlock*,
  not a safety-rated function. See [Safety & Scope](#safety--scope).
- **Not a real-time controller.** The planner operates at seconds-per-decision; it
  never enters the motor control loop.
- **Not a claim that simulation guarantees real-world safety.** The sim-to-real gap
  is real; a sim PASS reduces risk, it does not prove safety.

## Architecture

```
 Natural-language task
        │
 ┌──────────────────────────────────────────────┐
 │ Orchestrator (LangGraph + Postgres state)      │  deterministic harness
 │   phase routing · retry budget · approval gate  │
 └──────────────────────────────────────────────┘
        │
 ┌──────────────────────────────────────────────┐
 │ Planner (Claude Opus 4.8) → N candidate plans  │
 │ Safety Critic (SAFER pattern) → prune unsafe    │
 └──────────────────────────────────────────────┘
        │ surviving candidates
 ┌──────────────────────────────────────────────┐
 │ Sim-Gate  (soft interlock — NOT a safety fn)   │
 │   L1 kinematic limits (URDF)        <1 ms       │
 │   L3 scene-graph preconditions      <1 ms       │
 │   L2 parallel physics (Isaac Lab, N envs)       │
 │   → physics score → select best plan            │
 └──────────────────────────────────────────────┘
        │ verified plan artifact (JSON)  ══ no code crosses this line ══
 ┌──────────────────────────────────────────────┐
 │ Executor → ROS 2 (sim backend / real Go2)      │
 │   research-grade RL fallback on divergence      │
 └──────────────────────────────────────────────┘
        │
 Audit: Langfuse (OTEL) + MCAP + Merkle checkpoint
```

Full design: [`docs/design/architecture.md`](docs/design/architecture.md).

## Status

🚧 **Early development.** Currently in **Phase 0** — de-risking benchmarks on the
target hardware before integration (see [`benchmarks/phase0/`](benchmarks/phase0/)).
The MVP targets a single Unitree Go2 quadruped, three MCP tools, and one task type
(fetch-and-place).

## Quick start

> Requires the verified Blackwell stack (see [Hardware](#hardware--requirements)).
> Phase 0 must pass before the simulation components are wired up.

```bash
pip install -e .            # pure-logic components (no GPU needed)
pytest tests/               # run the test suite
```

## Hardware & requirements

Verified known-good stack on NVIDIA RTX Pro 6000 Blackwell (sm_120), June 2026:

| Layer | Version |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| NVIDIA driver | 580.65.06 (**not** 595.x) |
| CUDA / PyTorch | 12.8 / 2.7.0+cu128 |
| Isaac Sim / Isaac Lab | 5.1.0 / 2.3.2 |
| ROS 2 | Humble |
| Planner LLM | Claude Opus 4.8 (cloud API — not local) |

> [!NOTE]
> The RTX Pro 6000 has a known sustained-compute chip-reset issue (drivers 570–595,
> unresolved by NVIDIA as of mid-2026). Phase 0 benchmark #1 stress-tests for this;
> mitigations are power-capping (`nvidia-smi -pl 400`) and keeping LLM inference on a
> cloud API rather than the local GPU.

## Safety & Scope

physgate is an **application-layer (Layer 4) soft interlock**. It sits *above* — and
does not replace — the hardware safety layers that a certified deployment requires:

```
Layer 4  physgate (LLM planner + Sim-Gate + RL fallback)   ← soft interlock, NOT certified
Layer 3  Motion control / ROS 2 / unitree_sdk2             ← not safety-rated
Layer 2  Safety PLC / FSoE (PLd/SIL2)                      ← does NOT exist on Go2
Layer 1  Hardware E-stop / SRSF                            ← does NOT exist on Go2
Layer 0  Mechanical limits / motor current cutoff (firmware)
```

- **Not a safety function** under ISO 10218-1:2025 / ISO 13849-1 / IEC 61508 / ISO 3691-4.
- **The RL fallback is not formally verified** — no barrier certificate; Black-Box
  Simplex (arXiv:2102.12981) / ASTM F3269-21 guarantees do not hold. It is a software
  watchdog with an RL fallback.
- **The Unitree Go2 is a research platform** with no certified hardware safety layer
  (FCC/CE radio compliance only).
- **EU Machinery Regulation 2023/1230** (mandatory 2027-01): AI safety components with
  self-evolving behavior require Notified Body assessment. physgate does not satisfy
  this and must not be represented as safety validation in EU deployments.

**What physgate CAN provide:** a meaningful reduction in the probability of
kinematically infeasible, contextually inappropriate, or unreviewed commands reaching
the robot, compared to unmediated LLM-to-robot execution — an engineering safeguard
for research settings, not a certified safety function.

PROVIDED "AS IS" WITHOUT WARRANTY. Do not deploy near persons without hardware safety
infrastructure independent of this software.

## License

[MIT](LICENSE) © 2026 Wei Cheng (Wayne) Chiu
