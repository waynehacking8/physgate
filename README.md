<h1 align="center">physgate</h1>

<p align="center">
  <b>Your LLM agent plans it. Physics proves the plan is right. Then the robot moves.</b><br>
  An open framework that validates LLM <i>agent orchestration</i> — task
  decomposition, preconditions, failure recovery — in GPU physics simulation,
  with deterministic navigation so feasibility never depends on the LLM
  guessing geometry.
</p>

<p align="center">
  <code>LLM decomposes → critic prunes → physics validates the plan logic → deterministic nav executes</code>
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
when capable of refusing (SafeAgentBench). And LLM-generated plans fail in
characteristic ways: wrong step ordering, violated preconditions, no recovery
after a failed action, and confidently attempting impossible tasks.

physgate is a **dual-system architecture** with a physics-verification gate between
the LLM agent and the robot:

1. **HIGH level — the LLM agent (Claude)** decomposes a natural-language task into
   semantic skill plans (`move_to_pose(<object>)` / `pick` / `place`). It does NOT
   do geometry: no coordinates, no waypoints, no obstacle avoidance.
2. **A safety-critic agent** adversarially prunes plans that violate safety contracts.
3. **The Sim-Gate** validates surviving plans in **parallel NVIDIA Isaac Lab physics
   simulation** — step ordering, preconditions, and physical outcome (does the box
   actually end up on the shelf?).
4. **LOW level — deterministic navigation** (A\* occupancy-grid planner behind a
   Nav2-compatible interface) executes the winning plan. Obstacle avoidance is
   GUARANTEED here, never guessed by the LLM.

The core thesis: **physics simulation is the highest-quality verifier of agent
orchestration** — it catches wrong decomposition, unmet preconditions, and
infeasible requests more reliably than a neural verifier or LLM self-critique.
What physics does NOT need to do is rescue bad route geometry: with navigation in
the right layer, any well-formed plan is executable (measured feasibility ~100%,
see `benchmarks/rebuild/`).

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
 │ Orchestrator (LangGraph state machine)         │  deterministic harness
 │   phase routing · retry budget · approval gate  │
 └──────────────────────────────────────────────┘
        │
 ┌──────────────────────────────────────────────┐
 │ HIGH: Planner (Claude Opus 4.8) → N candidates │  semantic skills only —
 │ Safety Critic (SAFER pattern) → prune unsafe    │  no geometry, no waypoints
 └──────────────────────────────────────────────┘
        │ surviving candidates
 ┌──────────────────────────────────────────────┐
 │ Sim-Gate  (soft interlock — NOT a safety fn)   │  validates ORCHESTRATION:
 │   L1 kinematic limits (URDF)        <1 ms       │  ordering, preconditions,
 │   L3 scene-graph preconditions      <1 ms       │  physical outcome
 │   L2 parallel physics (Isaac Lab, N envs)       │
 │   → select a verified plan                      │
 └──────────────────────────────────────────────┘
        │ verified plan artifact (JSON)  ══ no code crosses this line ══
 ┌──────────────────────────────────────────────┐
 │ LOW: deterministic navigation (A* / Nav2-ready)│  obstacle avoidance is
 │   Executor (Isaac sim backend / walking policy) │  guaranteed here
 │   [planned, NOT implemented: ROS 2 → real Go2]  │
 └──────────────────────────────────────────────┘
        │
 Audit: three-stream records + Merkle checkpoint (in-memory MVP)
```

Full design: [`docs/design/architecture.md`](docs/design/architecture.md) and the
architecture-correction record [`docs/design/REBUILD.md`](docs/design/REBUILD.md).

## Status

🚀 **The pipeline runs end to end, and the architecture was corrected after
adversarial review** (2026-06-03). The full loop — natural-language task → N=8
candidate plans → safety critic → Sim-Gate (L1 kinematic → L3 scene-graph → L2
parallel Isaac Lab physics) → deterministic-navigation execution with a trained
Go2 walking policy — works on the target hardware.

Key correction ([`docs/design/REBUILD.md`](docs/design/REBUILD.md)): an earlier
version pushed obstacle avoidance to the LLM (straight-line low level + a
hand-placed rescue waypoint), which produced a misleading "only 17% of LLM plans
are feasible, best-of-N fixes it" headline. That was an artifact of the layering
defect. With deterministic A\* navigation in the low level, **feasibility of
well-formed plans is ~100%** (mock 6/6, real Claude 8/8 — `benchmarks/rebuild/`),
and the project's evaluation focus is **agent-orchestrator quality**
(`benchmarks/orchestration/`): decomposition, preconditions, recovery,
infeasibility recognition.

See [`STATUS.md`](STATUS.md) for the component matrix, [`DECISIONS.md`](DECISIONS.md)
for build decisions, and [`benchmarks/`](benchmarks/) for results and demo transcripts.

## Quick start

```bash
# Pure-logic pipeline (no GPU, no API key needed)
pip install -e ".[dev]"
pytest                                    # pure-logic test suite (~200 tests)
python examples/fetch_and_place.py        # offline end-to-end demo
python benchmarks/orchestration/run_eval.py --planner mock   # orchestrator eval

# With real LLM planning — either credential works:
ANTHROPIC_API_KEY=sk-ant-api03-... python examples/fetch_and_place.py    # API key
CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-... python examples/fetch_and_place.py
#   ^ Claude subscription token (from `claude setup-token`); routed through
#     Claude Code headless mode automatically — see DECISIONS.md D-015

# With Isaac Sim physics validation + execution (requires the env_isaaclab venv,
# see scripts/install_sim_stack.sh and DECISIONS.md D-005)
source ~/env_isaaclab/bin/activate
pytest tests/test_isaac_sim_gate.py       # Isaac integration tests
python examples/fetch_and_place.py --isaac
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
