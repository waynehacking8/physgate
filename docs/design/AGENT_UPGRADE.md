# physgate — Agent Architecture Upgrade

**Status: AUTHORITATIVE design for the next build pass.**
**Date: 2026-06-04. Source: code review identified that the current repo has agent
*skeleton* (LangGraph + tool-use + replan loop) but demo scenes are too simple to
exercise real agent decision-making. This upgrade makes it a genuine agent orchestrator.**

---

## Problem Statement

The current physgate has the right bones but doesn't prove the agent is smart:
- Only 1 task type (fetch & place), 1 scene, 3 tools
- No complex tool selection — agent never chooses WHICH tool to use
- No dynamic tool discovery — tools are hardcoded in ToolName enum
- No multi-agent coordination
- Replanning is blind regeneration, not failure-aware targeted repair
- 100% orchestration eval pass rate = the tasks are too easy

## Design Principles

1. **HIGH level = agent decisions** — the LLM chooses which tools, which order, handles failures intelligently
2. **LOW level = deterministic execution** — A* navigation, physics-based manipulation, no LLM in the loop
3. **Every new capability must have a scenario where it FAILS** — if the agent always succeeds, the evaluation is worthless

---

## 1. Expanded Tool Set (3 → 10)

### Current tools (keep):
- `query_scene` — observe the world
- `move_to_pose` — navigate to an object (deterministic A*)
- `execute_skill(pick/place)` — manipulation

### New tools:
- `open_door(door_id)` — precondition: robot near door, door is closed+unlocked
- `unlock_door(door_id, key_id)` — precondition: robot holding key_id, near door
- `press_button(button_id)` — precondition: robot near button; effect depends on button (opens gate, calls elevator, activates conveyor)
- `call_elevator(elevator_id, target_floor)` — precondition: robot near elevator panel, elevator doors closed
- `push_object(object_id, direction)` — move a light obstacle out of the way; precondition: object is pushable, robot near it
- `inspect_object(object_id)` — returns weight, graspability, state (locked/unlocked/broken); needed before the agent can decide HOW to interact
- `request_assistance(message)` — agent recognizes it can't do the task alone and escalates (multi-agent placeholder; in v1 this is an explicit "I need help" signal that ends the task with partial_success + handoff info)

### Implementation boundary:
- All new tools follow the same pattern: thin adapter in `mcp_server/tools/core.py`, delegates to `WorldBackend`
- `MockWorldBackend` gains these methods with deterministic symbolic logic
- `SimBackend` gets Isaac Sim implementations where physics matters (push_object, open_door)
- `ToolName` enum expands; `PlanStep` schema unchanged (tool + args + preconditions + effects)
- **Dynamic tool discovery**: `query_scene` response now includes an `available_tools` field listing which tools are usable given the current scene. The planner system prompt tells the LLM to check available_tools before planning. Tools not in the scene (no elevator → no call_elevator) are absent. This is NOT runtime MCP tool registration — it's scene-conditional tool availability reported through the existing query_scene channel.

## 2. Multi-Step Task Suite (1 → 6 task types)

### T1: Fetch & Place (existing, keep as baseline)
"Pick up box_03 and place it on shelf_A."

### T2: Locked Room Delivery
"Deliver box_03 to shelf_B. shelf_B is behind a locked door. The key is on table_01."
- Requires: inspect_object → query_scene → move_to_pose → execute_skill(pick key) → move_to_pose → unlock_door → open_door → move_to_pose → execute_skill(place box)
- Agent must figure out the key→unlock→open→deliver chain
- Failure mode: agent tries to open door without key → replan with failure context

### T3: Blocked Path Clearance
"Deliver box_03 to shelf_A. The direct path is blocked by a pushable crate."
- Requires: inspect_object(crate) → push_object(crate, aside) → proceed with delivery
- OR: agent finds alternate route if available (A* handles this IF no pushable shortcut)
- Failure mode: agent tries to walk through crate → physics rejects → replan

### T4: Multi-Object Sequential Delivery
"Deliver box_03 to shelf_A, then box_04 to shelf_B. shelf_B must be cleared first (box_05 is on it)."
- Requires: ordering reasoning — clear shelf_B first, then deliver box_03, then box_04
- Failure mode: wrong ordering (deliver box_04 to occupied shelf → precondition violation)

### T5: Elevator Floor Transfer
"Move box_03 from floor 1 to shelf_A on floor 2."
- Requires: pick box → move to elevator → call_elevator → ride → move to shelf → place
- Failure mode: agent forgets to call elevator, tries to path-plan to unreachable floor

### T6: Infeasible Task Recognition
"Deliver the anvil to shelf_A." (anvil is 200kg, robot max carry = 5kg)
"Deliver box_03 to room_C." (room_C has no door, fully walled off)
- Agent MUST recognize infeasibility via inspect_object (too heavy) or query_scene (no path)
- Success = agent calls request_assistance or reports infeasible; Failure = agent tries forever

## 3. Analytical Replanning (blind → targeted)

### Current (v1): blind regeneration
On failure, the entire feedback string goes back to the planner:
`"execution failed after validation: box_03 not on shelf_A"`
Planner generates N completely new plans from scratch.

### Upgraded (v2): structured failure diagnosis + targeted repair

New module: `physgate/planner/failure_analyst.py`

```
FailureReport:
  failed_step: PlanStep          # which step failed
  failure_type: enum             # PRECONDITION_UNMET | TOOL_ERROR | PHYSICS_REJECT | TIMEOUT
  root_cause: str                # LLM-generated diagnosis: "door was locked, plan didn't include unlock"
  affected_steps: list[int]      # which downstream steps are invalidated
  suggested_fix: str             # "insert unlock_door before open_door at step 3"
```

The failure analyst is a SECOND LLM call (lightweight, ~500 tokens) that:
1. Receives the failed plan + the specific step failure
2. Diagnoses the root cause
3. Suggests a minimal fix (insert/replace/reorder steps)

The planner then receives the FailureReport instead of a raw string, and is prompted to:
- Keep the working prefix of the plan
- Fix only the broken part
- Explain what changed and why

This is the difference between "start over" and "debug and patch."

### Replan budget splits:
- Attempt 1: targeted repair (keep working prefix, fix the failure)
- Attempt 2: full regeneration (if targeted repair also fails)
- Attempt 3: escalate

## 4. Multi-Agent Coordination (v1: cooperative handoff)

Full multi-agent (multiple LLM instances negotiating) is out of scope for this pass.
What IS in scope: **cooperative task handoff**.

### Design:
- `request_assistance(message)` tool allows the agent to recognize it cannot complete a subtask alone
- The orchestrator interprets this as a PARTIAL_SUCCESS outcome with a handoff payload
- A SECOND orchestrator instance can be spawned with the handoff context to continue
- Evaluation metric: does the agent correctly identify WHEN to hand off vs when to retry?

### Implementation:
- New orchestrator state: `handoff_request: str | None`
- New outcome: `"partial_success"` (distinguished from "escalated" which means budget exhaustion)
- New eval scenario: T6 infeasible tasks where the correct action is request_assistance, not infinite retry
- Multi-agent demo: `examples/cooperative_delivery.py` — agent_1 clears path, hands off to agent_2 for delivery

## 5. Evaluation Upgrade

### New orchestration scenarios (target: 15-20, up from current ~10):
- T1-T6 × feasible/infeasible variants
- Each task type has at least 2 scenarios where the agent SHOULD fail on first attempt and must replan
- Target eval pass rate: 60-80% (not 100% — tasks are genuinely hard)

### New metrics:
- **Tool selection accuracy**: did the agent choose the right tool? (e.g., unlock before open)
- **Replan efficiency**: targeted repair success rate vs blind regeneration success rate
- **Failure recognition**: precision/recall on infeasible task detection
- **Plan prefix preservation**: how much of the working plan is kept during targeted repair (higher = better)
- **Handoff accuracy**: does request_assistance fire at the right time?

### Ablation:
- A0: no gate (baseline)
- A1: + critic
- A2: + symbolic gate
- A3: + navigation-aware gate
- A4: + physics gate
- **A5 (NEW): + failure analyst** (targeted repair vs blind regeneration)

## 6. Implementation Plan

### Phase 1: Tools & Backend (~60% of work)
- Expand ToolName enum, PlanStep unchanged
- Add 7 new tool functions in core.py
- Expand MockWorldBackend with door/elevator/button/push state
- Expand Scene/SceneObject with new properties (locked, pushable, weight, floor)
- Update scene_graph.py for new relation types
- Add available_tools to query_scene response
- Tests for every new tool (precondition checks, state transitions)

### Phase 2: Task Suite & Scenes
- 6 task type scene layouts in layout.py (procedural generation where possible)
- Multi-floor layout support (floor field on objects)
- Locked door + key + button state logic
- Tests for each task scenario

### Phase 3: Analytical Replanning
- failure_analyst.py module
- FailureReport schema
- Planner prompt upgrade for targeted repair
- Integration into orchestrator graph (new node between execute failure and replan)
- Tests: targeted repair vs blind regeneration on known failures

### Phase 4: Multi-Agent Handoff
- request_assistance tool
- partial_success outcome in orchestrator
- cooperative_delivery.py demo
- Eval scenarios for handoff accuracy

### Phase 5: Evaluation & Documentation
- Expand eval scenarios to 15-20
- Run full eval suite, expect 60-80% pass rate
- A5 ablation (failure analyst)
- Update README, architecture.md, EVALUATION_METHODOLOGY.md
- Update all results tables

## 7. Boundaries

- **LOW level stays deterministic.** A* path planning, physics sim, manipulation — NO LLM.
- **HIGH level is where intelligence lives.** Tool selection, ordering, failure diagnosis, handoff decisions.
- **Scene-conditional tools, not dynamic MCP registration.** available_tools in query_scene response, not runtime tool schema changes.
- **Cooperative handoff, not negotiation.** Agent says "I need help with X" and a second agent picks up. No bidding, no conflict resolution.
- **Targeted repair is a prompt technique, not a code rewrite.** The failure analyst is an LLM call that produces structured FailureReport; the planner receives it as context. The orchestrator graph gains one new node, not a rewrite.
