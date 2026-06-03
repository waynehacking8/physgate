# Contributing to physgate

## Development setup

```bash
# 1. Clone and create a pure-logic venv (no GPU)
git clone https://github.com/waynehacking8/physgate.git
cd physgate
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Run the test suite
pytest                        # 253+ pure-logic tests
ruff check physgate/ benchmarks/

# 3. Run the offline demo
python examples/fetch_and_place.py
```

## Isaac Sim integration (GPU required)

Isaac Sim tests and physics validation require a separate venv:

```bash
source ~/env_isaaclab/bin/activate   # Python 3.11, Isaac Lab 2.3.2
pytest tests/test_isaac_sim_gate.py  # ~19 tests, needs GPU
```

See `scripts/install_sim_stack.sh` and `DECISIONS.md` D-005 through D-007
for the full Isaac Lab installation procedure.

## Test conventions

- Pure-logic tests go in `tests/` and must NOT import Isaac Sim
- Isaac-dependent tests go in `tests/test_isaac_sim_gate.py`
- Use `pytest.mark.parametrize` for data-driven tests
- Target 80%+ coverage on new code

## Code style

- `ruff check` must pass (zero errors)
- Type annotations on all public function signatures
- Immutable data: use `frozen=True` on dataclasses and Pydantic models
- No bare `except Exception` — use specific exception types

## Commit conventions

```
<type>: <description>

Types: feat, fix, refactor, docs, test, chore, perf
```
