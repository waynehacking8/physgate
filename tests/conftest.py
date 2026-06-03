"""Shared pytest configuration.

Isaac-dependent tests are marked ``@pytest.mark.isaac`` and are skipped
automatically when Isaac Sim is not importable (i.e. when running in the
pure-logic .venv instead of ~/env_isaaclab).
"""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "isaac: requires Isaac Sim + Isaac Lab (run inside the env_isaaclab venv)",
    )
