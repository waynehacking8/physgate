"""Tests for the Claude Code headless-mode client (subscription OAuth path).

Subscription OAuth tokens (sk-ant-oat01-...) are rejected by the raw Anthropic
API, so they must be routed through ``claude -p`` instead (DECISIONS.md D-015).
These tests mock the subprocess; no real CLI or network is needed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from physgate.planner.headless_client import (
    ClaudeCodeError,
    ClaudeCodeHeadlessClient,
    ClaudeCodeRateLimitError,
    claude_cli_available,
)
from physgate.planner.planner import (
    ClaudePlanner,
    llm_credentials_available,
    make_anthropic_client,
)

TOKEN = "sk-ant-oat01-test-not-real"


def _envelope(result: str, *, is_error: bool = False, api_error_status: int | None = None) -> str:
    """Build a claude -p --output-format json result envelope."""
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": is_error,
            "api_error_status": api_error_status,
            "result": result,
            "duration_ms": 1234,
            "session_id": "test",
        }
    )


def _fake_run(stdout: str, returncode: int = 0):
    """Return a fake subprocess.run that records calls and returns canned output."""
    calls: list[dict] = []

    def run(cmd, **kwargs):
        calls.append({"cmd": cmd, **kwargs})
        return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)

    return run, calls


# ------------------------------------------------------------------ client core


def test_headless_client_returns_text_response(monkeypatch):
    run, _ = _fake_run(_envelope("hello from claude"))
    monkeypatch.setattr("subprocess.run", run)

    client = ClaudeCodeHeadlessClient(oauth_token=TOKEN)
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=100,
        system="sys",
        messages=[{"role": "user", "content": "hi"}],
    )

    text = "".join(b.text for b in response.content if b.type == "text")
    assert text == "hello from claude"


def test_headless_client_command_has_model_and_system_prompt(monkeypatch):
    run, calls = _fake_run(_envelope("ok"))
    monkeypatch.setattr("subprocess.run", run)

    client = ClaudeCodeHeadlessClient(oauth_token=TOKEN)
    client.messages.create(
        model="claude-opus-4-8",
        max_tokens=100,
        system="you are a planner",
        messages=[{"role": "user", "content": "plan it"}],
    )

    cmd = calls[0]["cmd"]
    assert "-p" in cmd
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "claude-opus-4-8"
    assert "--system-prompt" in cmd
    assert cmd[cmd.index("--system-prompt") + 1] == "you are a planner"
    assert "--output-format" in cmd and "json" in cmd
    # bare mode must NOT be used: it does not read CLAUDE_CODE_OAUTH_TOKEN
    assert "--bare" not in cmd


def test_headless_client_sends_prompt_via_stdin(monkeypatch):
    """Prompts contain large scene/plan JSON — must go via stdin, not argv."""
    run, calls = _fake_run(_envelope("ok"))
    monkeypatch.setattr("subprocess.run", run)

    client = ClaudeCodeHeadlessClient(oauth_token=TOKEN)
    client.messages.create(
        model="m",
        system="s",
        messages=[{"role": "user", "content": "the user prompt"}],
    )

    assert calls[0]["input"] == "the user prompt"
    assert "the user prompt" not in calls[0]["cmd"]


def test_headless_client_env_carries_oauth_token_only(monkeypatch):
    """The subprocess env must carry the token but not leak this process's env."""
    run, calls = _fake_run(_envelope("ok"))
    monkeypatch.setattr("subprocess.run", run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-not-leak")

    client = ClaudeCodeHeadlessClient(oauth_token=TOKEN)
    client.messages.create(model="m", system="s", messages=[{"role": "user", "content": "x"}])

    env = calls[0]["env"]
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == TOKEN
    assert "ANTHROPIC_API_KEY" not in env
    assert "HOME" in env and "PATH" in env


def test_headless_client_raises_rate_limit_error(monkeypatch):
    run, _ = _fake_run(_envelope("rate limited", is_error=True, api_error_status=429))
    monkeypatch.setattr("subprocess.run", run)

    client = ClaudeCodeHeadlessClient(oauth_token=TOKEN)
    with pytest.raises(ClaudeCodeRateLimitError):
        client.messages.create(model="m", system="s", messages=[{"role": "user", "content": "x"}])


def test_headless_client_raises_on_error_result(monkeypatch):
    run, _ = _fake_run(_envelope("Not logged in · Please run /login", is_error=True))
    monkeypatch.setattr("subprocess.run", run)

    client = ClaudeCodeHeadlessClient(oauth_token=TOKEN)
    with pytest.raises(ClaudeCodeError) as excinfo:
        client.messages.create(model="m", system="s", messages=[{"role": "user", "content": "x"}])
    assert "Not logged in" in str(excinfo.value)


def test_headless_client_raises_on_non_json_output(monkeypatch):
    run, _ = _fake_run("segfault garbage", returncode=1)
    monkeypatch.setattr("subprocess.run", run)

    client = ClaudeCodeHeadlessClient(oauth_token=TOKEN)
    with pytest.raises(ClaudeCodeError):
        client.messages.create(model="m", system="s", messages=[{"role": "user", "content": "x"}])


def test_headless_client_requires_a_token(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(ValueError):
        ClaudeCodeHeadlessClient()


# -------------------------------------------------------- credential selection


def test_make_client_returns_headless_for_claude_code_oauth_token(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", TOKEN)

    client = make_anthropic_client()
    assert isinstance(client, ClaudeCodeHeadlessClient)


def test_make_client_returns_headless_for_oat_auth_token(monkeypatch):
    """ANTHROPIC_AUTH_TOKEN holding a subscription (sk-ant-oat) token must be
    routed through claude -p — the raw API rejects those tokens."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", TOKEN)

    client = make_anthropic_client()
    assert isinstance(client, ClaudeCodeHeadlessClient)


def test_make_client_prefers_api_key_over_oauth(monkeypatch):
    """A real API key (pay-per-token, unrestricted) wins over OAuth tokens."""
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test-not-real")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", TOKEN)

    client = make_anthropic_client()
    assert isinstance(client, anthropic.Anthropic)


def test_make_client_non_oat_auth_token_uses_raw_sdk(monkeypatch):
    """A non-subscription bearer token (proxy/gateway) still uses the raw SDK."""
    import anthropic

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "some-gateway-bearer-token")

    client = make_anthropic_client()
    assert isinstance(client, anthropic.Anthropic)


def test_llm_credentials_available_with_claude_code_token(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", TOKEN)
    assert llm_credentials_available() is True


def test_claude_cli_available_is_bool():
    assert isinstance(claude_cli_available(), bool)


# --------------------------------------------------- planner over headless path


def test_claude_planner_works_over_headless_client(monkeypatch):
    """End-to-end (mocked subprocess): ClaudePlanner -> claude -p -> parsed Plans."""
    plan_json = json.dumps(
        [
            {
                "plan_id": "p1",
                "task": "fetch the box",
                "rationale": "direct",
                "steps": [
                    {
                        "step_id": 1,
                        "tool": "move_to_pose",
                        "args": {"target": "box_03"},
                        "preconditions": ["box_03 exists"],
                        "effects": [],
                    }
                ],
            }
        ]
    )
    run, calls = _fake_run(_envelope(plan_json))
    monkeypatch.setattr("subprocess.run", run)

    from physgate.gate.schemas import Scene, SceneObject

    scene = Scene(
        objects=[SceneObject(id="box_03", label="box", affordances=["graspable"], is_anomaly=True)],
        relations=[],
        gripper_empty=True,
    )

    planner = ClaudePlanner(client=ClaudeCodeHeadlessClient(oauth_token=TOKEN))
    plans = planner("fetch the box", scene, 1, None)

    assert len(plans) == 1
    assert plans[0].plan_id == "p1"
    # the planner's system prompt must have been forwarded to claude -p
    cmd = calls[0]["cmd"]
    assert "robot task planner" in cmd[cmd.index("--system-prompt") + 1]
