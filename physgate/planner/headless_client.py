"""Claude Code headless-mode client for subscription OAuth tokens.

Claude subscription OAuth tokens (``sk-ant-oat01-...``) are **rejected by the
raw Anthropic API** (api.anthropic.com) — requests fail with an opaque
429 ``rate_limit_error`` regardless of actual quota. The officially supported
way to use a subscription programmatically is Claude Code's headless mode
(``claude -p``), authenticated via the ``CLAUDE_CODE_OAUTH_TOKEN`` environment
variable (the token produced by ``claude setup-token``).

This module wraps ``claude -p`` behind the same ``client.messages.create()``
interface the Anthropic SDK exposes, so :class:`ClaudePlanner` and
:class:`ClaudeCritic` work unchanged with either credential type.

Implementation notes (verified against claude CLI on 2026-06-03):

* The prompt is piped via **stdin** (scene/plan JSON exceeds argv limits).
* ``--system-prompt`` replaces Claude Code's own system prompt with the
  planner/critic prompt.
* ``--bare`` must NOT be used: bare mode does not read
  ``CLAUDE_CODE_OAUTH_TOKEN`` and fails with "Not logged in".
* Errors come back as ``is_error: true`` with ``api_error_status`` holding the
  HTTP status (429 = subscription rate limit, retryable).

See DECISIONS.md D-015.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass

#: claude -p calls include CLI startup + a full agent turn; allow generous time.
DEFAULT_TIMEOUT_S = 600.0


class ClaudeCodeError(RuntimeError):
    """``claude -p`` returned an error result."""


class ClaudeCodeRateLimitError(ClaudeCodeError):
    """``claude -p`` hit the subscription rate limit (retryable)."""


@dataclass(frozen=True)
class TextBlock:
    """Anthropic-SDK-compatible text content block."""

    text: str
    type: str = "text"


@dataclass(frozen=True)
class HeadlessResponse:
    """Anthropic-SDK-compatible response: ``.content`` is a list of text blocks."""

    content: tuple[TextBlock, ...]


def claude_cli_available(claude_bin: str = "claude") -> bool:
    """True when the ``claude`` CLI is on PATH."""
    return shutil.which(claude_bin) is not None


class _HeadlessMessages:
    """Mimics ``anthropic.Anthropic().messages``."""

    def __init__(self, client: ClaudeCodeHeadlessClient):
        """Initialize with a reference to the parent headless client."""
        self._client = client

    def create(
        self,
        *,
        model: str,
        messages: list[dict],
        system: str = "",
        max_tokens: int | None = None,  # noqa: ARG002 — accepted for SDK parity, claude -p manages it
        **_ignored,
    ) -> HeadlessResponse:
        """Run a prompt through claude -p and return an SDK-compatible response."""
        prompt = "\n\n".join(
            m["content"]
            for m in messages
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        )
        if not prompt:
            raise ValueError("messages must contain at least one user message with string content")
        text = self._client.run_prompt(model=model, system=system, prompt=prompt)
        return HeadlessResponse(content=(TextBlock(text=text),))


class ClaudeCodeHeadlessClient:
    """Anthropic-client-compatible wrapper around ``claude -p``.

    Usage::

        client = ClaudeCodeHeadlessClient()           # reads CLAUDE_CODE_OAUTH_TOKEN
        client.messages.create(model=..., system=..., messages=[...])
    """

    def __init__(
        self,
        oauth_token: str | None = None,
        claude_bin: str = "claude",
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        """Initialize with an OAuth token and claude CLI configuration."""
        token = (
            oauth_token
            or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        )
        if not token:
            raise ValueError(
                "no OAuth token available: pass oauth_token= or set CLAUDE_CODE_OAUTH_TOKEN "
                "(generate one with `claude setup-token`)"
            )
        self._token = token
        self._claude_bin = claude_bin
        self._timeout_s = timeout_s
        self.messages = _HeadlessMessages(self)

    # ------------------------------------------------------------- internals

    def _build_command(self, model: str, system: str) -> list[str]:
        cmd = [
            self._claude_bin,
            "-p",
            "--model",
            model,
            "--output-format",
            "json",
            "--no-session-persistence",
            "--tools",
            "",
        ]
        if system:
            cmd += ["--system-prompt", system]
        return cmd

    def _build_env(self) -> dict[str, str]:
        # Minimal controlled env: the OAuth token decides auth. Deliberately do
        # NOT inherit this process's env (a parent Claude Code session's vars,
        # ANTHROPIC_API_KEY, etc. would change claude -p's behaviour).
        return {
            "HOME": os.environ.get("HOME", ""),
            "PATH": os.environ.get("PATH", ""),
            "CLAUDE_CODE_OAUTH_TOKEN": self._token,
        }

    def run_prompt(self, *, model: str, system: str, prompt: str) -> str:
        """Run one ``claude -p`` call and return the response text."""
        proc = subprocess.run(  # noqa: S603 — fixed binary, prompt via stdin
            self._build_command(model, system),
            input=prompt,
            env=self._build_env(),
            capture_output=True,
            text=True,
            timeout=self._timeout_s,
        )
        return self._parse_result(proc)

    @staticmethod
    def _parse_result(proc: subprocess.CompletedProcess) -> str:
        try:
            envelope = json.loads(proc.stdout)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ClaudeCodeError(
                f"claude -p produced no JSON envelope (exit {proc.returncode}): "
                f"stdout={proc.stdout[:500]!r} stderr={proc.stderr[:500]!r}"
            ) from exc

        result_text = envelope.get("result") or ""
        if envelope.get("is_error"):
            status = envelope.get("api_error_status")
            message = f"claude -p error (api_error_status={status}): {result_text[:500]}"
            if status == 429 or "rate limit" in result_text.lower():
                raise ClaudeCodeRateLimitError(message)
            raise ClaudeCodeError(message)
        return result_text
