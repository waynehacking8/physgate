"""Shared test doubles for the Anthropic SDK client interface."""

from __future__ import annotations


class FakeContent:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class FakeResponse:
    def __init__(self, text: str):
        self.content = [FakeContent(text)]


class FakeMessages:
    def __init__(self, response_text: str):
        self._response_text = response_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(self._response_text)


class FakeAnthropicClient:
    """Anthropic-SDK-compatible test double that returns a canned response."""

    def __init__(self, response_text: str):
        self.messages = FakeMessages(response_text)
