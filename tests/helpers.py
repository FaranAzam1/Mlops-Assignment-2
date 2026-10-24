"""Shared test helpers: throw-away git repositories and a scripted stand-in for the LLM."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Callable, Union

from auditor.llm import LLMReply, ToolCall

GIT_IDENTITY = ["-c", "user.name=Test User", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"]


def git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(["git", *GIT_IDENTITY, *args], cwd=cwd, check=True, capture_output=True, text=True)
    return proc.stdout


def make_repo(root: Path, files: dict[str, Union[str, bytes]], messages: tuple[str, ...] = ("initial commit",)) -> Path:
    """Create a git repo containing `files`, committed once per message (extra commits touch NOTES)."""
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q", "-b", "main")
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", messages[0])
    for i, message in enumerate(messages[1:], start=1):
        (root / "NOTES").write_text(f"note {i}\n", encoding="utf-8")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", message)
    return root


def call(name: str, **arguments: Any) -> tuple[str, dict[str, Any]]:
    arguments.setdefault("reasoning", "test")
    return name, arguments


Step = Union[list[tuple[str, dict[str, Any]]], Callable[[list[dict[str, Any]]], list[tuple[str, dict[str, Any]]]]]


class ScriptedLLM:
    """Plays back a fixed sequence of tool-call turns and records what the agent sent."""

    def __init__(self, steps: list[Step]) -> None:
        self.steps = list(steps)
        self.requests: list[dict[str, Any]] = []
        self._counter = 0

    def complete(self, messages, tools, tool_choice) -> LLMReply:
        self.requests.append({"messages": json.loads(json.dumps(messages)), "tool_choice": tool_choice, "tools": tools})
        if not self.steps:
            return LLMReply(content="done", tool_calls=[])
        step = self.steps.pop(0)
        calls = step(messages) if callable(step) else step
        if isinstance(tool_choice, dict):  # a forced call must be honoured, as the real API does
            calls = [(tool_choice["function"]["name"], {"summary": "forced", "reasoning": "forced"})]
        tool_calls = []
        for name, args in calls:
            self._counter += 1
            raw = args if isinstance(args, str) else json.dumps(args)
            tool_calls.append(ToolCall(id=f"call_{self._counter}", name=name, arguments=raw))
        return LLMReply(content=None, tool_calls=tool_calls, usage={"prompt_tokens": 10, "completion_tokens": 5})

    @staticmethod
    def tool_results(messages: list[dict[str, Any]]) -> list[str]:
        return [m["content"] for m in messages if m["role"] == "tool"]
