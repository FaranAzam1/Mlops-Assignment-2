"""The explicit agent loop: Observe -> Reason -> Select tool -> Execute -> Interpret -> Report.

All control flow is plain Python. The LLM chooses *which* tool to call next (and may call
several, including follow-ups prompted by earlier results); this module only validates the
calls, executes them, feeds redacted results back and keeps the findings ledger.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

from auditor.llm import LLMClient
from auditor.prompts import OBSERVATION_INTRO, SYSTEM_PROMPT
from auditor.redact import redact_text
from auditor.report import validate_violation
from auditor.rules import CATEGORIES, SEVERITIES
from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS, RepoTools

MAX_TOOL_OUTPUT_CHARS = 12_000
COMPACT_THRESHOLD_CHARS = 140_000
KEEP_RECENT_MESSAGES = 14
ELIDE_MIN_CHARS = 500

REASONING_PROP = {
    "type": "string",
    "description": "One or two sentences: what the previous result showed and why you are making this call.",
}

BOOKKEEPING_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "record_violation",
            "description": (
                "Record ONE confirmed DevOps violation. The harness verifies the file/line/quote against the real repository "
                "and rejects unsupported or duplicate findings (you will see why and can correct them)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {"type": "string", "description": "Catalogue id, e.g. SEC-001."},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "severity": {"type": "string", "enum": list(SEVERITIES)},
                    "file": {"type": "string", "description": "Exact repo-relative path. Omit (or null) if something is missing or the problem is repo-wide."},
                    "line": {"type": "integer", "description": "1-based line number. Omit (or null) when not applicable."},
                    "description": {"type": "string"},
                    "evidence": {"type": "string", "description": "What you observed (brief quote allowed; never secret values)."},
                    "recommendation": {"type": "string"},
                    "evidence_quote": {"type": "string", "description": "Exact snippet from the file, used by the harness to verify the finding."},
                },
                "required": ["rule_id", "severity", "description", "evidence", "recommendation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retract_violation",
            "description": "Withdraw a previously recorded violation that you now believe is wrong (use the index returned by record_violation).",
            "parameters": {
                "type": "object",
                "properties": {"index": {"type": "integer"}, "why": {"type": "string"}},
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "End the investigation once you are confident every relevant area has been checked.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string", "description": "Brief summary of what was checked."}},
                "required": ["summary"],
            },
        },
    },
]

BOOKKEEPING_TOOLS = {"record_violation", "retract_violation", "finish"}


def _with_reasoning(schema: dict[str, Any]) -> dict[str, Any]:
    fn = json.loads(json.dumps(schema))
    props = fn["function"]["parameters"]["properties"]
    props["reasoning"] = REASONING_PROP
    return fn


ALL_SCHEMAS = [_with_reasoning(s) for s in (*TOOL_SCHEMAS, *BOOKKEEPING_SCHEMAS)]
FINISH_CHOICE = {"type": "function", "function": {"name": "finish"}}


_PRESENCE_PATTERNS: list[tuple[str, str]] = [
    ("README", r"(?i)(^|/)readme(\.[a-z]+)?$"),
    (".gitignore", r"(^|/)\.gitignore$"),
    ("tests", r"(?i)(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]*\.py$|_test\.(py|go)$|\.(test|spec)\.[jt]sx?$"),
    ("CI config", r"^\.github/workflows/[^/]+\.ya?ml$|(^|/)\.gitlab-ci\.yml$|(^|/)Jenkinsfile$|^\.circleci/|(^|/)azure-pipelines\.ya?ml$"),
    ("dependency manifests", r"(?i)(^|/)(requirements[^/]*\.txt|pyproject\.toml|setup\.py|setup\.cfg|Pipfile|package\.json|go\.mod|Cargo\.toml|pom\.xml|build\.gradle(\.kts)?|Gemfile|composer\.json)$"),
    ("lockfiles", r"(?i)(^|/)(poetry\.lock|uv\.lock|Pipfile\.lock|package-lock\.json|yarn\.lock|pnpm-lock\.yaml|Cargo\.lock|go\.sum|Gemfile\.lock|composer\.lock)$"),
    ("lint/format config", r"(?i)(^|/)(\.flake8|ruff\.toml|\.ruff\.toml|tox\.ini|\.pylintrc|\.eslintrc[^/]*|eslint\.config\.[a-z]+|\.prettierrc[^/]*|\.pre-commit-config\.ya?ml|\.editorconfig)$"),
    ("Dockerfile", r"(?i)(^|/)(Dockerfile[^/]*|docker-compose[^/]*\.ya?ml)$"),
    ("env example", r"(?i)(^|/)\.env\.(example|sample|template)$"),
]


def presence_summary(files: list[str]) -> str:
    """Name-based facts about common project files (pyproject.toml/setup.cfg may also hold lint settings: read them)."""
    parts = []
    for label, pattern in _PRESENCE_PATTERNS:
        found = [f for f in files if re.search(pattern, f)]
        shown = ", ".join(found[:4]) + (f" (+{len(found) - 4} more)" if len(found) > 4 else "")
        parts.append(f"{label}: {shown or 'none found'}")
    return "; ".join(parts)


_SKIP_SUFFIXES = (".lock", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf", ".zip", ".gz", ".pyc", ".bin", ".woff", ".woff2", ".ttf", ".mp4", ".mp3")
_SKIP_NAMES = {"package-lock.json", "go.sum", "license", "licence", "copying", "__init__.py"}


def _norm_path(path: str) -> str:
    path = path.replace("\\", "/").strip()
    while path.startswith("./"):
        path = path[2:]
    return path.lstrip("/")


class TraceWriter:
    """Append-only JSONL trace of every observation, tool call and result (all text redacted)."""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path
        self._fh = None
        if path is not None:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                self._fh = open(path, "w", encoding="utf-8")
            except OSError as exc:
                print(f"warning: cannot write trace file {path}: {exc}", file=sys.stderr)

    def event(self, **fields: Any) -> None:
        if self._fh is None:
            return
        clean = {k: (redact_text(v) if isinstance(v, str) else v) for k, v in fields.items()}
        clean["t"] = round(time.time(), 2)
        self._fh.write(json.dumps(clean, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class AuditAgent:
    def __init__(
        self,
        llm: LLMClient,
        tools: RepoTools,
        repo_url: str,
        trace: Optional[TraceWriter] = None,
        max_steps: int = 40,
        min_tool_calls: int = 4,
        deadline_seconds: float = 900.0,
        log: Optional[Callable[[str], None]] = None,
        review_gate: bool = True,
        review_max_files: int = 40,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.repo_url = repo_url
        self.trace = trace or TraceWriter(None)
        self.max_steps = max_steps
        self.min_tool_calls = min_tool_calls
        self.deadline = time.monotonic() + deadline_seconds
        self.log = log or (lambda msg: print(msg, file=sys.stderr))
        self.messages: list[dict[str, Any]] = []
        self._ledger: list[Optional[dict[str, Any]]] = []
        self.inspection_calls = 0
        self.finished = False
        self.summary = ""
        self.stop_reason = ""
        self.token_usage: Counter = Counter()
        self._force_finish = False
        self.review_gate = review_gate
        self.review_max_files = review_max_files
        self._review_done = False
        self._read_paths: set[str] = set()

    # ---------------------------------------------------------------- observe
    @property
    def violations(self) -> list[dict[str, Any]]:
        return [v for v in self._ledger if v is not None]

    def observe(self) -> str:
        files = self.tools.tracked_files()
        top_dirs: Counter = Counter()
        top_files: list[str] = []
        for f in files:
            if "/" in f:
                top_dirs[f.split("/", 1)[0] + "/"] += 1
            else:
                top_files.append(f)
        ext = Counter((Path(f).suffix.lower() or Path(f).name) for f in files)
        workflows = [f for f in files if f.startswith(".github/workflows/")]
        lines = [
            f"Repository: {self.repo_url}",
            f"Tracked files: {len(files)}",
            "Top-level directories: " + (", ".join(f"{d} ({n})" for d, n in sorted(top_dirs.items())) or "(none)"),
            "Top-level files: " + (", ".join(sorted(top_files)[:60]) or "(none)"),
            "Most common file types: " + ", ".join(f"{e} x{n}" for e, n in ext.most_common(10)),
            "GitHub Actions workflow files: " + (", ".join(workflows) if workflows else "(none found under .github/workflows/)"),
            "Common files present (name-based facts, not findings; judge whether anything is missing): " + presence_summary(files),
            "Latest commit: " + self.tools.git_log(max_count=1).replace("\n", " | "),
        ]
        return "\n".join(lines)

    # -------------------------------------------------------------------- run
    def run(self) -> list[dict[str, Any]]:
        observation = self.observe()
        self.trace.event(step=0, phase="observe", observation=observation)
        self.log("[observe] " + observation.splitlines()[1])
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{OBSERVATION_INTRO}\n\n{observation}"},
        ]
        for step in range(1, self.max_steps + 1):
            if time.monotonic() - float(getattr(self.llm, "waited_seconds", 0.0)) > self.deadline:
                self._force_finish, self.stop_reason = True, "wall-clock time limit reached"
            elif step == self.max_steps:
                self._force_finish, self.stop_reason = True, "step limit reached"
            elif step == self.max_steps - 2:
                self.messages.append({
                    "role": "user",
                    "content": "Only two steps remain. Record any confirmed findings not yet recorded, then call finish.",
                })
            self._maybe_compact()
            choice: Any = FINISH_CHOICE if self._force_finish else "required"
            reply = self.llm.complete(self.messages, ALL_SCHEMAS, choice)
            self.token_usage.update(reply.usage)
            self.messages.append(reply.as_message())
            if reply.content:
                self.trace.event(step=step, phase="reason", text=reply.content)
            if not reply.tool_calls:
                self.messages.append({"role": "user", "content": "Respond only by calling a tool. Use finish when done."})
                self.trace.event(step=step, phase="nudge", text="model returned no tool call")
                if self._force_finish:
                    break
                continue
            for call in reply.tool_calls:
                result = self._dispatch(step, call.id, call.name, call.arguments)
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            if self.finished:
                break
        if not self.finished and not self.stop_reason:
            self.stop_reason = "model never called finish"
        self.trace.event(
            phase="end", finished=self.finished, stop_reason=self.stop_reason,
            violations=len(self.violations), inspection_calls=self.inspection_calls, tokens=dict(self.token_usage),
        )
        return self.violations

    # --------------------------------------------------------------- dispatch
    def _dispatch(self, step: int, call_id: str, name: str, raw_args: str) -> str:
        if self.finished:
            return "IGNORED: the investigation has already finished."
        try:
            args = json.loads(raw_args) if raw_args and raw_args.strip() else {}
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            result = f"ERROR: tool arguments were not valid JSON ({exc}). Re-issue the call."
            self.trace.event(step=step, phase="tool_error", tool=name, raw_arguments=raw_args[:300], result=result)
            return result
        reasoning = args.pop("reasoning", "")
        shown = ", ".join(f"{k}={str(v)[:50]!r}" for k, v in args.items())
        self.log(f"[step {step}] {name}({shown})")

        if name in TOOL_FUNCTIONS:
            self.inspection_calls += 1
            if name == "read_file" and isinstance(args.get("path"), str):
                self._read_paths.add(_norm_path(args["path"]))
            output = self.tools.execute(name, args)
            result = self._clip(output)
            phase = "tool_call"
        elif name == "record_violation":
            result, phase = self._record(args), "record"
        elif name == "retract_violation":
            result, phase = self._retract(args), "retract"
        elif name == "finish":
            result, phase = self._finish(args), "finish"
        else:
            result, phase = f"ERROR: unknown tool '{name}'", "tool_error"
        self.trace.event(
            step=step, phase=phase, tool=name, arguments=args, reasoning=reasoning,
            result_chars=len(result), result_preview=result[:400],
        )
        return result

    @staticmethod
    def _clip(text: str) -> str:
        text = redact_text(text)
        if len(text) > MAX_TOOL_OUTPUT_CHARS:
            text = text[:MAX_TOOL_OUTPUT_CHARS] + f"\n...[output truncated at {MAX_TOOL_OUTPUT_CHARS} chars; narrow the request]"
        return text

    def _record(self, args: dict[str, Any]) -> str:
        violation, message = validate_violation(args, self.tools, self.violations)
        if violation is None:
            return message
        self._ledger.append(violation)
        index = len(self._ledger) - 1
        where = violation["file"] or "repository level"
        extra = f" ({message})" if message else ""
        return f"RECORDED #{index}: {violation['rule_id']} {violation['severity']} at {where}{extra}. Total recorded: {len(self.violations)}."

    def _retract(self, args: dict[str, Any]) -> str:
        index = args.get("index")
        if not isinstance(index, int) or not 0 <= index < len(self._ledger) or self._ledger[index] is None:
            return f"ERROR: no active recorded violation with index {index!r}"
        removed, self._ledger[index] = self._ledger[index], None
        return f"RETRACTED #{index} ({removed['rule_id']}). Total recorded: {len(self.violations)}."

    def _finish(self, args: dict[str, Any]) -> str:
        if self.inspection_calls < self.min_tool_calls and not self._force_finish:
            return (
                f"REJECTED: only {self.inspection_calls} inspection call(s) made. Investigate the repository properly "
                f"(workflows, manifests, secrets, tracked files, configuration) before finishing."
            )
        if self.review_gate and not self._review_done and not self._force_finish:
            self._review_done = True
            unread = self._unread_files()
            if unread:
                shown = ", ".join(unread[:15]) + (f" (+{len(unread) - 15} more)" if len(unread) > 15 else "")
                return (
                    "REVIEW BEFORE FINISHING (one-time): these tracked files were never opened: "
                    f"{shown}. Open any that could hide defects (source, configuration, workflows, build files, manifests): "
                    "machine-specific paths, hardcoded secrets, unsafe code, wrong settings. Record anything new, "
                    "then call finish again; the next finish call will end the investigation."
                )
        self.finished = True
        self.summary = str(args.get("summary", ""))[:500]
        return f"Inspection finished with {len(self.violations)} recorded violation(s)."

    def _unread_files(self) -> list[str]:
        """Small repositories only: non-empty text-like tracked files the agent never opened with read_file."""
        files = self.tools.tracked_files()
        if len(files) > self.review_max_files:
            return []
        unread = []
        for rel in files:
            low = rel.lower()
            if rel in self._read_paths or low.endswith(_SKIP_SUFFIXES) or low.rsplit("/", 1)[-1] in _SKIP_NAMES:
                continue
            try:
                if (self.tools.root / rel).stat().st_size == 0:
                    continue
            except OSError:
                continue
            unread.append(rel)
        return unread

    # ------------------------------------------------------------- compaction
    def _maybe_compact(self) -> None:
        total = sum(len(m.get("content") or "") for m in self.messages)
        if total <= COMPACT_THRESHOLD_CHARS:
            return
        changed = 0
        for msg in self.messages[1:-KEEP_RECENT_MESSAGES]:
            content = msg.get("content") or ""
            if msg["role"] == "tool" and len(content) > ELIDE_MIN_CHARS and not content.endswith("[elided]"):
                msg["content"] = content[:300] + "\n...[older tool output elided to save context; re-run the tool if needed] [elided]"
                changed += 1
        if changed:
            ledger = "\n".join(
                f"#{i} {v['rule_id']} {v['severity']} {v['file'] or '-'}:{v['line'] or '-'}"
                for i, v in enumerate(self._ledger) if v
            ) or "(none yet)"
            self.messages.append({
                "role": "user",
                "content": f"Context note: older tool outputs were trimmed. Violations recorded so far:\n{ledger}\nDo not re-record these.",
            })
            self.trace.event(phase="compact", trimmed=changed, chars_before=total)
