"""Inspection tools exposed to the LLM.

All tools are strictly read-only with respect to the inspected repository and never
execute code from it. Every text result is redacted (see auditor.redact) so secret
values never leave this module. Tools return plain text; failures are returned as
`ERROR: ...` / `UNAVAILABLE: ...` strings so the agent can reason about them instead of
crashing, and a tool never fabricates a result for a check it could not perform.
"""

from __future__ import annotations

import fnmatch
import inspect
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

from auditor.redact import redact_line, redact_lines, scan_lines

MAX_FILE_BYTES = 1_000_000
MAX_READ_LINES = 300
MAX_READ_CHARS = 12_000
MAX_LINE_CHARS = 300
FLAT_LISTING_LIMIT = 250
SEARCH_BUDGET_SECONDS = 10.0

# Filenames that commonly hold credentials; flagged by name (contents are never shown).
_SENSITIVE_NAME = re.compile(
    r"(?i)(^|/)(\.env(\.(?!example|sample|template|dist)[^/]*)?|id_(rsa|dsa|ecdsa|ed25519)|"
    r"[^/]*\.(pem|key|p12|pfx|jks|keystore)|\.pypirc|\.netrc|credentials(\.json)?|"
    r"terraform\.tfstate(\.backup)?|[^/]*\.tfvars|secrets?\.(ya?ml|json|toml))$"
)


def _truncate(text: str, limit: int = MAX_LINE_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + " …[line truncated]"


class RepoTools:
    """Read-only inspection primitives bound to one checked-out repository."""

    def __init__(self, root: Path, slug: Optional[str] = None, github_token: Optional[str] = None) -> None:
        self.root = Path(root).resolve()
        self.slug = slug
        self.github_token = github_token
        self._files: Optional[list[str]] = None

    # ------------------------------------------------------------------ helpers
    def _git(self, *args: str, timeout: int = 30) -> subprocess.CompletedProcess:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"}
        return subprocess.run(
            ["git", "-c", "core.fsmonitor=false", *args],
            cwd=self.root, capture_output=True, text=True, timeout=timeout,
            env=env, encoding="utf-8", errors="replace",
        )

    def tracked_files(self) -> list[str]:
        if self._files is None:
            files: list[str] = []
            try:
                proc = subprocess.run(
                    ["git", "-c", "core.quotepath=false", "ls-files", "-z"],
                    cwd=self.root, capture_output=True, timeout=30,
                    env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
                )
                if proc.returncode == 0:
                    files = [p for p in proc.stdout.decode("utf-8", "replace").split("\0") if p]
            except (OSError, subprocess.SubprocessError):
                files = []
            if not files:  # not a git checkout: fall back to a directory walk
                for dirpath, dirnames, filenames in os.walk(self.root):
                    dirnames[:] = [d for d in dirnames if d != ".git"]
                    for name in filenames:
                        full = Path(dirpath) / name
                        files.append(full.relative_to(self.root).as_posix())
            self._files = sorted(set(files))
        return self._files

    def _resolve(self, rel: str) -> Path:
        rel = (rel or "").replace("\\", "/").strip()
        while rel.startswith("./"):
            rel = rel[2:]
        rel = rel.lstrip("/")
        if not rel:
            raise ValueError("path is empty")
        parts = rel.split("/")
        if ".." in parts:
            raise ValueError("path escapes the repository")
        if parts[0] == ".git":
            raise ValueError("the .git directory is not readable")
        target = (self.root / rel).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError("path escapes the repository")
        return target

    def _read_text(self, rel: str) -> tuple[Optional[list[str]], Optional[str]]:
        """Return (lines, error). Lines are raw (unredacted); callers redact."""
        try:
            path = self._resolve(rel)
        except ValueError as exc:
            return None, f"ERROR: {exc}"
        if not path.exists():
            return None, f"ERROR: file not found: {rel}"
        if not path.is_file():
            return None, f"ERROR: not a regular file: {rel}"
        try:
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                return None, f"ERROR: file too large to read ({size} bytes > {MAX_FILE_BYTES})"
            data = path.read_bytes()
        except OSError as exc:
            return None, f"ERROR: cannot read {rel}: {exc.strerror or exc}"
        if b"\0" in data[:8000]:
            return None, f"ERROR: {rel} looks like a binary file ({size} bytes)"
        return data.decode("utf-8", "replace").splitlines(), None

    def file_text_redacted(self, rel: str) -> Optional[str]:
        lines, err = self._read_text(rel)
        return None if err else "\n".join(redact_lines(lines))

    def line_count(self, rel: str) -> Optional[int]:
        lines, err = self._read_text(rel)
        return None if err else len(lines)

    def file_exists(self, rel: str) -> bool:
        try:
            return self._resolve(rel).is_file()
        except ValueError:
            return False

    # ------------------------------------------------------------------- tools
    def list_files(self, directory: str = "", pattern: str = "", recursive: bool = False) -> str:
        """Discover repository files. Small repos list flat; large ones show one level."""
        files = self.tracked_files()
        directory = (directory or "").replace("\\", "/").strip("/")
        if ".." in directory.split("/"):
            return "ERROR: directory escapes the repository"
        prefix = f"{directory}/" if directory else ""
        scoped = [f for f in files if f.startswith(prefix)]
        if not scoped:
            return f"No tracked files under '{directory or '.'}'. ({len(files)} tracked files in total)"

        def size_of(rel: str) -> str:
            try:
                return f"{(self.root / rel).stat().st_size}B"
            except OSError:
                return "?"

        if pattern:
            matches = [f for f in scoped if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(f.rsplit("/", 1)[-1], pattern)]
            shown = matches[:FLAT_LISTING_LIMIT]
            lines = [f"{len(matches)} tracked file(s) in '{directory or '.'}' matching '{pattern}'"]
            lines += [f"{f} ({size_of(f)})" for f in shown]
            if len(matches) > len(shown):
                lines.append(f"... {len(matches) - len(shown)} more; narrow the pattern or directory")
            return "\n".join(lines)

        if recursive or len(scoped) <= FLAT_LISTING_LIMIT:
            shown = scoped[:FLAT_LISTING_LIMIT]
            lines = [f"{len(scoped)} tracked file(s) under '{directory or '.'}' ({len(files)} in repository)"]
            lines += [f"{f} ({size_of(f)})" for f in shown]
            if len(scoped) > len(shown):
                lines.append(f"... {len(scoped) - len(shown)} more; use a sub-directory or pattern")
            return "\n".join(lines)

        children: dict[str, int] = Counter()
        direct_files: list[str] = []
        for f in scoped:
            rest = f[len(prefix):]
            if "/" in rest:
                children[rest.split("/", 1)[0] + "/"] += 1
            else:
                direct_files.append(rest)
        lines = [f"'{directory or '.'}': {len(scoped)} tracked files (too many to list flat; showing one level)"]
        lines += [f"{name} ({count} files)" for name, count in sorted(children.items())]
        lines += [f"{prefix}{name} ({size_of(prefix + name)})" for name in direct_files[:FLAT_LISTING_LIMIT]]
        return "\n".join(lines)

    def read_file(self, path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a text file with line numbers (secrets redacted)."""
        lines, err = self._read_text(path)
        if err:
            return err
        total = len(lines)
        try:
            start = max(1, int(start_line or 1))
            end = int(end_line or 0) or min(total, start + MAX_READ_LINES - 1)
        except (TypeError, ValueError):
            return "ERROR: start_line/end_line must be integers"
        end = min(end, total, start + MAX_READ_LINES - 1)
        if total == 0:
            return f"{path}: file is empty (0 lines)"
        if start > total:
            return f"ERROR: start_line {start} is beyond end of file ({total} lines)"
        redacted = redact_lines(lines)
        out, chars = [], 0
        for number in range(start, end + 1):
            text = _truncate(redacted[number - 1])
            chars += len(text)
            if chars > MAX_READ_CHARS:
                end = number - 1
                break
            out.append(f"{number:>5}: {text}")
        header = f"{path} (lines {start}-{end} of {total})"
        footer = f"\n... continue with start_line={end + 1}" if end < total else ""
        return header + "\n" + "\n".join(out) + footer

    def search_code(self, pattern: str, glob: str = "", ignore_case: bool = True, max_results: int = 40) -> str:
        """Regex search across tracked text files (matched lines are redacted)."""
        if not pattern:
            return "ERROR: pattern is required"
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            return f"ERROR: invalid regular expression: {exc}"
        try:
            max_results = max(1, min(int(max_results), 100))
        except (TypeError, ValueError):
            max_results = 40
        deadline = time.monotonic() + SEARCH_BUDGET_SECONDS
        hits: list[str] = []
        files_with_hits: set[str] = set()
        total_hits = 0
        timed_out = False
        for rel in self.tracked_files():
            if glob and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(rel.rsplit("/", 1)[-1], glob)):
                continue
            if time.monotonic() > deadline:
                timed_out = True
                break
            lines, err = self._read_text(rel)
            if err:
                continue
            for number, line in enumerate(lines, start=1):
                if regex.search(line):
                    total_hits += 1
                    files_with_hits.add(rel)
                    if len(hits) < max_results:
                        hits.append(f"{rel}:{number}: {_truncate(redact_line(line.strip()), 200)}")
        header = f"{total_hits} match(es) in {len(files_with_hits)} file(s) for /{pattern}/"
        if not total_hits:
            return header + (" (search time budget exhausted)" if timed_out else "")
        extra = f"\n... {total_hits - len(hits)} more match(es) not shown" if total_hits > len(hits) else ""
        note = "\n(search time budget exhausted; results are partial)" if timed_out else ""
        return header + "\n" + "\n".join(hits) + extra + note

    def git_log(self, max_count: int = 25, path: str = "", message_grep: str = "", pickaxe: str = "") -> str:
        """Inspect commit history (optionally for a path, message text or added/removed string)."""
        try:
            max_count = max(1, min(int(max_count), 100))
        except (TypeError, ValueError):
            max_count = 25
        shallow = self._git("rev-parse", "--is-shallow-repository")
        is_shallow = shallow.returncode == 0 and shallow.stdout.strip() == "true"
        count = self._git("rev-list", "--count", "HEAD")
        if count.returncode != 0:
            return "UNAVAILABLE: no git history in this checkout"
        args = ["log", f"-n{max_count}", "--date=short", "--pretty=format:%h\t%ad\t%an\t%s"]
        if message_grep:
            args.append(f"--grep={message_grep}")
        if pickaxe:
            args.append(f"-S{pickaxe}")
        if path:
            try:
                self._resolve(path)
            except ValueError as exc:
                return f"ERROR: {exc}"
            args += ["--", path.replace("\\", "/")]
        proc = self._git(*args)
        if proc.returncode != 0:
            return f"ERROR: git log failed: {proc.stderr.strip()[:200]}"
        rows = [ln.split("\t", 3) for ln in proc.stdout.splitlines() if ln.strip()]
        head = (
            f"{count.stdout.strip()} commit(s) reachable"
            + (" (shallow clone: older history is not available)" if is_shallow else "")
        )
        if not rows:
            return head + "\nNo commits matched."
        lines = [head, "hash\tdate\tauthor\tsubject"]
        for row in rows:
            row += [""] * (4 - len(row))
            lines.append("\t".join([row[0], row[1], row[2], _truncate(redact_line(row[3]), 140)]))
        return "\n".join(lines)

    def run_ruff(self, path: str = "", select: str = "E9,F63,F7,F82") -> str:
        """Static lint for Python via ruff. Never reads repo config, never writes a cache, never imports code."""
        if not any(f.endswith(".py") for f in self.tracked_files()):
            return "UNAVAILABLE: repository contains no Python files"
        if not re.fullmatch(r"[A-Z0-9,]+", select or ""):
            return "ERROR: select must be a comma-separated list of ruff rule codes, e.g. E9,F63,F7,F82"
        target = "."
        if path:
            try:
                self._resolve(path)
            except ValueError as exc:
                return f"ERROR: {exc}"
            target = path.replace("\\", "/")
        cmd = [
            sys.executable, "-m", "ruff", "check", "--isolated", "--no-cache", "--exit-zero",
            "--output-format", "json", "--select", select, target,
        ]
        try:
            proc = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=90, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return "ERROR: ruff timed out"
        except OSError as exc:
            return f"UNAVAILABLE: could not start ruff: {exc}"
        if "No module named ruff" in (proc.stderr or ""):
            return "UNAVAILABLE: ruff is not installed in the agent's environment"
        try:
            findings = json.loads(proc.stdout or "[]")
        except json.JSONDecodeError:
            return f"ERROR: ruff produced unreadable output: {(proc.stderr or '')[:200]}"
        if not findings:
            return f"ruff (--isolated, select={select}): no findings"
        by_code = Counter(f.get("code") or "syntax-error" for f in findings)
        lines = [f"ruff (--isolated, select={select}): {len(findings)} finding(s): " + ", ".join(f"{c}x{n}" for c, n in by_code.most_common())]
        for f in findings[:30]:
            file = Path(f["filename"]).resolve()
            try:
                rel = file.relative_to(self.root).as_posix()
            except ValueError:
                rel = str(file)
            loc = f.get("location") or {}
            lines.append(f"{rel}:{loc.get('row', '?')}: {f.get('code') or 'syntax'} {_truncate(f.get('message', ''), 160)}")
        if len(findings) > 30:
            lines.append(f"... {len(findings) - 30} more")
        return "\n".join(lines)

    def _http_get_json(self, url: str) -> Any:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "devops-audit-agent"}
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))

    def github_actions(self, max_runs: int = 15) -> str:
        """Registered workflows and recent run outcomes from the public GitHub API."""
        if not self.slug:
            return "UNAVAILABLE: repository is not hosted on GitHub; no Actions data"
        try:
            max_runs = max(1, min(int(max_runs), 30))
        except (TypeError, ValueError):
            max_runs = 15
        base = f"https://api.github.com/repos/{self.slug}/actions"
        out: list[str] = []
        try:
            workflows = self._http_get_json(f"{base}/workflows?per_page=50").get("workflows", [])
            out.append(f"{len(workflows)} registered workflow(s):")
            out += [f"  {w.get('path')} name='{w.get('name')}' state={w.get('state')}" for w in workflows]
            runs = self._http_get_json(f"{base}/runs?per_page={max_runs}").get("workflow_runs", [])
        except urllib.error.HTTPError as exc:
            return f"UNAVAILABLE: GitHub API returned HTTP {exc.code} (rate limit or Actions data not public)"
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            return f"UNAVAILABLE: could not reach GitHub API: {exc}"
        if not runs:
            out.append("No workflow runs recorded.")
            return "\n".join(out)
        outcomes = Counter((r.get("conclusion") or r.get("status") or "unknown") for r in runs)
        out.append(f"Last {len(runs)} run(s): " + ", ".join(f"{k}x{v}" for k, v in outcomes.most_common()))
        for r in runs:
            out.append(
                f"  {str(r.get('created_at', ''))[:10]} {r.get('name')} event={r.get('event')} "
                f"branch={r.get('head_branch')} -> {r.get('conclusion') or r.get('status')}"
            )
        return "\n".join(out)

    def scan_secrets(self, directory: str = "", glob: str = "") -> str:
        """Locate credential-like values and secret-bearing files. Values are never printed."""
        directory = (directory or "").replace("\\", "/").strip("/")
        hits: list[str] = []
        sensitive: list[str] = []
        scanned = 0
        for rel in self.tracked_files():
            if directory and not rel.startswith(directory + "/"):
                continue
            if glob and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(rel.rsplit("/", 1)[-1], glob)):
                continue
            if _SENSITIVE_NAME.search(rel):
                sensitive.append(rel)
            lines, err = self._read_text(rel)
            if err:
                continue
            scanned += 1
            for number, kind in scan_lines(lines):
                hits.append(f"{rel}:{number}: {kind}")
        out = [f"Scanned {scanned} text file(s); {len(hits)} credential-like value(s) found (values withheld)."]
        out += hits[:100]
        if len(hits) > 100:
            out.append(f"... {len(hits) - 100} more")
        if sensitive:
            out.append("Tracked files whose NAME suggests secrets (inspect with read_file; values are redacted):")
            out += [f"  {s}" for s in sensitive[:50]]
        return "\n".join(out)

    # ---------------------------------------------------------------- dispatch
    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a tool call by name; never raises."""
        func: Optional[Callable[..., str]] = TOOL_FUNCTIONS.get(name)
        if func is None:
            return f"ERROR: unknown tool '{name}'. Available: {', '.join(TOOL_FUNCTIONS)}"
        method = getattr(self, func)
        allowed = set(inspect.signature(method).parameters)
        unknown = set(arguments) - allowed
        if unknown:
            return f"ERROR: unexpected argument(s) {sorted(unknown)} for {name}; accepted: {sorted(allowed)}"
        try:
            return method(**arguments)
        except TypeError as exc:
            return f"ERROR: bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001 - tools must never crash the agent loop
            return f"ERROR: {name} failed: {type(exc).__name__}: {exc}"


TOOL_FUNCTIONS: dict[str, str] = {
    "list_files": "list_files",
    "read_file": "read_file",
    "search_code": "search_code",
    "git_log": "git_log",
    "run_ruff": "run_ruff",
    "github_actions": "github_actions",
    "scan_secrets": "scan_secrets",
}


def _schema(name: str, description: str, properties: dict[str, Any], required: Optional[list[str]] = None) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _schema(
        "list_files",
        "Discover repository files and structure (tracked files only, with sizes). Small repos are listed flat; "
        "large ones show one directory level at a time. Use `pattern` (glob, e.g. '*.yml') to find specific files.",
        {
            "directory": {"type": "string", "description": "Sub-directory to list; empty for repository root."},
            "pattern": {"type": "string", "description": "Optional glob filter on path or file name."},
            "recursive": {"type": "boolean", "description": "Force a flat recursive listing."},
        },
    ),
    _schema(
        "read_file",
        "Read a text file with line numbers (max ~300 lines per call; use start_line/end_line to page). Secret values are redacted.",
        {
            "path": {"type": "string", "description": "Repo-relative path, e.g. '.github/workflows/ci.yml'."},
            "start_line": {"type": "integer", "description": "First line (1-based)."},
            "end_line": {"type": "integer", "description": "Last line (inclusive)."},
        },
        ["path"],
    ),
    _schema(
        "search_code",
        "Regex search (Python syntax) across tracked text files. Returns file:line: text with secrets redacted.",
        {
            "pattern": {"type": "string", "description": "Regular expression."},
            "glob": {"type": "string", "description": "Optional glob restricting files, e.g. '*.py'."},
            "ignore_case": {"type": "boolean"},
            "max_results": {"type": "integer", "description": "Max lines returned (<=100)."},
        },
        ["pattern"],
    ),
    _schema(
        "git_log",
        "Inspect commit history. Optionally filter by path, commit-message regex, or a string added/removed in a commit (pickaxe, useful to spot committed-then-removed secrets).",
        {
            "max_count": {"type": "integer", "description": "Number of commits (<=100)."},
            "path": {"type": "string"},
            "message_grep": {"type": "string"},
            "pickaxe": {"type": "string"},
        },
    ),
    _schema(
        "run_ruff",
        "Static Python lint with ruff in isolated mode (ignores repo config, no cache, never executes repo code). "
        "Default rule set finds only real errors (syntax errors, undefined names, invalid comparisons).",
        {
            "path": {"type": "string", "description": "Optional file/directory to restrict the check."},
            "select": {"type": "string", "description": "Comma-separated ruff codes, default E9,F63,F7,F82."},
        },
    ),
    _schema(
        "github_actions",
        "List registered GitHub Actions workflows and recent run conclusions from the public GitHub API (may be unavailable).",
        {"max_runs": {"type": "integer", "description": "Recent runs to fetch (<=30)."}},
    ),
    _schema(
        "scan_secrets",
        "Find credential-like values (keys, tokens, passwords, private keys) and secret-bearing filenames across tracked files. "
        "Reports file:line and kind only; values are never shown. Always follow up with read_file to judge context.",
        {
            "directory": {"type": "string"},
            "glob": {"type": "string"},
        },
    ),
]
