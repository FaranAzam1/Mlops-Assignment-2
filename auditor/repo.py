"""Safe acquisition of the target repository.

The repository is cloned read-only into a temporary directory. Nothing from it is ever
executed: hooks are not copied by `git clone`, symlinks are materialised as plain files,
Git LFS smudging is skipped and submodules are not fetched.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional


class AuditError(Exception):
    """Fatal, user-facing error with a process exit code."""

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


EXIT_BAD_INPUT = 2
EXIT_CLONE_FAILED = 3
EXIT_LLM_FAILED = 4

_GITHUB_URL = re.compile(
    r"^https?://(?:www\.)?github\.com/(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9_.\-]{0,99}))/"
    r"(?P<name>[A-Za-z0-9_.\-]{1,100}?)(?:\.git)?(?:/tree/(?P<branch>[^\s?#]+))?/?$"
)


@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str
    branch: Optional[str] = None
    local_source: Optional[str] = None  # development-only: path/file:// source

    @property
    def url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"

    @property
    def clone_source(self) -> str:
        return self.local_source or f"{self.url}.git"

    @property
    def slug(self) -> Optional[str]:
        return None if self.local_source else f"{self.owner}/{self.name}"


def parse_repo_url(url: str, allow_local: bool = False) -> RepoRef:
    """Validate the user-supplied repository URL (public github.com only)."""
    url = (url or "").strip()
    if not url:
        raise AuditError("--repo must be a non-empty GitHub repository URL", EXIT_BAD_INPUT)
    match = _GITHUB_URL.match(url)
    if match:
        name = match.group("name")
        if name in {".", ".."}:
            raise AuditError(f"Invalid GitHub repository URL: {url}", EXIT_BAD_INPUT)
        return RepoRef(match.group("owner"), name, match.group("branch"))
    if allow_local:
        path = Path(url[len("file://"):] if url.startswith("file://") else url)
        if path.exists():
            resolved = path.resolve()
            return RepoRef("local", resolved.name, None, local_source=resolved.as_uri())
    raise AuditError(
        f"Invalid repository URL '{url}'. Expected https://github.com/<owner>/<repo>",
        EXIT_BAD_INPUT,
    )


def _on_rm_error(func: Callable, path: str, _exc: object) -> None:
    """Windows keeps git objects read-only; clear the flag and retry."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, onerror=_on_rm_error)


def make_workdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="devops-audit-"))


def clone_repo(ref: RepoRef, workdir: Path, depth: int = 200, timeout: int = 180) -> Path:
    """Shallow-clone `ref` into `workdir/repo` and return the checkout path."""
    dest = workdir / "repo"
    cmd = [
        "git",
        "-c", "core.symlinks=false",
        "-c", "core.autocrlf=false",
        "-c", "protocol.ext.allow=never",
        "-c", f"protocol.file.allow={'always' if ref.local_source else 'never'}",
        "clone", "--quiet", "--no-tags", "--single-branch", "--no-recurse-submodules",
        "--depth", str(depth),
    ]
    if ref.branch:
        cmd += ["--branch", ref.branch]
    cmd += ["--", ref.clone_source, str(dest)]
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",   # never block on credential prompts
        "GIT_LFS_SKIP_SMUDGE": "1",
        "GIT_ASKPASS": "echo",
    }
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, encoding="utf-8", errors="replace"
        )
    except FileNotFoundError:
        raise AuditError("git is not installed or not on PATH", EXIT_CLONE_FAILED) from None
    except subprocess.TimeoutExpired:
        raise AuditError(f"Timed out after {timeout}s while cloning {ref.url}", EXIT_CLONE_FAILED) from None
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        low = stderr.lower()
        if any(s in low for s in ("not found", "could not read username", "authentication failed", "terminal prompts disabled")):
            reason = "repository not found or not public"
        elif "could not resolve host" in low or "unable to access" in low:
            reason = "network error while contacting GitHub"
        else:
            reason = stderr.splitlines()[-1] if stderr else "unknown git error"
        raise AuditError(f"Could not clone {ref.url}: {reason}", EXIT_CLONE_FAILED)
    return dest
