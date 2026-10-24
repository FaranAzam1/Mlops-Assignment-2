"""Secret detection and redaction.

Every piece of repository text is passed through this module *before* it reaches the
LLM, the trace file or the JSON report, so secret values are never exposed. Only the
fact that a secret exists (kind, file, line) survives.
"""

from __future__ import annotations

import re
from typing import Iterable

REDACTED = "[REDACTED]"

_TOKEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b")),
    ("api_key_sk", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("stripe_live_key", re.compile(r"\b[sr]k_live_[0-9a-zA-Z]{16,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
]

# user:password@host in URLs; the span covers only the password.
_URL_CRED = re.compile(r"://[^/\s:@]+:([^@\s/]{3,})@")

# NAME = "value" / NAME: value / "name": "value" where NAME looks credential-like.
_ASSIGNMENT = re.compile(
    r"""
    (?P<key>["']?[\w.\-]*(?:secret|token|passw(?:or)?d|passwd|api[_\-]?key|private[_\-]?key|
        access[_\-]?key|auth[_\-]?key|credential)[\w.\-]*["']?)
    \s*[:=]\s*
    (?P<quote>["']?)
    (?P<value>[^\s"',;#)}\]]{6,})
    (?P=quote)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PRIVATE_KEY_BEGIN = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----")
_PRIVATE_KEY_END = re.compile(r"-----END [A-Z ]*PRIVATE KEY(?: BLOCK)?-----")

_PLACEHOLDER_WORDS = (
    "changeme", "change_me", "your", "example", "placeholder", "dummy", "redacted",
    "xxxx", "****", "todo", "fixme", "replace", "<", "insert", "sample", "none", "null",
)
_REFERENCE_MARKERS = (
    "os.environ", "getenv", "process.env", "secrets.", "env.", "vars.", "config.", "settings.",
    "input(", "getpass", "${", "{{", "%(", "$(",
)
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\[\]]*$")


def _is_placeholder(value: str, quoted: bool) -> bool:
    low = value.lower()
    if value.startswith(("$", "%", "<", "{", "@")):
        return True
    if any(marker in low for marker in _REFERENCE_MARKERS):
        return True
    if any(word in low for word in _PLACEHOLDER_WORDS):
        return True
    if low in {"true", "false", "none", "null", "undefined"}:
        return True
    if len(set(value)) <= 2:  # "aaaaaa", "******"
        return True
    # Unquoted identifiers (e.g. `password = password_input`) are variable references.
    if not quoted and _IDENTIFIER.match(value) and not any(c.isdigit() for c in value):
        return True
    return False


def find_secret_spans(line: str) -> list[tuple[str, int, int]]:
    """Return (kind, start, end) spans of secret *values* in a single line."""
    spans: list[tuple[str, int, int]] = []
    for kind, pattern in _TOKEN_PATTERNS:
        for m in pattern.finditer(line):
            spans.append((kind, m.start(), m.end()))
    for m in _URL_CRED.finditer(line):
        if not _is_placeholder(m.group(1), quoted=True):
            spans.append(("url_credentials", m.start(1), m.end(1)))
    for m in _ASSIGNMENT.finditer(line):
        value = m.group("value")
        if _is_placeholder(value, quoted=bool(m.group("quote"))):
            continue
        spans.append(("hardcoded_credential", m.start("value"), m.end("value")))
    begin = _PRIVATE_KEY_BEGIN.search(line)
    if begin:
        spans.append(("private_key", begin.start(), begin.end()))
    return _merge(spans)


def _merge(spans: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    spans = sorted(spans, key=lambda s: (s[1], -s[2]))
    merged: list[tuple[str, int, int]] = []
    for kind, start, end in spans:
        if merged and start < merged[-1][2]:
            prev = merged[-1]
            merged[-1] = (prev[0], prev[1], max(prev[2], end))
        else:
            merged.append((kind, start, end))
    return merged


def redact_line(line: str) -> str:
    """Replace every secret value in one line with a marker."""
    spans = find_secret_spans(line)
    if not spans:
        return line
    out, cursor = [], 0
    for kind, start, end in spans:
        out.append(line[cursor:start])
        out.append(f"{REDACTED}:{kind}" if kind == "private_key" else REDACTED)
        cursor = end
    out.append(line[cursor:])
    return "".join(out)


def redact_lines(lines: Iterable[str]) -> list[str]:
    """Redact a sequence of lines, blanking whole PEM private-key blocks."""
    result: list[str] = []
    in_key = False
    for line in lines:
        if in_key:
            result.append(f"{REDACTED}:private_key")
            if _PRIVATE_KEY_END.search(line):
                in_key = False
            continue
        if _PRIVATE_KEY_BEGIN.search(line):
            result.append(f"{REDACTED}:private_key")
            in_key = not _PRIVATE_KEY_END.search(line)
            continue
        result.append(redact_line(line))
    return result


def redact_text(text: str) -> str:
    """Redact multi-line text while preserving the line structure."""
    return "\n".join(redact_lines(text.split("\n")))


def scan_lines(lines: Iterable[str]) -> list[tuple[int, str]]:
    """Return (1-based line number, kind) for each secret found."""
    hits: list[tuple[int, str]] = []
    in_key = False
    for number, line in enumerate(lines, start=1):
        if in_key:
            if _PRIVATE_KEY_END.search(line):
                in_key = False
            continue
        if _PRIVATE_KEY_BEGIN.search(line):
            hits.append((number, "private_key"))
            in_key = not _PRIVATE_KEY_END.search(line)
            continue
        for kind, _, _ in find_secret_spans(line):
            hits.append((number, kind))
    return hits
