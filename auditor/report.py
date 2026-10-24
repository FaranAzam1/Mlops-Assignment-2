"""Violation validation/normalisation and JSON report writing."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from auditor.redact import redact_text
from auditor.rules import CATEGORIES, SEVERITIES, normalise_rule

VIOLATION_FIELDS = ("rule_id", "category", "severity", "file", "line", "description", "evidence", "recommendation")
MAX_TEXT = 600


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def validate_violation(raw: dict[str, Any], tools: Any, existing: list[dict[str, Any]]) -> tuple[Optional[dict[str, Any]], str]:
    """Check a model-proposed violation against the real repository.

    Returns (violation, message). When violation is None the message explains the rejection so
    the model can correct itself; otherwise the message lists any automatic adjustments.
    """
    notes: list[str] = []

    severity = str(raw.get("severity", "")).strip().lower()
    if severity not in SEVERITIES:
        return None, f"REJECTED: severity must be one of {list(SEVERITIES)}, got '{raw.get('severity')}'"

    rule_id, category, note = normalise_rule(str(raw.get("rule_id", "")), str(raw.get("category", "") or "").strip().lower())
    if category not in CATEGORIES:
        return None, f"REJECTED: provide a catalogue rule_id (e.g. SEC-001) or a category from {list(CATEGORIES)}"
    if note:
        notes.append(note)

    texts: dict[str, str] = {}
    for key in ("description", "evidence", "recommendation"):
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            return None, f"REJECTED: '{key}' is required and must be a non-empty string"
        value = redact_text(_norm_ws(value))
        texts[key] = value if len(value) <= MAX_TEXT else value[: MAX_TEXT - 1] + "…"

    file = raw.get("file")
    if isinstance(file, str):
        file = file.replace("\\", "/").strip()
        while file.startswith("./"):
            file = file[2:]
        if file.lower() in {"", "null", "none", "n/a"}:
            file = None
    elif file is not None:
        return None, "REJECTED: 'file' must be a string path or null"
    if file is not None and not tools.file_exists(file):
        return None, (
            f"REJECTED: file '{file}' does not exist in the repository. If the defect is something that is MISSING, "
            "set file to null and name the missing item in the description; otherwise use the exact path from list_files."
        )

    line = raw.get("line")
    if isinstance(line, str) and line.strip().isdigit():
        line = int(line.strip())
    if line is not None and (not isinstance(line, int) or isinstance(line, bool) or line < 1):
        return None, "REJECTED: 'line' must be a positive integer or null"
    if file is None and line is not None:
        line = None
        notes.append("line dropped because file is null")
    if file is not None and line is not None:
        total = tools.line_count(file)
        if total is not None and line > total:
            return None, f"REJECTED: line {line} is beyond the end of {file} ({total} lines)"

    quote = raw.get("evidence_quote")
    if isinstance(quote, str) and quote.strip() and file is not None:
        body = tools.file_text_redacted(file)
        if body is not None:
            if _norm_ws(quote) not in _norm_ws(body):
                return None, (
                    f"REJECTED: evidence_quote was not found in {file}. Re-read the file and quote the text exactly "
                    "(secret values appear as [REDACTED]); evidence must be grounded in tool output."
                )
            first = _norm_ws(next((ln for ln in quote.splitlines() if ln.strip()), quote))
            matches = [n for n, text in enumerate(body.splitlines(), start=1) if first and first in _norm_ws(text)]
            if matches and line not in matches:
                best = min(matches, key=lambda n: abs(n - line)) if line is not None else matches[0]
                if line is not None:
                    notes.append(f"line corrected {line} -> {best} to match the quoted evidence")
                line = best

    violation = {
        "rule_id": rule_id,
        "category": category,
        "severity": severity,
        "file": file,
        "line": line,
        "description": texts["description"],
        "evidence": texts["evidence"],
        "recommendation": texts["recommendation"],
    }
    for other in existing:
        if other and other["rule_id"] == rule_id and other["file"] == file and (other["line"] == line or line is None or other["line"] is None):
            return None, f"REJECTED: duplicate of an already recorded violation ({rule_id} at {file or 'repository level'}). Do not repeat findings."
    return violation, ("; ".join(notes) if notes else "")


def check_report(report: Any) -> list[str]:
    """Return a list of schema problems (empty when the report honours the output contract)."""
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["report must be an object"]
    if not isinstance(report.get("repository"), str) or not report["repository"]:
        errors.append("'repository' must be a non-empty string")
    violations = report.get("violations")
    if not isinstance(violations, list):
        return errors + ["'violations' must be an array"]
    for i, v in enumerate(violations):
        if not isinstance(v, dict):
            errors.append(f"violations[{i}] must be an object")
            continue
        missing = [f for f in VIOLATION_FIELDS if f not in v]
        if missing:
            errors.append(f"violations[{i}] missing {missing}")
            continue
        if v["severity"] not in SEVERITIES:
            errors.append(f"violations[{i}].severity invalid")
        if v["file"] is not None and not isinstance(v["file"], str):
            errors.append(f"violations[{i}].file must be string or null")
        if v["line"] is not None and (not isinstance(v["line"], int) or isinstance(v["line"], bool)):
            errors.append(f"violations[{i}].line must be integer or null")
        for key in ("rule_id", "category", "description", "evidence", "recommendation"):
            if not isinstance(v[key], str) or not v[key].strip():
                errors.append(f"violations[{i}].{key} must be a non-empty string")
    return errors


def write_report(repository: str, violations: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    """Validate and atomically write the JSON report; returns the report dict."""
    report = {"repository": repository, "violations": violations}
    problems = check_report(report)
    if problems:
        raise ValueError("internal error: report violates the output contract: " + "; ".join(problems))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, output)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return report
