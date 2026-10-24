import json

import pytest

from auditor.report import check_report, validate_violation, write_report
from tests.helpers import make_repo
from tools import RepoTools


@pytest.fixture(scope="module")
def tools(tmp_path_factory):
    root = make_repo(
        tmp_path_factory.mktemp("rep") / "r",
        {
            "config.py": "import os\n\nSETTINGS = 1\nAPI_KEY = 'abcdef123456XYZ'\n",
            ".github/workflows/ci.yml": "on: push\njobs:\n  t:\n    steps:\n      - run: pytest || true\n",
        },
    )
    return RepoTools(root)


def good(**over):
    base = dict(
        rule_id="SEC-001", category="security", severity="critical", file="config.py", line=4,
        description="Hardcoded API credential", evidence="API_KEY assigned a literal", recommendation="Use an env var",
    )
    base.update(over)
    return base


def test_valid_violation_has_exactly_the_contract_fields(tools):
    v, msg = validate_violation(good(), tools, [])
    assert v is not None and msg == ""
    assert list(v) == ["rule_id", "category", "severity", "file", "line", "description", "evidence", "recommendation"]


def test_rejects_bad_severity_missing_text_and_unknown_file(tools):
    assert validate_violation(good(severity="urgent"), tools, [])[0] is None
    assert validate_violation(good(evidence=" "), tools, [])[0] is None
    v, msg = validate_violation(good(file="nope.py"), tools, [])
    assert v is None and "does not exist" in msg and "null" in msg


def test_line_must_exist_and_missing_file_forces_null_line(tools):
    assert validate_violation(good(line=999), tools, [])[0] is None
    v, msg = validate_violation(good(rule_id="REPO-001", category="repository_hygiene", file=None, line=7), tools, [])
    assert v["file"] is None and v["line"] is None and "dropped" in msg


def test_category_is_derived_from_rule_prefix_and_unknown_ids_normalised(tools):
    v, _ = validate_violation(good(category="testing"), tools, [])
    assert v["category"] == "security"
    v, msg = validate_violation(good(rule_id="MADE-UP", category="security"), tools, [])
    assert v["rule_id"] == "SEC-900" and "unknown rule id" in msg


def test_evidence_quote_is_verified_and_corrects_line(tools):
    v, msg = validate_violation(good(line=1, evidence_quote="API_KEY = '[REDACTED]'"), tools, [])
    assert v["line"] == 4 and "corrected" in msg
    v, msg = validate_violation(good(evidence_quote="this text is not in the file"), tools, [])
    assert v is None and "not found" in msg


def test_quote_fills_missing_line(tools):
    v, _ = validate_violation(
        good(rule_id="CI-003", category="github_actions_ci", severity="high", file=".github/workflows/ci.yml", line=None,
             evidence_quote="run: pytest || true"), tools, [])
    assert v["line"] == 5


def test_secret_values_in_model_text_are_redacted(tools):
    v, _ = validate_violation(good(evidence="password = 'hunter2hunter2' is set"), tools, [])
    assert "hunter2hunter2" not in v["evidence"]


def test_duplicates_are_rejected(tools):
    first, _ = validate_violation(good(), tools, [])
    again, msg = validate_violation(good(description="different words"), tools, [first])
    assert again is None and "duplicate" in msg
    other, _ = validate_violation(good(file=".github/workflows/ci.yml", line=1), tools, [first])
    assert other is not None


def test_check_report_accepts_empty_and_rejects_malformed():
    assert check_report({"repository": "https://github.com/a/b", "violations": []}) == []
    assert check_report({"repository": "x"})
    assert check_report({"repository": "x", "violations": [{"rule_id": "A"}]})
    bad = {"rule_id": "A-1", "category": "c", "severity": "urgent", "file": None, "line": None,
           "description": "d", "evidence": "e", "recommendation": "r"}
    assert any("severity" in e for e in check_report({"repository": "x", "violations": [bad]}))


def test_write_report_is_valid_json_with_nulls(tmp_path):
    v = dict(rule_id="REPO-001", category="repository_hygiene", severity="medium", file=None, line=None,
             description="No README", evidence="none found", recommendation="Add one")
    out = tmp_path / "nested" / "report.json"
    write_report("https://github.com/a/b", [v], out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["violations"][0]["file"] is None and data["violations"][0]["line"] is None
    assert not list(out.parent.glob("*.tmp"))


def test_write_report_refuses_contract_violations(tmp_path):
    with pytest.raises(ValueError):
        write_report("x", [{"rule_id": "A"}], tmp_path / "r.json")
    assert not (tmp_path / "r.json").exists()
