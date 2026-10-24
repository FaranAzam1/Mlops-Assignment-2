import json
import os

import pytest

import agent as agent_cli
from auditor.repo import AuditError, RepoRef, clone_repo, make_workdir, parse_repo_url, remove_tree
from auditor.report import check_report
from tests.helpers import ScriptedLLM, call, make_repo


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/owner/name", ("owner", "name", None)),
        ("https://github.com/owner/name.git", ("owner", "name", None)),
        ("https://github.com/owner/name/", ("owner", "name", None)),
        ("http://www.github.com/Owner-1/na.me_x", ("Owner-1", "na.me_x", None)),
        ("https://github.com/owner/name/tree/dev", ("owner", "name", "dev")),
    ],
)
def test_parse_valid_urls(url, expected):
    ref = parse_repo_url(url)
    assert (ref.owner, ref.name, ref.branch) == expected
    assert ref.url == f"https://github.com/{expected[0]}/{expected[1]}"


@pytest.mark.parametrize(
    "url",
    ["", "   ", "not a url", "https://gitlab.com/o/r", "https://github.com/onlyowner", "https://github.com/o/r/issues",
     "ssh://git@github.com/o/r", "https://github.com.evil.com/o/r", "https://github.com/../x", "file:///etc/passwd",
     "https://github.com/o/r; rm -rf /"],
)
def test_parse_rejects_invalid_urls(url):
    with pytest.raises(AuditError) as err:
        parse_repo_url(url)
    assert err.value.exit_code == 2


def test_clone_local_repo_is_isolated_and_cleaned_up(tmp_path):
    src = make_repo(tmp_path / "src", {"a.txt": "hello\n"}, messages=("one", "two"))
    ref = parse_repo_url(str(src), allow_local=True)
    work = make_workdir()
    try:
        checkout = clone_repo(ref, work, depth=1)
        assert (checkout / "a.txt").read_text() == "hello\n"
    finally:
        remove_tree(work)
    assert not work.exists()


def test_clone_failure_maps_to_exit_code_3(tmp_path):
    ref = RepoRef("local", "missing", None, local_source=(tmp_path / "does-not-exist").as_uri())
    work = make_workdir()
    try:
        with pytest.raises(AuditError) as err:
            clone_repo(ref, work, timeout=60)
        assert err.value.exit_code == 3 and "Could not clone" in str(err.value)
    finally:
        remove_tree(work)


@pytest.fixture
def source_repo(tmp_path):
    return make_repo(tmp_path / "src", {
        "README.md": "# x\n",
        "app.py": "PASSWORD = 'hunter2hunter2'\n",
        ".github/workflows/ci.yml": "on: push\njobs:\n  t:\n    steps:\n      - run: pytest || true\n",
    })


def scripted_factory(steps):
    return lambda model: ScriptedLLM(steps)


GOOD_STEPS = [
    [call("list_files")], [call("read_file", path="app.py")], [call("scan_secrets")], [call("read_file", path=".github/workflows/ci.yml")],
    [call("record_violation", rule_id="SEC-001", severity="critical", file="app.py", line=1, description="Hardcoded password",
          evidence="PASSWORD literal in source", recommendation="Load from environment", evidence_quote="PASSWORD = '[REDACTED]'"),
     call("record_violation", rule_id="CI-003", severity="high", file=".github/workflows/ci.yml", line=None, description="Test failures ignored",
          evidence="run: pytest || true", recommendation="Remove || true", evidence_quote="run: pytest || true")],
    [call("finish", summary="done")],
]


def test_end_to_end_writes_valid_report_and_trace(source_repo, tmp_path):
    out = tmp_path / "out" / "report.json"
    code = agent_cli.main(["--repo", str(source_repo), "--output", str(out), "--allow-local-repo", "--quiet"],
                          llm_factory=scripted_factory(GOOD_STEPS))
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert check_report(report) == []
    assert report["repository"].startswith("https://github.com/local/")
    assert {v["rule_id"] for v in report["violations"]} == {"SEC-001", "CI-003"}
    assert "hunter2hunter2" not in out.read_text(encoding="utf-8")
    trace = (tmp_path / "out" / "report.trace.jsonl").read_text(encoding="utf-8")
    assert '"phase": "tool_call"' in trace and "hunter2hunter2" not in trace


def test_empty_violations_still_produces_valid_json(source_repo, tmp_path):
    steps = [[call("list_files")], [call("read_file", path="README.md")], [call("git_log")], [call("scan_secrets")], [call("finish", summary="clean")]]
    out = tmp_path / "r.json"
    assert agent_cli.main(["--repo", str(source_repo), "--output", str(out), "--allow-local-repo", "--quiet", "--no-trace"],
                          llm_factory=scripted_factory(steps)) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["violations"] == []
    assert not (tmp_path / "r.trace.jsonl").exists()


def test_invalid_url_exits_2_without_writing(tmp_path, capsys):
    out = tmp_path / "r.json"
    code = agent_cli.main(["--repo", "https://example.com/x/y", "--output", str(out)], llm_factory=scripted_factory([]))
    assert code == 2 and not out.exists()
    assert "Invalid repository URL" in capsys.readouterr().err


def test_missing_api_key_exits_2_with_hint(monkeypatch, tmp_path, capsys):
    for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(agent_cli, "_load_env", lambda: None)
    code = agent_cli.main(["--repo", "https://github.com/o/r", "--output", str(tmp_path / "r.json")])
    assert code == 2 and "OPENAI_API_KEY" in capsys.readouterr().err


def test_clone_failure_exits_3(tmp_path, capsys):
    def failing_clone(ref, workdir, depth=200, timeout=180):
        raise AuditError("Could not clone x: repository not found or not public", 3)

    out = tmp_path / "r.json"
    code = agent_cli.main(["--repo", "https://github.com/o/r", "--output", str(out), "--quiet"],
                          llm_factory=scripted_factory([]), clone_fn=failing_clone)
    assert code == 3 and not out.exists() and "not found or not public" in capsys.readouterr().err


def test_llm_failure_exits_4_and_cleans_up(source_repo, tmp_path, capsys):
    from auditor.llm import LLMError

    class Broken:
        def complete(self, *a, **k):
            raise LLMError("OpenAI rejected the API key (authentication failed).")

    out = tmp_path / "r.json"
    code = agent_cli.main(["--repo", str(source_repo), "--output", str(out), "--allow-local-repo", "--quiet"], llm_factory=lambda m: Broken())
    assert code == 4 and not out.exists() and "authentication failed" in capsys.readouterr().err


def test_unexpected_exception_is_reported_not_raised(source_repo, tmp_path, capsys):
    def boom(ref, workdir, depth=200, timeout=180):
        raise RuntimeError("kaboom")

    code = agent_cli.main(["--repo", "https://github.com/o/r", "--output", str(tmp_path / "r.json"), "--quiet"],
                          llm_factory=scripted_factory([]), clone_fn=boom)
    assert code == 1 and "kaboom" in capsys.readouterr().err


def test_bad_numeric_options_are_rejected(tmp_path):
    assert agent_cli.main(["--repo", "https://github.com/o/r", "--max-steps", "1"], llm_factory=scripted_factory([])) == 2


def test_environment_example_lists_required_variables():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, ".env.example"), encoding="utf-8").read()
    assert "OPENAI_API_KEY=" in text and "sk-" not in text
