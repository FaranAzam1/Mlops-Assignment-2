import subprocess
import urllib.error

import pytest

from tests.helpers import git, make_repo
from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS, RepoTools

SECRET = "sk-" + "Z9y8X7w6V5u4T3s2R1q0P9o8N7"


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    files = {
        "README.md": "# Demo\n",
        "src/app.py": "import os\n\nAPI_KEY = '" + SECRET + "'\n\ndef main():\n    return os.getcwd()\n",
        "src/broken.py": "def f(:\n    pass\n",
        "src/undefined.py": "def g():\n    return missing_name\n",
        ".github/workflows/ci.yml": "name: ci\non: [push]\njobs:\n  t:\n    runs-on: ubuntu-latest\n    steps:\n      - run: pytest || true\n",
        "tests/test_app.py": "def test_x():\n    assert True\n",
        "data/blob.bin": b"\x00\x01\x02binary",
        ".env": "DB_PASSWORD=hunter22xyz\n",
        "pkg/__pycache__/m.cpython-311.pyc": b"\x00pyc",
    }
    return make_repo(tmp_path_factory.mktemp("shared") / "r", files, messages=("initial commit", "add ci", "fix typo"))


@pytest.fixture(scope="module")
def tools(repo):
    return RepoTools(repo, slug="owner/demo")


def test_every_schema_has_an_implementation():
    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    assert names == set(TOOL_FUNCTIONS)
    for required in ("list_files", "read_file", "search_code", "git_log"):
        assert required in names


def test_list_files_flat_for_small_repo(tools):
    out = tools.list_files()
    assert "src/app.py" in out and ".github/workflows/ci.yml" in out and "__pycache__" in out  # tracked artefacts visible


def test_list_files_pattern_and_directory(tools):
    assert "ci.yml" in tools.list_files(pattern="*.yml")
    assert "app.py" in tools.list_files(directory="src")
    assert "No tracked files" in tools.list_files(directory="nope")


def test_list_files_collapses_large_directories(tmp_path):
    root = make_repo(tmp_path / "big", {f"pkg{i % 5}/f{i}.txt": "x" for i in range(400)})
    out = RepoTools(root).list_files()
    assert "too many to list flat" in out and "pkg0/ (80 files)" in out


def test_read_file_numbers_lines_and_redacts_secrets(tools):
    out = tools.read_file("src/app.py")
    assert "    3: API_KEY = '[REDACTED]'" in out
    assert SECRET not in out


def test_read_file_ranges_and_paging(tools, tmp_path):
    out = tools.read_file("src/app.py", start_line=5, end_line=6)
    assert "lines 5-6 of 6" in out and "def main" in out
    big = RepoTools(make_repo(tmp_path / "long", {"f.txt": "\n".join(f"l{i}" for i in range(1000))}))
    paged = big.read_file("f.txt")
    assert "lines 1-300 of 1000" in paged and "start_line=301" in paged


@pytest.mark.parametrize("bad", ["../outside.txt", "src/../../x", ".git/config", "/etc/passwd", ""])
def test_read_file_blocks_escapes(tools, bad):
    assert tools.read_file(bad).startswith("ERROR")


def test_read_file_handles_missing_binary_and_directory(tools):
    assert "not found" in tools.read_file("nope.txt")
    assert "binary" in tools.read_file("data/blob.bin")
    assert "not a regular file" in tools.read_file("src")


def test_env_file_values_are_redacted(tools):
    out = tools.read_file(".env")
    assert "hunter22xyz" not in out and "DB_PASSWORD=[REDACTED]" in out


def test_search_code_finds_and_redacts(tools):
    out = tools.search_code(r"API_KEY")
    assert "src/app.py:3:" in out and SECRET not in out
    assert "0 match" in tools.search_code("definitely-not-present-xyz")


def test_search_code_glob_case_and_errors(tools):
    assert "ci.yml" in tools.search_code("pytest", glob="*.yml")
    assert "app.py" not in tools.search_code("pytest", glob="*.yml")
    assert tools.search_code("(unclosed").startswith("ERROR: invalid regular expression")
    assert tools.search_code("").startswith("ERROR")


def test_git_log_variants(tools):
    out = tools.git_log()
    assert "3 commit(s) reachable" in out and "fix typo" in out and "initial commit" in out
    assert "add ci" in tools.git_log(message_grep="ci")
    assert "No commits matched" in tools.git_log(message_grep="zzz-none")
    assert "initial commit" in tools.git_log(path="README.md")
    assert tools.git_log(path="../x").startswith("ERROR")


def test_git_log_pickaxe_finds_commit_that_removed_secret(tmp_path):
    root = make_repo(tmp_path / "px", {"a.txt": "token_marker_123\n"})
    (root / "a.txt").write_text("clean\n", encoding="utf-8")
    git(root, "commit", "-aqm", "remove marker")
    out = RepoTools(root).git_log(pickaxe="token_marker_123")
    assert "remove marker" in out and "initial commit" in out


def test_git_log_without_history(tmp_path):
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "a.txt").write_text("x")
    assert RepoTools(tmp_path / "plain").git_log().startswith("UNAVAILABLE")


def test_run_ruff_reports_real_errors_and_leaves_repo_untouched(tools, repo):
    out = tools.run_ruff()
    assert "F821" in out or "syntax" in out.lower()
    assert "src/undefined.py" in out
    assert git(repo, "status", "--porcelain").strip() == ""  # no cache or fixes written into the target


def test_run_ruff_input_validation_and_no_python(tools, tmp_path):
    assert tools.run_ruff(select="E9; rm -rf").startswith("ERROR")
    assert RepoTools(make_repo(tmp_path / "nopy", {"a.md": "x"})).run_ruff().startswith("UNAVAILABLE")


def test_scan_secrets_never_prints_values(tools):
    out = tools.scan_secrets()
    assert "src/app.py:3:" in out and ".env:1:" in out
    assert SECRET not in out and "hunter22xyz" not in out
    assert ".env" in out.split("NAME suggests secrets")[1]


def test_github_actions_summarises_runs(tools, monkeypatch):
    def fake(url):
        if "workflows" in url:
            return {"workflows": [{"path": ".github/workflows/ci.yml", "name": "ci", "state": "active"}]}
        return {"workflow_runs": [
            {"created_at": "2025-01-02T00:00:00Z", "name": "ci", "event": "push", "head_branch": "main", "conclusion": "failure"},
            {"created_at": "2025-01-01T00:00:00Z", "name": "ci", "event": "push", "head_branch": "main", "conclusion": "success"},
        ]}

    monkeypatch.setattr(tools, "_http_get_json", fake)
    out = tools.github_actions()
    assert "1 registered workflow" in out and "failurex1" in out and "successx1" in out


def test_github_actions_degrades_gracefully(tools, monkeypatch):
    def forbidden(url):
        raise urllib.error.HTTPError(url, 403, "rate limited", {}, None)

    monkeypatch.setattr(tools, "_http_get_json", forbidden)
    assert tools.github_actions().startswith("UNAVAILABLE") and "403" in tools.github_actions()

    def offline(url):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(tools, "_http_get_json", offline)
    assert tools.github_actions().startswith("UNAVAILABLE")
    assert RepoTools(tools.root, slug=None).github_actions().startswith("UNAVAILABLE")


def test_execute_never_raises(tools):
    assert tools.execute("nope", {}).startswith("ERROR: unknown tool")
    assert "unexpected argument" in tools.execute("read_file", {"path": "README.md", "bogus": 1})
    assert tools.execute("read_file", {}).startswith("ERROR")
    assert "match(es)" in tools.execute("search_code", {"pattern": "x", "max_results": "abc"})


def test_tools_do_not_modify_the_repository(tools, repo):
    for name, args in [("list_files", {}), ("read_file", {"path": "src/app.py"}), ("search_code", {"pattern": "def"}),
                       ("git_log", {}), ("run_ruff", {}), ("scan_secrets", {})]:
        tools.execute(name, args)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout
    assert status.strip() == ""
