#!/usr/bin/env python3
"""Live evaluation on three local fixture repositories (needs OPENAI_API_KEY).

(A) obvious defects, (B) subtler CI/security/config defects, (C) a healthy repository for
false positives. Fixtures are built at run time in a temp directory; expectations below are
for *this evaluation only* and are not known to the agent.

    python scripts/eval_fixture.py            # all three
    python scripts/eval_fixture.py A C        # a subset
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent as agent_cli  # noqa: E402

AWS = "AKIA" + "IOSFODNN7EXAMPLE"

HEALTHY_CI = """name: CI
on:
  push:
    branches: [main]
  pull_request:
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: pytest
"""

HEALTHY = {
    "README.md": "# Healthy demo\n\nSmall library.\n\n## Install\n`pip install -r requirements.txt`\n\n## Test\n`pytest`\n",
    "LICENSE": "MIT License\n",
    ".gitignore": "__pycache__/\n*.pyc\n.venv/\n.env\n.pytest_cache/\n.ruff_cache/\ndist/\n",
    "pyproject.toml": '[project]\nname = "healthy"\nversion = "1.0"\nrequires-python = ">=3.10"\n\n[tool.ruff]\nline-length = 100\n',
    "requirements.txt": "requests==2.32.3\npytest==8.3.3\nruff==0.6.9\n",
    ".env.example": "API_TOKEN=\n",
    "healthy/__init__.py": "",
    "healthy/core.py": "import os\n\n\ndef token() -> str:\n    return os.environ.get('API_TOKEN', '')\n\n\ndef add(a: int, b: int) -> int:\n    return a + b\n",
    "tests/test_core.py": "from healthy.core import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    ".github/workflows/ci.yml": HEALTHY_CI,
    ".github/CODEOWNERS": "* @example\n",
}

OBVIOUS = {
    "main.py": "import requests\nimport flask\n\nAWS_ACCESS_KEY_ID = '" + AWS + "'\n\ndef run():\n    return requests.get('http://x')\n",
    ".env": "DATABASE_PASSWORD=Sup3rS3cretPassw0rd\nDEBUG=true\n",
    "__pycache__/main.cpython-311.pyc": b"\x00\x01compiled",
    "node_modules/leftpad/index.js": "module.exports = 1;\n",
    "requirements.txt": "requests\nflask\n",
    "app.log": "started\n",
}

SUBTLE_CI = """name: CI
on:
  push:
    branches: [mian]
  pull_request_target:
permissions: write-all
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@master
        with:
          ref: ${{ github.event.pull_request.head.sha }}
      - run: pip install -r requirements.txt
      - run: pytest
        continue-on-error: true
      - run: echo "Deploying with ${{ secrets.DEPLOY_TOKEN }}"
"""

SUBTLE = {
    "README.md": "# Subtle demo\n",
    ".gitignore": "__pycache__/\n.env\n",
    "requirements.txt": "requests==2.31.0\nflask>=2.0\n",
    "pyproject.toml": '[project]\nname = "subtle"\nversion = "0.1"\ndependencies = ["requests>=2.0,<2.5", "flask==1.0"]\n',
    "subtle/__init__.py": "",
    "subtle/io.py": "DATA_DIR = '/home/johnsmith/projects/data'\n\n\ndef load():\n    return open(DATA_DIR + '/in.csv').read()\n",
    "tests/test_io.py": "def test_ok():\n    assert True\n",
    ".github/workflows/ci.yml": SUBTLE_CI,
}

# (rule prefixes or ids that count as a hit, file or None) - coarse on purpose
EXPECTED = {
    "A": [({"SEC-001"}, "main.py"), ({"SEC-001", "SEC-002"}, ".env"), ({"REPO-001"}, None), ({"REPO-003"}, None),
          ({"TEST-001"}, None), ({"CI-001"}, None), ({"DEP-004", "DEP-003"}, None), ({"REPO-002"}, None)],
    "B": [({"SEC-003"}, ".github/workflows/ci.yml"), ({"SEC-004"}, ".github/workflows/ci.yml"), ({"CI-003"}, ".github/workflows/ci.yml"),
          ({"CI-002"}, ".github/workflows/ci.yml"), ({"CI-004"}, ".github/workflows/ci.yml"), ({"SEC-005"}, ".github/workflows/ci.yml"),
          ({"CFG-001"}, "subtle/io.py"), ({"DEP-002"}, None)],
    "C": [],
}
FIXTURES = {"A": OBVIOUS, "B": SUBTLE, "C": HEALTHY}


def build(root: Path, files: dict) -> Path:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    ident = ["-c", "user.name=Eval", "-c", "user.email=eval@example.com", "-c", "commit.gpgsign=false"]
    for args in (["init", "-q", "-b", "main"], ["add", "-A", "-f"], [*ident, "commit", "-q", "-m", "initial commit"]):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)
    return root


def hit(violation: dict, expected: tuple[set[str], str | None]) -> bool:
    rules, file = expected
    return violation["rule_id"] in rules and (file is None or violation["file"] == file)


def main(argv: list[str]) -> int:
    names = [a.upper() for a in argv] or ["A", "B", "C"]
    outdir = Path(tempfile.mkdtemp(prefix="eval-"))
    exit_code = 0
    for name in names:
        repo = build(outdir / f"repo_{name}", FIXTURES[name])
        out = outdir / f"report_{name}.json"
        print(f"\n=== Fixture {name} ===", file=sys.stderr)
        code = agent_cli.main(["--repo", str(repo), "--output", str(out), "--allow-local-repo"])
        if code != 0:
            print(f"fixture {name}: agent exited with {code}")
            exit_code = code
            continue
        violations = json.loads(out.read_text(encoding="utf-8"))["violations"]
        expected = EXPECTED[name]
        found = [e for e in expected if any(hit(v, e) for v in violations)]
        useful = [v for v in violations if any(hit(v, e) for e in expected)]
        precision = len(useful) / len(violations) if violations else (1.0 if not expected else 0.0)
        recall = len(found) / len(expected) if expected else 1.0
        print(f"\nFixture {name}: {len(violations)} reported | recall {recall:.0%} ({len(found)}/{len(expected)}) | precision {precision:.0%}")
        for v in violations:
            tag = "HIT " if any(hit(v, e) for e in expected) else "extra"
            print(f"  [{tag}] {v['rule_id']:<10} {v['severity']:<8} {v['file']}:{v['line']}  {v['description'][:80]}")
        for e in expected:
            if e not in found:
                print(f"  [MISS] {sorted(e[0])} {e[1]}")
    print(f"\nReports and traces are in {outdir}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
