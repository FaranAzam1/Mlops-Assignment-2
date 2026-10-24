# DevOps Repository Auditing Agent

An AI agent that autonomously audits a **public GitHub repository** for DevOps best-practice violations
(repository hygiene, dependencies, code quality, testing, GitHub Actions/CI, security, configuration,
collaboration, build/release) and writes an evidence-based, machine-readable JSON report.

```
python agent.py --repo https://github.com/<owner>/<repo> --output report.json
```

The LLM drives the investigation: it chooses which inspection tool to call next from what it has observed,
follows up on surprises, records findings, and decides when it is done. There is no fixed checklist and no
framework (no LangChain / LangGraph / n8n); the loop, tool dispatch and control flow are plain Python. Only the
basic `openai` SDK is used for the model call.

## 1. Installation

Requirements: **Python 3.10+** and **git** on `PATH`.

```bash
git clone https://github.com/FaranAzam1/devops-repo-auditing-agent.git
cd devops-repo-auditing-agent

python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# Linux/macOS:           source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Configuration

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Edit `.env` and set your key:

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | yes | API key for the LLM |
| `OPENAI_MODEL` | no | Chat model supporting tool calling (default `gpt-4.1`; also `--model`) |
| `OPENAI_BASE_URL` | no | OpenAI-compatible endpoint |
| `GITHUB_TOKEN` | no | Only raises the GitHub API rate limit for the `github_actions` tool (public data; no scopes needed) |

Variables already present in the environment take precedence over `.env`.

## 3. Running

```bash
python agent.py --repo https://github.com/example/demo-project --output report.json
```

It runs unattended (no prompts). On success it exits `0` and writes valid JSON, including an empty
`violations` array when nothing is found. Progress is logged to stderr (`--quiet` silences it).

| Option | Default | Meaning |
|---|---|---|
| `--repo URL` | required | Public `https://github.com/<owner>/<repo>` (a `/tree/<branch>` suffix selects a branch) |
| `--output PATH` | `report.json` | Report path (parent directories are created) |
| `--model NAME` | `$OPENAI_MODEL` or `gpt-4.1` | LLM model |
| `--max-steps N` | `40` | Maximum LLM turns; at the limit the agent is forced to finish |
| `--timeout SEC` | `900` | Wall-clock budget for the investigation; at the limit the agent is forced to finish |
| `--clone-depth N` | `200` | Git history depth fetched (affects `git_log`) |
| `--trace PATH` / `--no-trace` | `<output>.trace.jsonl` | Tool-call trace (see below) |

**Exit codes:** `0` success · `1` unexpected error · `2` invalid URL/options or missing API key ·
`3` clone failed (not found / not public / network) · `4` LLM failure (auth, quota, timeout after retries).
On any nonzero exit a one-line `error: ...` message is printed to stderr and no report is written.

### Output

```json
{
  "repository": "https://github.com/example/demo-project",
  "violations": [
    {
      "rule_id": "SEC-001",
      "category": "security",
      "severity": "critical",
      "file": "config.py",
      "line": 17,
      "description": "Hardcoded API credential",
      "evidence": "Credential assigned directly in source code",
      "recommendation": "Use a secrets manager or environment variable"
    }
  ]
}
```

* `severity` ∈ `critical | high | medium | low`. `file` / `line` are JSON `null` when not applicable
  (for example "README is missing").
* `category` ∈ `repository_hygiene, dependency_management, code_quality, testing, github_actions_ci, security,
  configuration, collaboration, build_release`; `rule_id`s come from the catalogue in
  [`auditor/rules.py`](auditor/rules.py) (`SEC-001`, `CI-003`, ...; `<PREFIX>-900` when nothing fits).
* **Secret values are never written** to the report, trace, or logs; they are redacted before the model even sees them.

[`sample_report.json`](sample_report.json) is real output from the agent (with its trace in
[`sample_report.trace.jsonl`](sample_report.trace.jsonl)).

## 4. Architecture

```
agent.py                CLI: args -> clone (isolated temp dir) -> AuditAgent.run() -> JSON report -> cleanup
tools.py                RepoTools: the inspection tools + their JSON schemas (read-only, redacted output)
auditor/loop.py         The explicit agent loop (Observe -> Reason -> Select tool -> Execute -> Interpret -> Report)
auditor/llm.py          LLM client (OpenAI chat completions with tool calling, retries, readable failures)
auditor/prompts.py      System prompt: role, evidence rules, DevOps areas, severity guide
auditor/rules.py        Rule catalogue / categories / severities (shared vocabulary, not a scanner)
auditor/report.py       Finding validation against the real repo; report schema check; atomic write
auditor/redact.py       Secret detection + redaction applied to every outbound string
auditor/repo.py         URL validation, safe shallow clone, cleanup, fatal-error type + exit codes
tests/                  85 automated tests (tools, redaction, validation, loop, CLI) using a scripted LLM
```

### The loop

1. **Observe** – the harness summarises the checkout (file counts, top-level layout, file types, workflow files, latest commit).
2. **Reason / Select tool** – the LLM is called with `tool_choice="required"`; it must answer with tool calls, and every
   call carries a `reasoning` argument (what the last result showed and why this call is next). Its choice of tool and
   arguments is the *only* thing that decides what is inspected next, including follow-ups.
3. **Execute** – the tool runs; output is redacted, size-capped, and returned as a `tool` message.
4. **Interpret** – the LLM reads the result and either investigates further or calls `record_violation`.
5. **Report** – `finish` ends the loop (rejected until a minimum amount of real inspection has happened); the harness
   writes the validated report. Step/time limits force a `finish` so a run always terminates.

The context is compacted for long investigations (old tool outputs are trimmed, the findings ledger is re-injected).

### Inspection tools

| Tool | Purpose |
|---|---|
| `list_files(directory, pattern, recursive)` | Tracked files with sizes; flat for small repos, one level at a time for large ones. Tracked-only, so committed junk (`.pyc`, `.env`, ...) is visible |
| `read_file(path, start_line, end_line)` | Numbered, paged, redacted file content. Blocks `..`, absolute paths, `.git/`, binaries, files > 1 MB |
| `search_code(pattern, glob, ...)` | Regex search across tracked text files (time-boxed, redacted) |
| `git_log(max_count, path, message_grep, pickaxe)` | Commit history; `pickaxe` finds commits that added/removed a string (committed-then-deleted secrets) |
| `run_ruff(path, select)` | Static Python lint, **isolated** (`--isolated --no-cache`): ignores repo config, writes nothing, never imports repo code. Default rule set only reports real errors |
| `github_actions(max_runs)` | Registered workflows and recent run conclusions via the public GitHub API; reports `UNAVAILABLE` if rate-limited/offline |
| `scan_secrets(directory, glob)` | Credential-like values and secret-bearing filenames as `file:line kind` (values withheld) |
| `record_violation` / `retract_violation` / `finish` | Findings ledger and termination (handled by the loop) |

Not provided on purpose: `run_pytest()` and `git_status()`. Running a stranger's tests means executing untrusted code
(the spec requires adequate isolation, which a plain subprocess is not), and a fresh clone's `git status` is always clean.
Test health is instead judged from CI evidence (`github_actions`) and static inspection, and the agent is told never to
claim failing tests without evidence.

### How false positives are controlled

* The prompt encodes the **evidence rule**: an absent optional file is not a defect when an equivalent exists (lint config in
  `pyproject.toml`, `uv.lock`/`poetry.lock`, README variants, ...), and remote branch protection is never inferred.
* `record_violation` is **verified against the real repository**: the file must exist, the line must exist, and an optional
  `evidence_quote` must literally appear in the (redacted) file; the line number is corrected to match the quote. Rejections
  are returned to the model so it can fix or drop the claim. Duplicates are refused and `retract_violation` withdraws a mistake.
* Rule ids and categories are normalised to the catalogue.

### Safety

* Target repositories are never modified: the clone lives in a temp directory that is deleted afterwards, tools are read-only,
  `ruff` runs without cache/fixes, and the original GitHub repository is only ever read.
* No repository code is executed: clone uses `core.symlinks=false`, no submodules, `GIT_LFS_SKIP_SMUDGE=1`, no credential prompts,
  `protocol.ext` disabled; paths are confined to the checkout.
* Only `https://github.com/<owner>/<repo>` URLs are accepted (shell metacharacters, other hosts, `file://` are rejected).
* Failures degrade gracefully: tool errors/unavailable checks are returned to the model as text (nothing is invented), API calls
  retry with backoff, timeouts and step limits are enforced, and any fatal problem exits nonzero with a clear message.

### Trace

`<output>.trace.jsonl` holds one JSON event per line: the observation, every tool call with its arguments and the model's
`reasoning`, a preview of each result, every recorded/rejected finding, and token usage. It is how a reviewer can verify
genuine agent behaviour. All text in it is redacted.

## 5. Tests

```bash
pytest
```

The 85 tests need no network or API key: they build throw-away git repositories and drive the real loop with a scripted
LLM. They cover every tool (including path-traversal, binary, redaction and unavailable-API cases), the redactor, finding
validation, the agent loop (dynamic tool selection from earlier results, early-finish rejection, bad tool arguments,
step/time limits, compaction, trace contents), URL validation, cloning, report schema/atomic write and every CLI exit code.

An optional live evaluation against a repository with deliberately injected defects (needs your API key):

```bash
python scripts/eval_fixture.py
```

## 6. Limitations

* History is limited to `--clone-depth` commits.
* Without `GITHUB_TOKEN` the GitHub API allows ~60 requests/hour per IP; `github_actions` reports `UNAVAILABLE` if it is exceeded.
* Results depend on the chosen model; detection quality is probabilistic, which is why findings are verified against the repository.
* Python projects get static lint (`run_ruff`); other languages are inspected through files, search and CI evidence.
