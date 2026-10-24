"""Rule catalogue, categories and severities used to normalise and validate findings.

The catalogue is *guidance for the LLM* (stable ids, shared vocabulary); it contains no
repository-specific knowledge and is not a scanner: the agent decides what to check.
"""

from __future__ import annotations

SEVERITIES = ("critical", "high", "medium", "low")

CATEGORIES = (
    "repository_hygiene",
    "dependency_management",
    "code_quality",
    "testing",
    "github_actions_ci",
    "security",
    "configuration",
    "collaboration",
    "build_release",
)

PREFIX_TO_CATEGORY = {
    "REPO": "repository_hygiene",
    "DEP": "dependency_management",
    "QUAL": "code_quality",
    "TEST": "testing",
    "CI": "github_actions_ci",
    "SEC": "security",
    "CFG": "configuration",
    "COLLAB": "collaboration",
    "BUILD": "build_release",
}

# rule_id -> (default severity, summary)
RULES: dict[str, tuple[str, str]] = {
    "REPO-001": ("medium", "No README (or README is empty / placeholder)"),
    "REPO-002": ("medium", ".gitignore missing or unsuitable for the project's languages/tooling"),
    "REPO-003": ("medium", "Generated/temporary files tracked (pyc, __pycache__, node_modules, build output, logs, .DS_Store, virtualenvs)"),
    "DEP-001": ("high", "Code imports third-party packages but no dependency manifest exists"),
    "DEP-002": ("medium", "Dependency specifications are inconsistent across manifests (conflicting versions/packages)"),
    "DEP-003": ("medium", "Application has loose/unpinned dependencies AND no lockfile (exact == pins or a lockfile each satisfy this; then do NOT report it)"),
    "DEP-004": ("medium", "Weak version constraints (`*`, `latest`, unbounded or unpinned critical deps)"),
    "DEP-005": ("high", "Dependency fetched from an unsafe/unpinned source (plain http index, unpinned git branch)"),
    "QUAL-001": ("low", "No linting/formatting configuration for a non-trivial code base"),
    "QUAL-002": ("medium", "Obvious maintainability problem (syntax errors, undefined names, bare except, dead/duplicated code, giant functions)"),
    "TEST-001": ("high", "No automated tests for a non-trivial code base"),
    "TEST-002": ("high", "Tests are failing (evidence: CI runs or deterministic inspection)"),
    "TEST-003": ("high", "Tests exist but are never executed by CI"),
    "TEST-004": ("medium", "Tests are disabled/skipped wholesale or contain no assertions"),
    "CI-001": ("medium", "No CI workflow for a project that has code/tests to validate"),
    "CI-002": ("medium", "Incorrect or ineffective workflow triggers (branch filters that can never match, e.g. typos like 'mian' or a branch that is not the default; no pull_request/push trigger so it never runs)"),
    "CI-003": ("high", "Ineffective gates (continue-on-error, `|| true`, `if: false`, failures ignored)"),
    "CI-004": ("medium", "Third-party actions pinned to mutable refs (@master/@main/@latest) or unpinned tooling"),
    "CI-005": ("medium", "Workflow runs files, scripts, paths or commands that do not exist in the repository"),
    "CI-006": ("medium", "Recent CI runs are consistently failing"),
    "SEC-001": ("critical", "Hardcoded credential/API key/token/password in source or config"),
    "SEC-002": ("critical", "Private key, certificate bundle or secret-bearing file (.env, id_rsa, *.pem) committed"),
    "SEC-003": ("high", "Overly broad workflow permissions (write-all, missing least-privilege on privileged workflows)"),
    "SEC-004": ("high", "Unsafe workflow trigger/use of untrusted input (pull_request_target + PR checkout, ${{ github.event.* }} in run:)"),
    "SEC-005": ("high", "Secrets exposed in workflow logs or hardcoded in workflow files"),
    "SEC-006": ("high", "Insecure remote execution or disabled verification (curl | bash, verify=False, --insecure)"),
    "SEC-007": ("medium", "Insecure container practice (runs as root, secrets in ENV/ARG, mutable base tag)"),
    "CFG-001": ("medium", "Machine-specific absolute paths or hostnames committed"),
    "CFG-002": ("low", "Local/IDE/personal settings committed"),
    "CFG-003": ("medium", "Poor environment separation (debug on in prod config, prod endpoints hardcoded, no dev/prod split)"),
    "CFG-004": ("low", "Code requires environment variables but no example/documentation of them exists"),
    "COLLAB-001": ("high", "Unsafe PR automation (auto-merge/auto-approve without checks, bot with excess rights)"),
    "COLLAB-002": ("low", "No contribution/review guidance (CODEOWNERS, PR template, CONTRIBUTING) where collaboration is evident"),
    "BUILD-001": ("medium", "No build/package validation in CI for a buildable project"),
    "BUILD-002": ("high", "Risky release/deploy automation (deploys on every push without gates/approval, publishes unpinned artifacts)"),
    "BUILD-003": ("medium", "Build definition broken or non-reproducible (Dockerfile/Makefile references missing files, `latest` tags)"),
}


def catalogue_text() -> str:
    lines = [f"- {rule_id} [{sev}] {summary}" for rule_id, (sev, summary) in RULES.items()]
    lines.append("- <PREFIX>-900: use only when nothing above fits (PREFIX from the catalogue above)")
    return "\n".join(lines)


def normalise_rule(rule_id: str, category: str) -> tuple[str, str, str | None]:
    """Return (rule_id, category, note) coerced into the catalogue's vocabulary."""
    rid = (rule_id or "").strip().upper()
    note = None
    prefix = rid.split("-")[0] if "-" in rid else ""
    if prefix in PREFIX_TO_CATEGORY:
        derived = PREFIX_TO_CATEGORY[prefix]
        if category != derived:
            note = f"category set to '{derived}' to match rule prefix {prefix}"
        category = derived
        if rid not in RULES and not rid.endswith("-900"):
            note = f"unknown rule id {rid}; recorded as {prefix}-900"
            rid = f"{prefix}-900"
    elif category in CATEGORIES:
        prefix = next(p for p, c in PREFIX_TO_CATEGORY.items() if c == category)
        note = f"unknown rule id '{rule_id}'; recorded as {prefix}-900"
        rid = f"{prefix}-900"
    return rid, category, note
