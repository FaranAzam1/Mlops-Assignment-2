"""Keep tests independent of the developer's real .env and shell environment."""

import pytest

import agent as agent_cli

PROVIDER_VARS = (
    "LLM_PROVIDER",
    "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "OPENAI_FALLBACK_MODELS",
    "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_BASE_URL", "GEMINI_FALLBACK_MODELS",
    "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "ANTHROPIC_BASE_URL", "ANTHROPIC_FALLBACK_MODELS",
    "GITHUB_TOKEN",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for var in PROVIDER_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(agent_cli, "_load_env", lambda: None)
