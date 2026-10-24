import pytest

from auditor.providers import resolve_provider
from auditor.repo import AuditError

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_auto_uses_the_only_configured_key():
    cfg = resolve_provider(env={"GEMINI_API_KEY": "g-key"})
    assert (cfg.name, cfg.api_key, cfg.model, cfg.base_url) == ("gemini", "g-key", "gemini-2.5-flash", GEMINI_URL)
    cfg = resolve_provider(env={"OPENAI_API_KEY": "o-key"})
    assert (cfg.name, cfg.model, cfg.base_url) == ("openai", "gpt-4.1", None)


def test_auto_prefers_openai_when_both_are_set():
    assert resolve_provider(env={"OPENAI_API_KEY": "o", "GEMINI_API_KEY": "g"}).name == "openai"


def test_llm_provider_env_and_cli_override_precedence():
    env = {"OPENAI_API_KEY": "o", "GEMINI_API_KEY": "g", "LLM_PROVIDER": "gemini"}
    assert resolve_provider(env=env).name == "gemini"
    assert resolve_provider("openai", env=env).api_key == "o"  # --provider beats $LLM_PROVIDER


def test_model_and_endpoint_overrides():
    env = {"GEMINI_API_KEY": "g", "GEMINI_MODEL": "gemini-2.5-pro", "GEMINI_BASE_URL": "http://proxy/v1"}
    cfg = resolve_provider(env=env)
    assert (cfg.model, cfg.base_url) == ("gemini-2.5-pro", "http://proxy/v1")
    assert resolve_provider(model="cli-model", env=env).model == "cli-model"


def test_whitespace_only_keys_count_as_missing():
    with pytest.raises(AuditError):
        resolve_provider(env={"OPENAI_API_KEY": "   ", "GEMINI_API_KEY": ""})


def test_errors_are_actionable_and_exit_2():
    with pytest.raises(AuditError) as err:
        resolve_provider(env={})
    assert err.value.exit_code == 2 and "OPENAI_API_KEY" in str(err.value) and "GEMINI_API_KEY" in str(err.value)
    with pytest.raises(AuditError) as err:
        resolve_provider("gemini", env={"OPENAI_API_KEY": "o"})
    assert "GEMINI_API_KEY is not set" in str(err.value)
    with pytest.raises(AuditError) as err:
        resolve_provider("claude", env={})
    assert "Unknown LLM provider" in str(err.value) and err.value.exit_code == 2
