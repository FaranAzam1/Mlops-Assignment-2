import pytest

from auditor.providers import resolve_provider
from auditor.repo import AuditError

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def test_auto_uses_the_only_configured_key():
    cfg = resolve_provider(env={"GEMINI_API_KEY": "g-key"})
    assert (cfg.name, cfg.api_key, cfg.model, cfg.base_url) == ("gemini", "g-key", "gemini-3.5-flash", GEMINI_URL)
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


def test_rate_limit_classification_and_delay_parsing():
    from auditor.llm import is_transient_limit, retry_delay

    class Err(Exception):
        def __init__(self, status):
            self.status_code = status

    per_minute = "429 You exceeded your current quota... Please retry in 34.5s."
    assert is_transient_limit(Err(429), per_minute)
    assert not is_transient_limit(Err(429), "insufficient_quota: you have no credits remaining")
    assert not is_transient_limit(Err(429), "GenerateRequestsPerDayPerProjectPerModel-FreeTier exceeded")
    assert is_transient_limit(Err(403), "Gemini API has not been used in project 1 before or it is disabled. SERVICE_DISABLED")
    assert not is_transient_limit(Err(403), "permission denied for this model")
    assert is_transient_limit(Err(503), "high demand") and is_transient_limit(Err(None), "timeout")
    assert retry_delay(per_minute, 1) == 35.5
    assert retry_delay("'retryDelay': '12s'", 1) == 13.0
    assert retry_delay("no hint", 3) == 20.0 and retry_delay("no hint", 9) == 60.0


def test_gemini_has_default_fallbacks_and_they_can_be_overridden():
    cfg = resolve_provider(env={"GEMINI_API_KEY": "g"})
    assert cfg.fallback_models and cfg.model not in cfg.fallback_models
    cfg = resolve_provider(env={"GEMINI_API_KEY": "g", "GEMINI_FALLBACK_MODELS": "a, b ,a,gemini-3.5-flash"})
    assert cfg.fallback_models == ("a", "b")
    assert resolve_provider(env={"GEMINI_API_KEY": "g", "GEMINI_FALLBACK_MODELS": ""}).fallback_models == ()
    assert resolve_provider(env={"OPENAI_API_KEY": "o"}).fallback_models == ()
