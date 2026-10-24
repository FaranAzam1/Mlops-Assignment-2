"""LLM provider selection. Both providers are reached through the OpenAI chat-completions API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

from auditor.repo import EXIT_BAD_INPUT, AuditError

PROVIDERS: dict[str, dict[str, Optional[str]]] = {
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "base_env": "OPENAI_BASE_URL",
        "default_model": "gpt-4.1",
        "default_base": None,
    },
    "gemini": {
        "key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
        "base_env": "GEMINI_BASE_URL",
        "default_model": "gemini-2.5-flash",
        "default_base": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
}
CHOICES = ("auto", *PROVIDERS)


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    api_key: str
    model: str
    base_url: Optional[str]


def _get(env: Mapping[str, str], name: Optional[str]) -> str:
    return (env.get(name, "") if name else "").strip()


def resolve_provider(requested: Optional[str] = None, model: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> ProviderConfig:
    """Pick provider, key, model and endpoint from CLI/environment.

    Order: --provider, then $LLM_PROVIDER, then "auto". Auto uses the only configured key, and
    prefers OpenAI when both keys are set. Model: --model, then the provider's *_MODEL, then its default.
    """
    env = os.environ if env is None else env
    choice = (requested or _get(env, "LLM_PROVIDER") or "auto").lower()
    if choice not in CHOICES:
        raise AuditError(f"Unknown LLM provider '{choice}'. Choose one of: {', '.join(CHOICES)}", EXIT_BAD_INPUT)
    if choice == "auto":
        configured = [n for n, spec in PROVIDERS.items() if _get(env, spec["key_env"])]
        if not configured:
            keys = " or ".join(str(spec["key_env"]) for spec in PROVIDERS.values())
            raise AuditError(
                f"No LLM API key found. Set {keys} (copy .env.example to .env and add a key, or export it in the environment).",
                EXIT_BAD_INPUT,
            )
        choice = configured[0]
    spec = PROVIDERS[choice]
    api_key = _get(env, spec["key_env"])
    if not api_key:
        raise AuditError(
            f"{spec['key_env']} is not set but provider '{choice}' was selected. Add it to .env or the environment, "
            "or pick another provider with --provider / LLM_PROVIDER.",
            EXIT_BAD_INPUT,
        )
    return ProviderConfig(
        name=choice,
        api_key=api_key,
        model=model or _get(env, spec["model_env"]) or str(spec["default_model"]),
        base_url=_get(env, spec["base_env"]) or spec["default_base"],
    )
