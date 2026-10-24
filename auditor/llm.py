"""Thin LLM layer: one `complete()` call returning tool calls. Only the basic OpenAI SDK is used."""

from __future__ import annotations

import copy
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Union

from auditor.repo import EXIT_LLM_FAILED, AuditError

RATE_LIMIT_ATTEMPTS = 6


_EXHAUSTED_MARKERS = ("insufficient_quota", "no credits", "credit_balance", "perday", "per day", "daily")


def is_transient_limit(exc: Exception, text: str) -> bool:
    """True for per-minute rate limits and 'API not enabled yet' propagation errors, False for exhausted quota."""
    low = text.lower()
    status = getattr(exc, "status_code", None)
    if status is None or (isinstance(status, int) and status >= 500):
        return True  # timeouts, dropped connections, 5xx 'high demand'
    if status == 403:
        return "service_disabled" in low or "has not been used in project" in low  # new-project propagation delay
    return not any(marker in low for marker in _EXHAUSTED_MARKERS)


def retry_delay(text: str, attempt: int) -> float:
    """Honour the provider's suggested delay ('retry in 34s' / retryDelay '34s'), else exponential backoff."""
    m = re.search(r"retry in ([\d.]+)\s*s|retrydelay\W+([\d.]+)s", text, re.IGNORECASE)
    if m:
        return min(90.0, float(m.group(1) or m.group(2)) + 1.0)
    return min(60.0, 5.0 * 2 ** (attempt - 1))


class LLMError(AuditError):
    def __init__(self, message: str) -> None:
        super().__init__(message, EXIT_LLM_FAILED)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string exactly as produced by the model
    extra_content: Optional[dict[str, Any]] = None  # provider metadata that must be echoed back (Gemini thought signatures)


@dataclass
class LLMReply:
    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    def as_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = []
            for tc in self.tool_calls:
                entry: dict[str, Any] = {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                if tc.extra_content:
                    entry["extra_content"] = tc.extra_content
                msg["tool_calls"].append(entry)
        return msg


ToolChoice = Union[str, dict[str, Any]]


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: ToolChoice) -> LLMReply: ...


class OpenAIClient:
    """Chat-completions client with SDK-level retries and mapped, readable failures."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        timeout: float = 90.0,
        max_retries: int = 2,
        fallback_models: tuple[str, ...] = (),
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        import openai  # imported lazily so unit tests do not need the SDK configured

        self._openai = openai
        self.model = model
        self._fallbacks = list(fallback_models)
        self._log = log or (lambda msg: print(msg, file=sys.stderr))
        self._gemini = "generativelanguage.googleapis.com" in (base_url or "")
        self._switched = False
        self.waited_seconds = 0.0  # time spent backing off; excluded from the agent's investigation budget
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout, max_retries=max_retries)

    def _create_with_backoff(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: ToolChoice) -> Any:
        """Retry transient rate limits / propagation errors; fail fast when quota or credit is truly exhausted."""
        openai = self._openai
        attempt = 0
        while True:
            try:
                return self._client.chat.completions.create(
                    model=self.model, messages=self._payload(messages), tools=tools, tool_choice=tool_choice
                )
            except (
                openai.RateLimitError, openai.PermissionDeniedError, openai.NotFoundError,
                openai.InternalServerError, openai.APITimeoutError, openai.APIConnectionError,
            ) as exc:
                text = str(exc)
                transient = not isinstance(exc, openai.NotFoundError) and is_transient_limit(exc, text)
                if transient and attempt + 1 < RATE_LIMIT_ATTEMPTS:
                    attempt += 1
                    delay = retry_delay(text, attempt)
                    self.waited_seconds += delay
                    time.sleep(delay)
                    continue
                if not self._switch_model(exc):
                    raise
                attempt = 0

    def _switch_model(self, exc: Exception) -> bool:
        """Move to the next fallback model after a quota/availability failure; False when none is left."""
        if not self._fallbacks:
            return False
        previous, self.model = self.model, self._fallbacks.pop(0)
        self._switched = True
        self._log(f"[llm] {previous} unavailable ({type(exc).__name__}); switching to {self.model}")
        return True

    def _payload(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """After a model switch, Gemini rejects another model's thought signatures; send its documented bypass value."""
        if not (self._switched and self._gemini):
            return messages
        patched = copy.deepcopy(messages)
        for msg in patched:
            for call in msg.get("tool_calls") or []:
                call["extra_content"] = {"google": {"thought_signature": "skip_thought_signature_validator"}}
        return patched

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: ToolChoice) -> LLMReply:
        openai = self._openai
        try:
            resp = self._create_with_backoff(messages, tools, tool_choice)
        except openai.AuthenticationError:
            raise LLMError("The provider rejected the API key (authentication failed). Check the key for the selected provider in .env.") from None
        except openai.PermissionDeniedError as exc:
            raise LLMError(f"The provider denied access to model '{self.model}': {exc}") from None
        except openai.NotFoundError:
            raise LLMError(f"Model '{self.model}' was not found or is not available to this API key. Choose another with --model or the provider's *_MODEL setting.") from None
        except openai.RateLimitError as exc:
            raise LLMError(f"LLM rate limit / quota exceeded (all configured models tried): {exc}") from None
        except openai.APITimeoutError:
            raise LLMError("The LLM request timed out after retries") from None
        except openai.APIConnectionError as exc:
            raise LLMError(f"Could not connect to the LLM API: {exc}") from None
        except openai.BadRequestError as exc:
            raise LLMError(f"The LLM API rejected the request: {exc}") from None
        except openai.APIError as exc:
            raise LLMError(f"LLM API error: {exc}") from None
        if not resp.choices:
            raise LLMError("OpenAI returned no choices")
        message = resp.choices[0].message
        self._call_counter = getattr(self, "_call_counter", 0)
        calls = []
        for tc in message.tool_calls or []:
            if getattr(tc, "type", "function") != "function":
                continue
            self._call_counter += 1
            # some OpenAI-compatible providers omit or repeat tool-call ids; ids must be unique per conversation
            call_id = tc.id if tc.id and not any(c.id == tc.id for c in calls) else f"call_{self._call_counter}_{tc.function.name}"
            extra = (getattr(tc, "model_extra", None) or {}).get("extra_content")
            calls.append(
                ToolCall(id=call_id, name=tc.function.name, arguments=tc.function.arguments or "{}", extra_content=extra or None)
            )
        usage = {}
        if resp.usage:
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
        return LLMReply(content=message.content, tool_calls=calls, usage=usage)
