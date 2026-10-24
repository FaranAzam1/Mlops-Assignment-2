"""Thin LLM layer: one `complete()` call returning tool calls. Only the basic OpenAI SDK is used."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Union

from auditor.repo import EXIT_LLM_FAILED, AuditError

RATE_LIMIT_ATTEMPTS = 6


class LLMError(AuditError):
    def __init__(self, message: str) -> None:
        super().__init__(message, EXIT_LLM_FAILED)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string exactly as produced by the model


@dataclass
class LLMReply:
    content: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)

    def as_message(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                for tc in self.tool_calls
            ]
        return msg


ToolChoice = Union[str, dict[str, Any]]


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: ToolChoice) -> LLMReply: ...


class OpenAIClient:
    """Chat-completions client with SDK-level retries and mapped, readable failures."""

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, timeout: float = 90.0, max_retries: int = 4) -> None:
        import openai  # imported lazily so unit tests do not need the SDK configured

        self._openai = openai
        self.model = model
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url or None, timeout=timeout, max_retries=max_retries)

    def _create_with_backoff(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: ToolChoice) -> Any:
        """Retry rate limits (free tiers hit them often) but fail fast when the quota/credit is exhausted."""
        openai = self._openai
        for attempt in range(1, RATE_LIMIT_ATTEMPTS + 1):
            try:
                return self._client.chat.completions.create(
                    model=self.model, messages=messages, tools=tools, tool_choice=tool_choice
                )
            except openai.RateLimitError as exc:
                text = str(exc).lower()
                if attempt == RATE_LIMIT_ATTEMPTS or any(s in text for s in ("insufficient_quota", "no credits", "billing", "exceeded your current quota")):
                    raise
                time.sleep(min(60.0, 5.0 * 2 ** (attempt - 1)))

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], tool_choice: ToolChoice) -> LLMReply:
        openai = self._openai
        try:
            resp = self._create_with_backoff(messages, tools, tool_choice)
        except openai.AuthenticationError:
            raise LLMError("OpenAI rejected the API key (authentication failed). Check OPENAI_API_KEY.") from None
        except openai.PermissionDeniedError as exc:
            raise LLMError(f"OpenAI denied access to model '{self.model}': {exc}") from None
        except openai.NotFoundError:
            raise LLMError(f"Model '{self.model}' was not found for this API key. Set OPENAI_MODEL to an available model.") from None
        except openai.RateLimitError as exc:
            raise LLMError(f"OpenAI rate limit / quota exceeded after retries: {exc}") from None
        except openai.APITimeoutError:
            raise LLMError("OpenAI request timed out after retries") from None
        except openai.APIConnectionError as exc:
            raise LLMError(f"Could not connect to the OpenAI API: {exc}") from None
        except openai.BadRequestError as exc:
            raise LLMError(f"OpenAI rejected the request: {exc}") from None
        except openai.APIError as exc:
            raise LLMError(f"OpenAI API error: {exc}") from None
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
            calls.append(ToolCall(id=call_id, name=tc.function.name, arguments=tc.function.arguments or "{}"))
        usage = {}
        if resp.usage:
            usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}
        return LLMReply(content=message.content, tool_calls=calls, usage=usage)
