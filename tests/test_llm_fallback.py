import openai
import pytest

from auditor.llm import LLMError, OpenAIClient

GEMINI = "https://generativelanguage.googleapis.com/v1beta/openai/"
DAILY = "Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier limit: 20. Please retry in 12h."


def api_error(cls, status, text):
    """Build an SDK error without an HTTP library (the SDK's HTTP backend differs between versions)."""
    exc = cls.__new__(cls)
    Exception.__init__(exc, text)
    exc.status_code, exc.message, exc.body = status, text, None
    return exc


class _Msg:
    content = None
    tool_calls = []


class _Resp:
    choices = [type("C", (), {"message": _Msg()})()]
    usage = None


class FakeCompletions:
    def __init__(self, plan):
        self.plan, self.calls = plan, []

    def create(self, **kw):
        self.calls.append(kw)
        outcome = self.plan.get(kw["model"], _Resp())
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def client(plan, fallbacks, base_url=None):
    c = OpenAIClient("k", "m1", base_url=base_url, fallback_models=fallbacks, log=lambda m: None)
    fake = FakeCompletions(plan)
    c._client = type("X", (), {"chat": type("Y", (), {"completions": fake})()})()
    return c, fake


def test_switches_to_next_model_when_daily_quota_is_exhausted():
    c, fake = client({"m1": api_error(openai.RateLimitError, 429, DAILY)}, ("m2", "m3"))
    c.complete([{"role": "user", "content": "x"}], [], "required")
    assert [call["model"] for call in fake.calls] == ["m1", "m2"] and c.model == "m2"


def test_skips_retired_models_and_walks_the_whole_list():
    plan = {
        "m1": api_error(openai.RateLimitError, 429, DAILY),
        "m2": api_error(openai.NotFoundError, 404, "model is no longer available to new users"),
    }
    c, fake = client(plan, ("m2", "m3"))
    c.complete([], [], "required")
    assert [call["model"] for call in fake.calls] == ["m1", "m2", "m3"]


def test_raises_readable_error_when_every_model_is_exhausted():
    plan = {m: api_error(openai.RateLimitError, 429, DAILY) for m in ("m1", "m2")}
    c, _ = client(plan, ("m2",))
    with pytest.raises(LLMError) as err:
        c.complete([], [], "required")
    assert err.value.exit_code == 4 and "rate limit" in str(err.value).lower() and "all configured models" in str(err.value)


def test_no_fallback_configured_fails_immediately_on_exhausted_quota():
    c, fake = client({"m1": api_error(openai.RateLimitError, 429, "insufficient_quota: no credits")}, ())
    with pytest.raises(LLMError):
        c.complete([], [], "required")
    assert len(fake.calls) == 1


def test_transient_rate_limit_is_retried_on_the_same_model(monkeypatch):
    monkeypatch.setattr("auditor.llm.time.sleep", lambda s: None)
    c, fake = client({}, ("m2",))
    seq = [api_error(openai.RateLimitError, 429, "Please retry in 3s."), _Resp()]
    fake.plan = {}
    original = fake.create

    def flaky(**kw):
        if seq and isinstance(seq[0], Exception):
            fake.calls.append(kw)
            raise seq.pop(0)
        return original(**kw)

    fake.create = flaky
    c.complete([], [], "required")
    assert c.model == "m1" and len(fake.calls) == 2


def test_gemini_history_gets_signature_bypass_only_after_a_switch():
    history = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}, "extra_content": {"google": {"thought_signature": "orig"}}}]}]
    c, fake = client({"m1": api_error(openai.RateLimitError, 429, DAILY)}, ("m2",), base_url=GEMINI)
    c.complete(history, [], "required")
    assert fake.calls[0]["messages"][0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "orig"
    assert fake.calls[1]["messages"][0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "skip_thought_signature_validator"
    assert history[0]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "orig"  # caller's list untouched


def test_overloaded_503_is_retried_then_falls_back(monkeypatch):
    monkeypatch.setattr("auditor.llm.time.sleep", lambda s: None)
    c, fake = client({"m1": api_error(openai.InternalServerError, 503, "high demand")}, ("m2",))
    c.complete([], [], "required")
    assert [call["model"] for call in fake.calls].count("m1") == 6 and fake.calls[-1]["model"] == "m2"
