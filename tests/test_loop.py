import json

import pytest

import auditor.loop as loop_mod
from auditor.loop import ALL_SCHEMAS, AuditAgent, TraceWriter
from tests.helpers import ScriptedLLM, call, make_repo
from tools import RepoTools

SECRET = "sk-" + "Q1w2E3r4T5y6U7i8O9p0A1s2D3"


@pytest.fixture(scope="module")
def tools(tmp_path_factory):
    root = make_repo(
        tmp_path_factory.mktemp("loop") / "r",
        {
            "README.md": "# demo\n",
            "app.py": f"KEY = '{SECRET}'\n",
            ".github/workflows/ci.yml": "on: push\njobs:\n  t:\n    steps:\n      - run: pytest || true\n",
        },
    )
    return RepoTools(root)


def record(**over):
    args = dict(rule_id="SEC-001", severity="critical", file="app.py", line=1, description="Hardcoded key",
                evidence="KEY literal", recommendation="Use env var", evidence_quote="KEY = '[REDACTED]'")
    args.update(over)
    return call("record_violation", **args)


def inspect_steps(n=4):
    return [[call("list_files")], [call("read_file", path="app.py")], [call("search_code", pattern="pytest")], [call("git_log")]][:n]


def run(tools, steps, **kw):
    llm = ScriptedLLM(steps)
    agent = AuditAgent(llm, tools, "https://github.com/o/r", log=lambda m: None, **kw)
    return agent, llm, agent.run()


def test_full_cycle_records_and_finishes(tools):
    agent, llm, violations = run(tools, [*inspect_steps(), [record()], [call("finish", summary="checked")]])
    assert agent.finished and len(violations) == 1
    assert violations[0]["rule_id"] == "SEC-001" and violations[0]["file"] == "app.py"
    assert agent.inspection_calls == 4


def test_tool_selection_follows_the_llm_not_a_fixed_script(tools):
    # Second call depends on the *content* of the first result: the agent must feed results back.
    def follow_up(messages):
        listing = ScriptedLLM.tool_results(messages)[-1]
        target = ".github/workflows/ci.yml" if "ci.yml" in listing else "README.md"
        return [call("read_file", path=target)]

    agent, llm, _ = run(tools, [[call("list_files")], follow_up, [call("scan_secrets")], [call("git_log")], [call("finish", summary="s")]])
    read_args = [json.loads(tc["function"]["arguments"]) for m in llm.requests[-1]["messages"] if m.get("tool_calls") for tc in m["tool_calls"]]
    assert {"path": ".github/workflows/ci.yml"} == {k: v for k, v in read_args[1].items() if k != "reasoning"}


def test_results_are_redacted_before_reaching_the_llm(tools):
    agent, llm, _ = run(tools, [[call("read_file", path="app.py")], [call("search_code", pattern="KEY")], [call("scan_secrets")],
                                [call("git_log")], [call("finish", summary="s")]])
    transcript = json.dumps(llm.requests[-1]["messages"])
    assert SECRET not in transcript and "[REDACTED]" in transcript


def test_early_finish_is_rejected_until_enough_inspection(tools):
    agent, llm, _ = run(tools, [[call("finish", summary="too soon")], *inspect_steps(), [call("finish", summary="now")]])
    results = ScriptedLLM.tool_results(llm.requests[-1]["messages"])
    assert results[0].startswith("REJECTED") and agent.finished


def test_invalid_findings_are_bounced_back_for_correction(tools):
    bad = record(file="ghost.py")
    agent, llm, violations = run(tools, [*inspect_steps(), [bad], [record()], [record()], [call("finish", summary="s")]])
    results = ScriptedLLM.tool_results(llm.requests[-1]["messages"])
    assert any("does not exist" in r for r in results) and any("duplicate" in r for r in results)
    assert len(violations) == 1


def test_retract_removes_a_finding(tools):
    agent, _, violations = run(tools, [*inspect_steps(), [record()], [call("retract_violation", index=0, why="mistake")], [call("finish", summary="s")]])
    assert violations == []


def test_malformed_arguments_unknown_tools_and_multi_call_turns(tools):
    steps = [[("read_file", "{not json")], [call("no_such_tool")], [call("list_files"), call("git_log"), call("scan_secrets")],
             [call("search_code", pattern="x")], [call("finish", summary="s")]]
    agent, llm, _ = run(tools, steps)
    results = ScriptedLLM.tool_results(llm.requests[-1]["messages"])
    assert results[0].startswith("ERROR: tool arguments were not valid JSON")
    assert "unknown tool" in results[1]
    # every assistant tool_call got a matching tool message (required by the chat API)
    msgs = llm.requests[-1]["messages"]
    ids = [tc["id"] for m in msgs if m.get("tool_calls") for tc in m["tool_calls"]]
    answered = [m["tool_call_id"] for m in msgs if m["role"] == "tool"]
    assert ids == answered


def test_step_limit_forces_finish_and_still_returns_findings(tools):
    steps = [[record()]] + [[call("list_files")] for _ in range(10)] + [[call("finish", summary="forced")]]
    agent, llm, violations = run(tools, steps, max_steps=5)
    assert agent.stop_reason == "step limit reached" and agent.finished
    assert llm.requests[-1]["tool_choice"] == {"type": "function", "function": {"name": "finish"}}
    assert len(violations) == 1


def test_model_that_never_calls_tools_does_not_hang(tools):
    agent, _, violations = run(tools, [], max_steps=4)
    assert not agent.finished and violations == [] and agent.stop_reason


def test_time_limit_forces_finish(tools):
    llm = ScriptedLLM([[call("finish", summary="s")]])
    agent = AuditAgent(llm, tools, "u", deadline_seconds=-1, log=lambda m: None)
    agent.run()
    assert agent.stop_reason == "wall-clock time limit reached" and agent.finished


def test_every_tool_schema_demands_reasoning_property_and_valid_shape():
    for schema in ALL_SCHEMAS:
        params = schema["function"]["parameters"]
        assert "reasoning" in params["properties"] and params["type"] == "object"
    names = {s["function"]["name"] for s in ALL_SCHEMAS}
    assert {"list_files", "read_file", "search_code", "git_log", "record_violation", "finish"} <= names


def test_context_compaction_trims_old_tool_output_and_keeps_ledger(tools, monkeypatch):
    monkeypatch.setattr(loop_mod, "COMPACT_THRESHOLD_CHARS", 800)
    monkeypatch.setattr(loop_mod, "KEEP_RECENT_MESSAGES", 4)
    monkeypatch.setattr(loop_mod, "ELIDE_MIN_CHARS", 20)
    steps = [[record()]] + [[call("list_files", recursive=True)] for _ in range(6)] + [[call("finish", summary="s")]]
    agent, llm, _ = run(tools, steps, max_steps=20)
    final = llm.requests[-1]["messages"]
    assert any("older tool outputs were trimmed" in (m.get("content") or "") for m in final)
    assert any("SEC-001" in (m.get("content") or "") and m["role"] == "user" for m in final)


def test_trace_file_records_reasoning_and_tool_calls(tools, tmp_path):
    trace_path = tmp_path / "t.jsonl"
    trace = TraceWriter(trace_path)
    llm = ScriptedLLM([*inspect_steps(), [record()], [call("finish", summary="s")]])
    AuditAgent(llm, tools, "u", trace=trace, log=lambda m: None).run()
    trace.close()
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert events[0]["phase"] == "observe" and events[-1]["phase"] == "end"
    tool_events = [e for e in events if e.get("phase") == "tool_call"]
    assert [e["tool"] for e in tool_events] == ["list_files", "read_file", "search_code", "git_log"]
    assert all(e["reasoning"] == "test" for e in tool_events)
    assert SECRET not in trace_path.read_text(encoding="utf-8")


def test_observation_summarises_repository(tools):
    obs = AuditAgent(ScriptedLLM([]), tools, "https://github.com/o/r").observe()
    assert "Tracked files: 3" in obs and ".github/workflows/ci.yml" in obs and "README.md" in obs
