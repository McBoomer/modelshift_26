import json
from types import SimpleNamespace

from modelshift.capture import instrument_openai
from modelshift.core import LLMRequest, LLMResponse
from modelshift.demo import run_demo
from modelshift.diff import compare, lenient_json, strict_json, validate_schema
from modelshift.suite import Case, build_suite
from modelshift.trace import Trace, TraceStore


def _case(expected_text, request=None):
    req = request or LLMRequest(messages=[{"role": "user", "content": "hi"}], system="sys")
    return Case("c1", req, LLMResponse(text=expected_text), "sig", "old")


def _symptoms(result):
    return {s["type"] for s in result.symptoms}


def test_json_helpers():
    assert strict_json('{"a": 1}') == {"a": 1}
    assert strict_json('```json\n{"a": 1}\n```') is None
    assert lenient_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}


def test_identical_output_passes():
    r = compare(
        _case('{"category": "bug", "urgency": "high"}'),
        LLMResponse(text='{"urgency": "high", "category": "bug"}'),
    )
    assert r.passed


def test_detects_fence_preamble_and_drift():
    exp = '{"category": "bug", "urgency": "high"}'
    act = 'Sure! Here you go:\n```json\n{"category": "bug", "urgency": "critical"}\n```'
    r = compare(_case(exp), LLMResponse(text=act))
    assert not r.passed
    assert {"code_fence", "preamble", "value_drift"} <= _symptoms(r)


def test_detects_refusal():
    r = compare(_case('{"a": 1}'), LLMResponse(text="I'm sorry, but I can't help with that."))
    assert _symptoms(r) == {"refusal"}


def test_tool_call_mismatch():
    case = Case(
        "c",
        LLMRequest(messages=[]),
        LLMResponse(tool_calls=[{"name": "search", "arguments": {"q": "x"}}]),
        "s",
        "old",
    )
    r = compare(case, LLMResponse(text="Here is an answer without searching"))
    assert "tool_mismatch" in _symptoms(r)


def test_schema_validation():
    schema = {
        "type": "object",
        "required": ["a"],
        "properties": {"a": {"type": "string", "enum": ["x"]}},
        "additionalProperties": False,
    }
    assert validate_schema({"a": "x"}, schema) == []
    assert len(validate_schema({"a": "y", "b": 1}, schema)) == 2


def test_suite_dedupes_and_balances(tmp_path):
    store = TraceStore(str(tmp_path / "t.db"))
    req = LLMRequest(messages=[{"role": "user", "content": "same"}], system="s")
    for _ in range(5):
        store.add(Trace(req, LLMResponse(text='{"a": 1}'), "p", "m"))
    store.add(
        Trace(
            LLMRequest(messages=[{"role": "user", "content": "tool"}], system="s"),
            LLMResponse(tool_calls=[{"name": "t", "arguments": {}}]),
            "p",
            "m",
        )
    )
    cases = build_suite(store.all())
    assert len(cases) == 2
    assert len({c.signature for c in cases}) == 2


def test_instrument_openai_records_without_breaking(tmp_path):
    msg = SimpleNamespace(content='{"ok": true}', tool_calls=None)
    result = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: result)))
    store = TraceStore(str(tmp_path / "t.db"))
    instrument_openai(client, store)
    out = client.chat.completions.create(
        model="gpt-x",
        temperature=0,
        messages=[{"role": "system", "content": "be terse"}, {"role": "user", "content": "hello"}],
    )
    assert out is result
    [t] = store.all()
    assert t.request.system == "be terse"
    assert t.request.messages == [{"role": "user", "content": "hello"}]
    assert t.model == "gpt-x" and t.request.params == {"temperature": 0}


def test_end_to_end_demo_repairs_to_full_pass(tmp_path):
    out = run_demo(tmp_path, quiet=True)
    assert out["before"]["pass_rate"] < 0.2
    assert out["after"]["pass_rate"] == 1.0
    r = out["repair"].repairs[0]
    assert r.holdout_after == 1.0
    assert (tmp_path / "migration_report.md").read_text().startswith("# Migration report")
    json.loads((tmp_path / "repairs.json").read_text())
