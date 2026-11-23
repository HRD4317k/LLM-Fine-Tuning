import json
import random

import pytest

pytest.importorskip("datasets")

from src.adapters.base_parser import UnifiedToolCallingRecord  # noqa: E402
from src.core.config import load_config  # noqa: E402
from src.pipelines.data_pipeline import DataPipeline, make_no_tool_negative  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "\n".join(f"<|{m['role']}|>{m['content']}" for m in messages)


TOOLS = [
    {"name": "get_weather", "parameters": {"city": {"type": "str"}, "units": {"type": "str, optional"}}},
    {"name": "convert", "parameters": {"amount": {"type": "float"}, "to": {"type": "str"}}},
]
ANSWERS = [{"name": "get_weather", "arguments": {"city": "Pune"}},
           {"name": "convert", "arguments": {"amount": 5, "to": "EUR"}}]


def _record(answers=ANSWERS):
    return UnifiedToolCallingRecord(query="Weather in Pune and 5 USD to EUR", tools=TOOLS,
                                    answers=json.dumps(answers))


def test_trace_mentions_every_parallel_call_and_required_params():
    text = DataPipeline(load_config("base"), None, FakeTokenizer()).generate_prompt_for_record(_record())
    assert "2 calls are needed: 'get_weather', 'convert' match" in text
    assert "get_weather: required ['city']" in text
    assert "convert: required ['amount', 'to']" in text
    payload = text.split("<tool_call>\n")[1].split("\n</tool_call>")[0]
    assert json.loads(payload) == ANSWERS


def test_no_tool_negative_removes_target_tools():
    neg = make_no_tool_negative(_record([ANSWERS[0]]), rate=1.0, rng=random.Random(0))
    assert neg.answers == "[]"
    assert [t["name"] for t in neg.tools] == ["convert"]
    text = DataPipeline(load_config("base"), None, FakeTokenizer()).generate_prompt_for_record(neg)
    assistant = text.split("<|assistant|>")[1]
    assert "[ABORT: No matching tool]" in assistant and "<tool_call>" not in assistant


def test_negative_rate_zero_is_identity():
    rec = _record()
    assert make_no_tool_negative(rec, 0.0, random.Random(0)) is rec


def test_config_defaults_protect_against_leakage():
    cfg = load_config("base")
    assert cfg.data.test_size >= 1000
    assert cfg.training.train_on_responses_only is True
    assert load_config("smoke_test").data.test_size == 20
