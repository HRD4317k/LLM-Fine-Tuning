import pytest

from src.models.ast_checker import check_call, check_sample, required_params, summarise
from src.models.validator import Validator

TOOLS = [
    {"name": "get_weather", "parameters": {
        "city": {"type": "str", "description": "City"},
        "units": {"type": "str, optional", "default": "metric"},
        "days": {"type": "int", "description": "Days"},
    }},
    {"name": "convert", "parameters": {
        "amount": {"type": "float"}, "to": {"type": "str"},
    }},
]
GOLD = {"name": "get_weather", "arguments": {"city": "Mumbai", "days": 3}}


def test_required_params_from_xlam_schema():
    assert required_params(TOOLS, "get_weather") == ["city", "days"]


@pytest.mark.parametrize("pred, error", [
    ({"name": "get_weather", "arguments": {"city": "mumbai ", "days": 3}}, None),
    ({"name": "get_news", "arguments": {}}, "error_hallucinated_tool"),
    ({"name": "convert", "arguments": {"amount": 1, "to": "INR"}}, "error_wrong_tool"),
    ({"name": "get_weather", "arguments": {"city": "Mumbai", "days": 3, "lang": "en"}}, "error_hallucinated_param"),
    ({"name": "get_weather", "arguments": {"city": "Mumbai"}}, "error_missing_param"),
    ({"name": "get_weather", "arguments": {"city": "Mumbai", "days": "3"}}, "error_type_mismatch"),
    ({"name": "get_weather", "arguments": {"city": "Pune", "days": 3}}, "error_value_mismatch"),
    ("not a dict", "error_syntax"),
])
def test_check_call_categories(pred, error):
    result = check_call(pred, GOLD, TOOLS)
    assert result.passed is (error is None)
    assert result.error == error


def test_int_accepted_for_float_param():
    gold = {"name": "convert", "arguments": {"amount": 10.0, "to": "USD"}}
    assert check_call({"name": "convert", "arguments": {"amount": 10, "to": "usd"}}, gold, TOOLS).passed


def test_parallel_calls_are_order_insensitive():
    gold = [GOLD, {"name": "convert", "arguments": {"amount": 5.0, "to": "EUR"}}]
    pred = list(reversed(gold))
    assert check_sample(pred, gold, TOOLS, False, True, True).outcome == "TP"


def test_second_parallel_call_is_graded_too():
    gold = [GOLD, {"name": "convert", "arguments": {"amount": 5.0, "to": "EUR"}}]
    pred = [GOLD, {"name": "convert", "arguments": {"amount": 6.0, "to": "EUR"}}]
    r = check_sample(pred, gold, TOOLS, False, True, True)
    assert (r.outcome, r.error) == ("FN", "error_value_mismatch")


def test_refusals_and_syntax():
    assert check_sample(None, [], TOOLS, True, False, False).outcome == "TN"
    assert check_sample([GOLD], [], TOOLS, False, True, True).outcome == "FP"
    assert check_sample(None, [GOLD], TOOLS, False, True, False).error == "error_syntax"
    assert check_sample(None, [GOLD], TOOLS, True, False, False).error == "error_missing_tool"
    assert check_sample([GOLD, GOLD], [GOLD], TOOLS, False, True, True).error == "error_call_count"


def test_summary_metrics():
    results = [check_sample([GOLD], [GOLD], TOOLS, False, True, True),
               check_sample(None, [], TOOLS, True, False, False),
               check_sample(None, [GOLD], TOOLS, False, True, False),
               check_sample([{"name": "get_weather", "arguments": {"city": "X", "days": 1}}], [GOLD], TOOLS, False, True, True)]
    s = summarise(results, [True, None, False, True])
    assert s["accuracy"] == 0.5
    assert s["valid_json_rate"] == pytest.approx(2 / 3)


def test_validator_parses_thought_and_call():
    text = '<thought>1. ...</thought>\n<tool_call>\n```json\n[{"name": "get_weather", "arguments": {"city": "Pune", "days": 1}}]\n```\n</tool_call>'
    out = Validator.parse_generated_text(text)
    assert out["has_thought"] and out["is_valid_json"]
    assert out["parsed_json"][0]["name"] == "get_weather"
