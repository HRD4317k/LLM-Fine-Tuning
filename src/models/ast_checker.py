"""
AST-style function-call checker (in the spirit of the Berkeley Function-Calling
Leaderboard "AST" category).

A tool call is parsed into a tree {name, arguments{param: value}} and checked
against both the tool *schema* and the *ground-truth* call:

    1. syntax              the <tool_call> block must parse into dict(s)
    2. tool name           must exist in the provided tool list (else hallucinated tool)
                           and match the expected tool
    3. parameters          no parameter outside the schema / ground truth (hallucinated param)
                           every required parameter present (missing param)
    4. types               each value must match the schema type (str/int/float/bool/list/dict)
    5. values              each value must equal ground truth (numbers with tolerance,
                           strings case/whitespace-normalised, lists element-wise)

Parallel calls are matched order-insensitively: every expected call must be
matched by exactly one predicted call and vice versa.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

ERROR_ORDER = [
    "error_syntax",
    "error_missing_tool",
    "error_hallucinated_tool",
    "error_wrong_tool",
    "error_hallucinated_param",
    "error_missing_param",
    "error_type_mismatch",
    "error_value_mismatch",
    "error_call_count",
]

_TYPE_ALIASES = {
    "str": str, "string": str,
    "int": int, "integer": int,
    "float": float, "number": float,
    "bool": bool, "boolean": bool,
    "list": list, "array": list,
    "dict": dict, "object": dict,
}


@dataclass
class CallCheck:
    passed: bool
    error: Optional[str] = None
    detail: str = ""


@dataclass
class SampleResult:
    outcome: str                      # "TP" | "TN" | "FP" | "FN"
    error: Optional[str] = None
    per_call: List[CallCheck] = field(default_factory=list)


# ── Schema helpers ─────────────────────────────────────────────────────────────

def _tool_name(tool: Dict[str, Any]) -> Optional[str]:
    if "function" in tool and isinstance(tool["function"], dict):
        return tool["function"].get("name")
    return tool.get("name")


def _tool_params(tool: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Return (param_specs, required_names) for xLAM-style or JSON-schema-style tools."""
    spec = tool.get("function", tool)
    params = spec.get("parameters") or {}
    if isinstance(params, dict) and "properties" in params:          # JSON-schema style
        return params.get("properties") or {}, list(params.get("required") or [])
    required = spec.get("required")
    if required is None and isinstance(params, dict):                  # xLAM style
        required = [
            name for name, p in params.items()
            if isinstance(p, dict)
            and "default" not in p
            and "optional" not in str(p.get("type", "")).lower()
        ]
    return (params if isinstance(params, dict) else {}), list(required or [])


def required_params(tools: List[Dict[str, Any]], tool_name: str) -> List[str]:
    for t in tools:
        if isinstance(t, dict) and _tool_name(t) == tool_name:
            return _tool_params(t)[1]
    return []


def _expected_python_type(type_str: Any) -> Optional[type]:
    if not isinstance(type_str, str):
        return None
    base = re.split(r"[\s,\[(|]", type_str.strip().lower(), maxsplit=1)[0]
    return _TYPE_ALIASES.get(base)


def _type_ok(value: Any, expected: Optional[type]) -> bool:
    if expected is None or value is None:
        return True
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is bool:
        return isinstance(value, bool)
    return isinstance(value, expected)


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_values_equal(a[k], b[k]) for k in a)
    return a == b


def _as_args(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if isinstance(value, dict) else {}


# ── Single call ────────────────────────────────────────────────────────────────

def check_call(pred: Any, gold: Dict[str, Any], tools: List[Dict[str, Any]]) -> CallCheck:
    if not isinstance(pred, dict) or "name" not in pred:
        return CallCheck(False, "error_syntax", "call is not an object with a name")

    available = {_tool_name(t): t for t in tools if isinstance(t, dict)}
    name = str(pred.get("name", "")).strip()
    if name not in available:
        return CallCheck(False, "error_hallucinated_tool", f"unknown tool '{name}'")
    if name != str(gold.get("name", "")).strip():
        return CallCheck(False, "error_wrong_tool", f"expected '{gold.get('name')}', got '{name}'")

    specs, required = _tool_params(available[name])
    p_args, g_args = _as_args(pred.get("arguments", {})), _as_args(gold.get("arguments", {}))

    allowed = set(specs) | set(g_args)
    extra = sorted(set(p_args) - allowed)
    if extra:
        return CallCheck(False, "error_hallucinated_param", f"unexpected {extra}")
    missing = sorted((set(required) | set(g_args)) - set(p_args))
    if missing:
        return CallCheck(False, "error_missing_param", f"missing {missing}")

    for k, v in p_args.items():
        expected = _expected_python_type((specs.get(k) or {}).get("type")) if isinstance(specs.get(k), dict) else None
        if expected is None and k in g_args and g_args[k] is not None:
            expected = float if isinstance(g_args[k], (int, float)) and not isinstance(g_args[k], bool) else type(g_args[k])
        if not _type_ok(v, expected):
            return CallCheck(False, "error_type_mismatch", f"{k}: {type(v).__name__} != {expected.__name__}")

    for k, v in g_args.items():
        if not _values_equal(p_args.get(k), v):
            return CallCheck(False, "error_value_mismatch", f"{k}: {p_args.get(k)!r} != {v!r}")
    return CallCheck(True)


# ── Whole sample (handles parallel calls and aborts) ──────────────────────────

def check_sample(
    predicted_calls: Optional[List[Any]],
    expected_calls: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    is_abort: bool,
    has_tool_call_block: bool,
    is_valid_json: bool,
) -> SampleResult:
    if not expected_calls:                                   # model should refuse
        if is_abort or not predicted_calls:
            return SampleResult("TN")
        return SampleResult("FP", "error_hallucinated_tool")

    if has_tool_call_block and not is_valid_json:
        return SampleResult("FN", "error_syntax")
    if is_abort or not predicted_calls:
        return SampleResult("FN", "error_missing_tool")
    if len(predicted_calls) != len(expected_calls):
        return SampleResult("FN", "error_call_count")

    remaining = list(range(len(predicted_calls)))
    checks: List[CallCheck] = []
    for gold in expected_calls:
        best: Optional[Tuple[int, CallCheck]] = None
        for i in remaining:
            c = check_call(predicted_calls[i], gold, tools)
            if c.passed:
                best = (i, c)
                break
            if best is None or ERROR_ORDER.index(c.error) > ERROR_ORDER.index(best[1].error):
                best = (i, c)                                # keep the "closest" failure
        assert best is not None
        checks.append(best[1])
        if best[1].passed:
            remaining.remove(best[0])

    failures = [c for c in checks if not c.passed]
    if not failures:
        return SampleResult("TP", None, checks)
    worst = min(failures, key=lambda c: ERROR_ORDER.index(c.error))
    return SampleResult("FN", worst.error, checks)


def summarise(results: List[SampleResult], valid_json_flags: List[Optional[bool]]) -> Dict[str, Any]:
    """Aggregate per-sample results into the report used in the README / CV.

    accuracy         (TP + TN) / total: call fully correct, or correct refusal
    valid_json_rate  share of responses containing a <tool_call> block whose JSON parses
    """
    n = len(results)
    counts = {k: sum(r.outcome == k for r in results) for k in ("TP", "TN", "FP", "FN")}
    errors = {e: sum(r.error == e for r in results) for e in ERROR_ORDER}
    blocks = [f for f in valid_json_flags if f is not None]
    return {
        "total_samples": n,
        "accuracy": (counts["TP"] + counts["TN"]) / n if n else 0.0,
        "valid_json_rate": sum(blocks) / len(blocks) if blocks else 0.0,
        "confusion_matrix": {
            "true_positives": counts["TP"], "true_negatives": counts["TN"],
            "false_positives": counts["FP"], "false_negatives": counts["FN"],
        },
        "error_distribution": {e.replace("error_", ""): c for e, c in errors.items()},
    }
