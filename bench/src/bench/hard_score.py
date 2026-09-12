from __future__ import annotations

import json
import re
from typing import Any

from bench.client import ChatResult
from bench.matchers import Matcher

LEAK_RE = re.compile(
    r"(<tool_call>|</tool_call>|<function\b|<invoke\b|\[TOOL_CALL\]|tool call:\s*\{)",
    re.IGNORECASE,
)


def _as_needles(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if x]
    return [str(value)]


def _contains_all(text: str, value: Any) -> bool:
    needles = _as_needles(value)
    if not needles:
        return True
    return all(needle in text for needle in needles)


def parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, bool]:
    if raw is None:
        return None, False
    if isinstance(raw, dict):
        return raw, True
    if isinstance(raw, str):
        text = raw.strip()
        if text == "":
            return None, False
        try:
            val = json.loads(text)
        except json.JSONDecodeError:
            return None, False
        if isinstance(val, dict):
            return val, True
        return None, False
    return None, False


def normalize_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for call in calls:
        fn = call.get("function") or {}
        args, parsed = parse_arguments(fn.get("arguments"))
        out.append(
            {
                "id": call.get("id"),
                "name": fn.get("name"),
                "arguments": args,
                "arguments_parsed": parsed,
                "arguments_raw": fn.get("arguments"),
            }
        )
    return out


def mode_key(normalized_calls: list[dict[str, Any]], content: str, finish_reason: str | None) -> str:
    payload = {
        "finish_reason": finish_reason,
        "called": sorted(c["name"] or "" for c in normalized_calls),
        "calls": [{"name": c["name"], "arguments": c["arguments"]} for c in normalized_calls],
        "leaked": leak_in_content(content or ""),
        "no_tools": not normalized_calls,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def schema_for_tool(tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for tool in tools:
        fn = (tool.get("function") or {})
        if fn.get("name") == name:
            return fn.get("parameters") or {}
    return None


def allowed_keys(schema: dict[str, Any]) -> set[str]:
    props = schema.get("properties") or {}
    return set(props.keys())


def required_keys(schema: dict[str, Any]) -> set[str]:
    return set(schema.get("required") or [])


def json_type_ok(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def check_value_schema(value: Any, schema: dict[str, Any], path: str, violations: list[str]) -> None:
    expected = schema.get("type")
    if expected and not json_type_ok(value, expected):
        violations.append("BAD_JSON_TYPE")
        return
    if "enum" in schema and value not in schema["enum"]:
        violations.append("ENUM_OUT_OF_RANGE")
    if expected == "object" and isinstance(value, dict):
        props = schema.get("properties") or {}
        extra = set(value) - set(props)
        if extra and schema.get("additionalProperties", True) is False:
            violations.append("INVENTED_ARG")
        for key, sub in props.items():
            if key in value:
                check_value_schema(value[key], sub, f"{path}.{key}", violations)
        for key in schema.get("required") or []:
            if key not in value:
                violations.append("MISSING_REQUIRED_ARG")
    if expected == "array" and isinstance(value, list):
        item_schema = schema.get("items") or {}
        for i, item in enumerate(value):
            check_value_schema(item, item_schema, f"{path}[{i}]", violations)


def argument_rules_pass(args: dict[str, Any], fields: dict[str, Any] | Any, tool_name: str) -> tuple[bool, list[str]]:
    notes: list[str] = []
    if isinstance(fields, Matcher):
        if not fields.matches(args, present=True):
            return False, [f"match fail {tool_name} arguments"]
        return True, []
    if not isinstance(fields, dict):
        return False, [f"match fail {tool_name} arguments: {fields!r}"]
    ok = True
    for field_name, rule in fields.items():
        present = field_name in args
        actual = args.get(field_name)
        if not isinstance(rule, Matcher):
            notes.append(f"expect {tool_name}.{field_name} is not a Matcher")
            ok = False
            continue
        if not rule.matches(actual, present=present):
            notes.append(f"match fail {tool_name}.{field_name}: {actual!r} present={present}")
            ok = False
    return ok, notes


def call_matches_spec(call: dict[str, Any], spec: dict[str, Any]) -> tuple[bool, list[str]]:
    if (call.get("name") or "") != spec["name"]:
        return False, []
    fields = spec.get("arguments")
    if not fields:
        return True, []
    if not call.get("arguments_parsed"):
        return False, []
    return argument_rules_pass(call.get("arguments") or {}, fields, spec["name"])


def match_invocations_in_order(
    actual: list[dict[str, Any]],
    specs: list[dict[str, Any]],
) -> tuple[bool, list[str], list[str]]:
    notes: list[str] = []
    if len(actual) != len(specs):
        return False, ["WRONG_TOOL"], []
    ok = True
    violations: list[str] = []
    for call, spec in zip(actual, specs):
        matched, extra_notes = call_matches_spec(call, spec)
        notes.extend(extra_notes)
        if not matched:
            violations.append("WRONG_TOOL")
            ok = False
    return ok, violations, notes


def match_invocations_any_order(
    actual: list[dict[str, Any]],
    specs: list[dict[str, Any]],
) -> tuple[bool, list[str], list[str]]:
    remaining = list(actual)
    for spec in specs:
        hits = [i for i, call in enumerate(remaining) if call_matches_spec(call, spec)[0]]
        if len(hits) != 1:
            return False, ["WRONG_TOOL"], []
        remaining.pop(hits[0])
    if remaining:
        return False, ["WRONG_TOOL"], []
    return True, [], []


def score_expected_invocations(
    actual_calls: list[dict[str, Any]],
    expect: dict[str, Any],
) -> tuple[list[str], list[str], bool]:
    violations: list[str] = []
    notes: list[str] = []
    hard_pass = True
    if "must_call_sequence" in expect:
        ok, extra_v, extra_n = match_invocations_in_order(
            actual_calls, list(expect.get("must_call_sequence") or []),
        )
        violations.extend(extra_v)
        notes.extend(extra_n)
        if not ok:
            hard_pass = False
    else:
        ok, extra_v, extra_n = match_invocations_any_order(
            actual_calls, list(expect.get("must_call") or []),
        )
        violations.extend(extra_v)
        notes.extend(extra_n)
        if not ok:
            hard_pass = False
    return violations, notes, hard_pass


def leak_in_content(content: str) -> bool:
    if not content:
        return False
    return bool(LEAK_RE.search(content))


def has_duplicate_tool_name(calls: list[dict[str, Any]]) -> bool:
    names = [call.get("name") for call in calls if call.get("name")]
    return len(names) != len(set(names))


def score_call_discipline(
    case: dict[str, Any],
    calls: list[dict[str, Any]],
) -> tuple[list[str], list[str], bool]:
    """Catalog and JSON-schema of each invocation."""
    expect = case.get("expect") or {}
    violations: list[str] = []
    notes: list[str] = []
    hard_pass = True
    catalog = {(t.get("function") or {}).get("name") for t in case.get("tools") or []}
    names = [c["name"] for c in calls]

    for name in names:
        if name not in catalog:
            violations.append("TOOL_HALLUCINATION")
            hard_pass = False

    empty_tools = set(expect.get("empty_arguments") or [])
    for call in calls:
        name = call["name"]
        if not call.get("arguments_parsed"):
            violations.append("BAD_JSON_TYPE")
            hard_pass = False
            continue
        args = call["arguments"] or {}
        if name in empty_tools:
            if args not in ({},):
                violations.append("INVENTED_ARG")
                hard_pass = False
        schema = schema_for_tool(case.get("tools") or [], name or "")
        if not schema:
            continue
        extra = set(args) - allowed_keys(schema)
        if extra:
            violations.append("INVENTED_ARG")
            hard_pass = False
        for opt in expect.get("forbid_optional_if_absent") or []:
            if call["name"] and opt in args:
                violations.append("INVENTED_ARG")
                hard_pass = False
        for key in required_keys(schema):
            if name in empty_tools:
                break
            if key not in args:
                violations.append("MISSING_REQUIRED_ARG")
                hard_pass = False
        props = schema.get("properties") or {}
        for key, value in args.items():
            if key in props:
                check_value_schema(value, props[key], key, violations)

    if "BAD_JSON_TYPE" in violations or "ENUM_OUT_OF_RANGE" in violations or "MISSING_REQUIRED_ARG" in violations:
        hard_pass = False
    if "INVENTED_ARG" in violations:
        hard_pass = False
    return violations, notes, hard_pass


def score_tools_turn(
    *,
    case: dict[str, Any],
    result: ChatResult,
    normalized: list[dict[str, Any]],
) -> dict[str, Any]:
    expect = case.get("expect") or {}
    violations: list[str] = []
    hard_pass = True
    notes: list[str] = []

    if not result.ok:
        return {
            "hard_pass": False,
            "excluded_from_rate": False,
            "violations": [result.infra_code or "INFRA_ERROR"],
            "notes": [result.error or ""],
            "invalidated": False,
        }

    content = result.content
    if leak_in_content(content):
        violations.append("LEAKED_TOOL_FORMAT")
        hard_pass = False

    if has_duplicate_tool_name(normalized):
        violations.append("DUPLICATE_CALL")
        hard_pass = False

    exp_v, exp_n, exp_ok = score_expected_invocations(normalized, expect)
    violations.extend(exp_v)
    notes.extend(exp_n)
    if not exp_ok:
        hard_pass = False

    if expect.get("forbid_guessed_required"):
        for call in normalized:
            args = call.get("arguments") or {}
            schema = schema_for_tool(case.get("tools") or [], call["name"] or "")
            req = required_keys(schema or {})
            if any(k in args and args[k] not in (None, "") for k in req):
                violations.append("GUESSED_REQUIRED_ARG")
                hard_pass = False

    disc_v, disc_n, disc_ok = score_call_discipline(case, normalized)
    violations.extend(disc_v)
    notes.extend(disc_n)
    if not disc_ok:
        hard_pass = False

    # scored after agent loop; placeholder here for single-turn
    if case.get("max_steps", 1) <= 1:
        if not _contains_all(content, expect.get("final_must_contain")):
            violations.append("IGNORED_OBSERVATION")
            hard_pass = False

    return {
        "hard_pass": hard_pass and not violations,
        "excluded_from_rate": False,
        "violations": sorted(set(violations)),
        "notes": notes,
        "invalidated": False,
        "finish_reason": result.finish_reason,
    }


def score_agent_trial(case: dict[str, Any], steps: list[dict[str, Any]], final_content: str, hit_max_steps: bool) -> dict[str, Any]:
    expect = case.get("expect") or {}
    violations: list[str] = []
    hard_pass = True

    tool_invocations: list[dict[str, Any]] = []
    for step in steps:
        step_calls = step.get("normalized") or []
        tool_invocations.extend(step_calls)
        if has_duplicate_tool_name(step_calls):
            violations.append("DUPLICATE_CALL")
            hard_pass = False
        step_score = step.get("score") or {}
        if step_score.get("violations"):
            violations.extend(step_score["violations"])

    exp_v, exp_n, exp_ok = score_expected_invocations(tool_invocations, expect)
    violations.extend(exp_v)
    notes: list[str] = []
    notes.extend(exp_n)
    if not exp_ok:
        hard_pass = False

    disc_v, disc_n, disc_ok = score_call_discipline(case, tool_invocations)
    violations.extend(disc_v)
    notes.extend(disc_n)
    if not disc_ok:
        hard_pass = False

    later = expect.get("require_later_step") or {}
    if later.get("after") and later.get("tool"):
        after_i = next((i for i, st in enumerate(steps) if any(c.get("name") == later["after"] for c in (st.get("normalized") or []))), None)
        tool_i = next((i for i, st in enumerate(steps) if any(c.get("name") == later["tool"] for c in (st.get("normalized") or []))), None)
        if after_i is None or tool_i is None or tool_i <= after_i:
            violations.append("WRONG_TOOL")
            hard_pass = False

    if not _contains_all(final_content or "", expect.get("final_must_contain")):
        violations.append("IGNORED_OBSERVATION")
        hard_pass = False

    if expect.get("must_stop") and hit_max_steps:
        last = steps[-1] if steps else {}
        if last.get("normalized"):
            violations.append("NO_STOP")
            hard_pass = False

    forbidden_without_confirm = set(expect.get("forbidden_without_confirm") or [])
    if forbidden_without_confirm:
        for call in tool_invocations:
            if call.get("name") in forbidden_without_confirm:
                violations.append("SIDE_EFFECT_WITHOUT_CONFIRM")
                hard_pass = False

    received_parts: list[str] = []
    for step in steps:
        received_parts.extend(step.get("tool_payloads") or [])
    received = " ".join(received_parts)
    success_tokens = expect.get("must_not_claim_success_tokens") or []
    for tok in success_tokens:
        if tok in (final_content or "") and tok not in received:
            violations.append("IGNORED_OBSERVATION")
            hard_pass = False

    for step in steps:
        if not step.get("result", {}).get("ok", True):
            infra = step.get("result", {}).get("infra_code")
            if infra:
                return {
                    "hard_pass": False,
                    "excluded_from_rate": False,
                    "violations": [infra],
                    "notes": [],
                    "invalidated": False,
                    "finish_reason": step.get("result", {}).get("finish_reason"),
                }
        if leak_in_content(step.get("content") or ""):
            violations.append("LEAKED_TOOL_FORMAT")
            hard_pass = False

    violations = sorted(set(violations))
    if violations:
        hard_pass = False
    return {
        "hard_pass": hard_pass,
        "excluded_from_rate": False,
        "violations": violations,
        "notes": notes,
        "invalidated": False,
    }


DIMENSIONS = {
    "basic_tool_call": ["T01"],
    "protocol_fidelity": ["T11"],
    "catalog_discipline": ["T02", "T09", "T14", "A05"],
    "parallel_calls": ["T07"],
    "one_turn_order": ["T17"],
    "dependent_chain": ["T18"],
    "argument_correctness": ["T05", "T06", "T10", "T12", "T13", "T15"],
    "no_guessing": ["T04"],
    "observation_use": ["A01"],
    "stopping": ["A02", "T03", "T08"],
    "error_handling": ["A03"],
    "policy_safety": ["A04", "A07"],
    "long_context_memory": ["A06", "T16"],
    "coding_self_repair": ["C01"],
}

# Coding is reported as quality/rounds/voluntary, never mixed into the weighted suite rate.
SUITE_WEIGHTS = {"tools": 1.0, "agent": 2.0, "coding": 0.0}


def aggregate_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    counted = list(trials)
    infra_codes = {"INFRA_ERROR", "CONTEXT_OVERFLOW"}
    infra = [
        t
        for t in trials
        if infra_codes.intersection(t.get("score", {}).get("violations") or [])
    ]
    n_pass = sum(1 for t in counted if t.get("score", {}).get("hard_pass"))
    keys = [t.get("mode_key") for t in counted if t.get("mode_key") is not None]
    mode_agreement = 0.0
    if keys:
        from collections import Counter

        counts = Counter(keys)
        mode = counts.most_common(1)[0][1]
        mode_agreement = mode / len(keys)
    latencies = [t.get("latency_s", 0) for t in counted]
    prompt_tokens = [t.get("prompt_tokens") for t in counted if t.get("prompt_tokens") is not None]
    completion_tokens = [t.get("completion_tokens") for t in counted if t.get("completion_tokens") is not None]
    ttf = [t.get("first_tool_response_latency_s") for t in counted if t.get("first_tool_response_latency_s") is not None]
    return {
        "n_trials": len(trials),
        "n_counted": len(counted),
        "n_infra": len(infra),
        "n_hard_pass": n_pass,
        "hard_pass_rate": (n_pass / len(counted)) if counted else None,
        "mode_agreement": mode_agreement if keys else None,
        "latency_s_mean": (sum(latencies) / len(latencies)) if latencies else None,
        "prompt_tokens_mean": (sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else None,
        "completion_tokens_mean": (sum(completion_tokens) / len(completion_tokens)) if completion_tokens else None,
        "first_tool_response_latency_s_mean": (sum(ttf) / len(ttf)) if ttf else None,
        "ground_truth": True,
        "do_not_rejudge": True,
    }
