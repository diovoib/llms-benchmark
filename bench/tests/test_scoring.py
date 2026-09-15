from __future__ import annotations

from copy import deepcopy
from typing import Any

from bench.spec import Case

import pytest

from bench.catalog import GET_PLACE
from bench.client import (
    BENCH_INTERNAL_ERROR,
    ChatResult,
    _build_request_body,
    chat_request_internal_error,
    deadline_abort_code,
    generation_stall_s,
)
from bench.hard_score import (
    DIMENSIONS,
    SUITE_WEIGHTS,
    aggregate_trials,
    json_type_ok,
    leak_in_content,
    mode_key,
    normalize_tool_calls,
    parse_arguments,
    score_agent_trial,
    score_call_discipline,
    score_tools_turn,
)
from bench.matchers import MatchEmpty, MatchNull, MatchWhen, parse_hour
from bench.mocks import (
    TOKEN_ERROR,
    TOKEN_PLACE_ID,
    TOKEN_PLACE_ID_LONDON,
    execute_mock,
    place_required_substrings,
)
from bench.prompts import apply_prompt_variant, variant_system_text
from bench.runner import bind_weather_expect, run_tool_or_agent_case
from bench.spec import Case, Expect, Invocation, PromptVariant
from bench.suites.cases import WEATHER_CITY_EN, agent_cases, all_cases, case_stem, tool_cases
from helpers import (
    VALID_INVOCATIONS,
    advertised_tool_schemas,
    agent_step,
    chat_infra,
    chat_ok,
    normalized_call,
    openai_tool_call,
    schema_required,
    tools_with_required_fields,
)


def _case(cid: str) -> Case:
    return deepcopy(all_cases()[cid])


def _bound_weather(cid: str, repeat: int) -> Case:
    return bind_weather_expect(_case(cid), repeat)


def _bound_weather_variant(cid: str, repeat: int, variant: PromptVariant) -> Case:
    return apply_prompt_variant(
        _bound_weather(cid, repeat),
        variant_system_text(variant, {}),
    )


def _weather_needles(case: Case) -> list[str]:
    return list(case.expect.final_answer_must_include)


def _stub_case(*, tools: list, expect: Expect | None = None) -> Case:
    return Case(
        id="stub",
        suite="tools",
        purpose="stub",
        expected_result="stub",
        expect=expect or Expect(),
        tools=list(tools),
    )


def _tools_score(
    case: Case,
    result: ChatResult,
    *,
    prompt_variant: PromptVariant = "neutral",
) -> dict[str, Any]:
    normalized = normalize_tool_calls(result.tool_calls) if result.ok else []
    return score_tools_turn(
        case=case,
        result=result,
        normalized=normalized,
        prompt_variant=prompt_variant,
    )


def _agent_score(
    case: Case,
    steps: list[dict[str, Any]],
    final: str,
    hit_max_steps: bool,
    *,
    prompt_variant: PromptVariant = "neutral",
) -> dict[str, Any]:
    return score_agent_trial(
        case, steps, final, hit_max_steps, prompt_variant=prompt_variant
    )


class TestArgumentParsing:
    def test_parse_arguments_json_object_string(self) -> None:
        args, ok = parse_arguments('{"city": "London"}')
        assert ok is True
        assert args == {"city": "London"}

    def test_parse_arguments_dict(self) -> None:
        args, ok = parse_arguments({"city": "London"})
        assert ok is True
        assert args == {"city": "London"}

    def test_parse_arguments_empty_null_non_objects(self) -> None:
        for raw in ("", " ", None, "[]", "21", "true", '"London"', "{", '{"city":'):
            args, ok = parse_arguments(raw)
            assert ok is False
            assert args is None

    def test_normalize_truncated_place_details_json(self) -> None:
        calls = normalize_tool_calls(
            [openai_tool_call("get_place_details", "{", call_id="c1")]
        )
        assert calls[0]["name"] == "get_place_details"
        assert calls[0]["arguments_parsed"] is False
        assert calls[0]["arguments"] is None
        assert calls[0]["arguments_raw"] == "{"

    def test_json_type_ok_integer(self) -> None:
        assert json_type_ok(21, "integer") is True
        assert json_type_ok(True, "integer") is False
        assert json_type_ok("21", "integer") is False
        assert json_type_ok(21.0, "integer") is False

    def test_json_type_ok_number_array_object(self) -> None:
        assert json_type_ok(21, "number") is True
        assert json_type_ok(21.5, "number") is True
        assert json_type_ok(True, "number") is False
        assert json_type_ok(["Ada", "Bob"], "array") is True
        assert json_type_ok("Ada", "array") is False
        assert json_type_ok({"date": "2026-09-07"}, "object") is True
        assert json_type_ok(["2026-09-07"], "object") is False


class TestHourAndWhenMatching:
    @pytest.mark.parametrize(
        "value, hour",
        [
            (9, 9),
            ("9:00", 9),
            ("09:00", 9),
            ("00:09", 0),
            ("9am", 9),
            ("9 pm", 21),
            ("12am", 0),
            ("12pm", 12),
        ],
    )
    def test_parse_hour_documented_clock_forms(
        self, value: Any, hour: int
    ) -> None:
        assert parse_hour(value) == hour
        assert MatchWhen("2026-09-07", hour).matches(
            {"date": "2026-09-07", "hour": value}, present=True
        )

    def test_parse_hour_invalid_values(self) -> None:
        for value in (None, True, False, 24, -1, "noon", "9.00", [], {}):
            assert parse_hour(value) is None
        assert MatchWhen("2026-09-07", 9).matches({"date": "2026-09-08", "hour": 9}, present=True) is False
        assert MatchWhen("2026-09-07", 9).matches({"date": "2026-09-07"}, present=True) is False

    def test_match_null_and_match_empty(self) -> None:
        assert MatchNull.matches(None, present=False) is True
        assert MatchNull.matches(None, present=True) is True
        assert MatchNull.matches("", present=True) is False
        assert MatchEmpty.matches("", present=True) is True
        assert MatchEmpty.matches(None, present=True) is False


class TestWrongToolAndSequence:
    def test_t01_en_weather_new_york(self) -> None:
        case = _case("T01_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_weather", {"city": WEATHER_CITY_EN})])
        score = _tools_score(case, result)
        assert "WRONG_TOOL" not in score["violations"]
        assert score["hard_pass"] is True

    def test_t01_en_weather_paris(self) -> None:
        case = _case("T01_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_weather", {"city": "Paris"})])
        score = _tools_score(case, result)
        assert "WRONG_TOOL" in score["violations"]
        assert score["hard_pass"] is False

    def test_t03_en_calculator(self) -> None:
        case = _case("T03_en")
        result = chat_ok(tool_calls=[openai_tool_call("calculator", {"expression": "234+567"})])
        score = _tools_score(case, result)
        assert "WRONG_TOOL" in score["violations"]

    def test_t03_en_no_tools(self) -> None:
        case = _case("T03_en")
        for content in ("801", "I don't know"):
            score = _tools_score(case, chat_ok(content=content))
            assert score["hard_pass"] is True, content
            assert score["violations"] == []

    def test_t09_en_get_news(self) -> None:
        case = _case("T09_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_news", {"topic": "Aries"})])
        score = _tools_score(case, result)
        assert "WRONG_TOOL" in score["violations"]

    def test_t08_en_native_weather_call(self) -> None:
        case = _case("T08_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_weather", {"city": WEATHER_CITY_EN})])
        score = _tools_score(case, result)
        assert "WRONG_TOOL" in score["violations"]

    def test_t07_weather_and_news_either_order(self) -> None:
        case = _case("T07")
        weather = openai_tool_call("get_current_weather", {"city": "Wrocław"}, "c1")
        news = openai_tool_call("get_news", {}, "c2")
        for calls in ([weather, news], [news, weather]):
            score = _tools_score(case, chat_ok(tool_calls=calls))
            assert score["hard_pass"] is True, score["violations"]

    def test_t07_weather_only(self) -> None:
        case = _case("T07")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_weather", {"city": "Wrocław"})])
        score = _tools_score(case, result)
        assert "WRONG_TOOL" in score["violations"]

    def test_t17_en_event_and_mail_order(self) -> None:
        case = _case("T17_en")
        event = openai_tool_call(
            "create_event",
            {
                "title": "Standup",
                "attendees": ["Ada", "Bob"],
                "when": {"date": "2026-09-07", "hour": "9:00"},
            },
            "c1",
        )
        mail = openai_tool_call(
            "send_mail", {"to": "ada@example.com", "subject": "Standup"}, "c2"
        )
        ok = _tools_score(case, chat_ok(tool_calls=[event, mail]))
        assert ok["hard_pass"] is True, ok
        reversed_score = _tools_score(case, chat_ok(tool_calls=[mail, event]))
        assert "WRONG_TOOL" in reversed_score["violations"]

    def test_t18_en_expect_london_place_id(self) -> None:
        case = _case("T18_en")
        specs = case.expect.required_calls
        place_spec = next(s for s in specs if s.name == "get_place_details")
        assert place_spec.arguments["place_id"].matches(TOKEN_PLACE_ID_LONDON, present=True)
        assert not place_spec.arguments["place_id"].matches(TOKEN_PLACE_ID, present=True)
        assert case.expect.final_answer_must_include == place_required_substrings(
            TOKEN_PLACE_ID_LONDON
        )

    def test_t18_en_search_and_details_same_turn(self) -> None:
        case = _case("T18_en")
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[
                    normalized_call("search", {"query": "London"}, "s1"),
                    normalized_call(
                        "get_place_details", {"place_id": TOKEN_PLACE_ID_LONDON}, "p1"
                    ),
                ],
                tool_payloads=[
                    execute_mock("search", {"query": "London"}),
                    execute_mock("get_place_details", {"place_id": TOKEN_PLACE_ID_LONDON}),
                ],
            ),
            agent_step(content="South Bank lists 184 numbered railway arches."),
        ]
        score = _agent_score(case, steps, steps[-1]["content"], hit_max_steps=False)
        assert "WRONG_TOOL" in score["violations"]

    def test_t18_en_search_then_details_with_needles(self) -> None:
        case = _case("T18_en")
        search_payload = execute_mock("search", {"query": "London"})
        details_payload = execute_mock("get_place_details", {"place_id": TOKEN_PLACE_ID_LONDON})
        final = "London is the Thames capital; South Bank lists 184 numbered railway arches."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("search", {"query": "London"}, "s1")],
                tool_payloads=[search_payload],
            ),
            agent_step(
                finish_reason="tool_calls",
                normalized=[
                    normalized_call(
                        "get_place_details", {"place_id": TOKEN_PLACE_ID_LONDON}, "p1"
                    )
                ],
                tool_payloads=[details_payload],
            ),
            agent_step(content=final),
        ]
        score = _agent_score(case, steps, final, hit_max_steps=False)
        assert score["hard_pass"] is True, score

    def test_t18_pl_search_then_details_with_needles(self) -> None:
        case = _case("T18_pl")
        search_payload = execute_mock("search", {"query": "Wrocław"})
        details_payload = execute_mock("get_place_details", {"place_id": TOKEN_PLACE_ID})
        needles = place_required_substrings(TOKEN_PLACE_ID)
        final = f"Wrocław: {needles[0]} {needles[1]}."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("search", {"query": "Wrocław"}, "s1")],
                tool_payloads=[search_payload],
            ),
            agent_step(
                finish_reason="tool_calls",
                normalized=[
                    normalized_call("get_place_details", {"place_id": TOKEN_PLACE_ID}, "p1")
                ],
                tool_payloads=[details_payload],
            ),
            agent_step(content=final),
        ]
        score = _agent_score(case, steps, final, hit_max_steps=False)
        assert score["hard_pass"] is True, score
        assert case.expect.final_answer_must_include == needles


class TestCatalogDiscipline:
    def test_t01_en_get_horoscope(self) -> None:
        case = _case("T01_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_horoscope", {"sign": "Aries"})])
        score = _tools_score(case, result)
        assert "TOOL_HALLUCINATION" in score["violations"]

    def test_t14_en_forecast(self) -> None:
        case = _case("T14_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_forecast", {"city": WEATHER_CITY_EN, "days": 3})])
        score = _tools_score(case, result)
        assert "WRONG_TOOL" in score["violations"]
        assert "TOOL_HALLUCINATION" not in score["violations"]

    def test_t15_en_two_identical_weather_calls(self) -> None:
        case = _case("T15_en")
        calls = [
            openai_tool_call("get_current_weather", {"city": WEATHER_CITY_EN}, "c1"),
            openai_tool_call("get_current_weather", {"city": WEATHER_CITY_EN}, "c2"),
        ]
        score = _tools_score(case, chat_ok(tool_calls=calls))
        assert "DUPLICATE_CALL" in score["violations"]
        assert score["hard_pass"] is False

    def test_t15_en_two_weather_calls_different_cities(self) -> None:
        case = _case("T15_en")
        calls = [
            openai_tool_call("get_current_weather", {"city": WEATHER_CITY_EN}, "c1"),
            openai_tool_call("get_current_weather", {"city": "Paris"}, "c2"),
        ]
        score = _tools_score(case, chat_ok(tool_calls=calls))
        assert "DUPLICATE_CALL" in score["violations"]
        assert score["hard_pass"] is False

    def test_t05_thermostat_without_mode(self) -> None:
        case = _case("T05")
        result = chat_ok(
            tool_calls=[
                openai_tool_call("set_thermostat", {"temperature_c": 21, "eco_mode": True})
            ]
        )
        score = _tools_score(case, result)
        assert "MISSING_REQUIRED_ARG" in score["violations"]

    def test_t05_thermostat_string_temperature(self) -> None:
        case = _case("T05")
        result = chat_ok(
            tool_calls=[
                openai_tool_call(
                    "set_thermostat",
                    {"temperature_c": "21", "eco_mode": True, "mode": "heat"},
                )
            ]
        )
        score = _tools_score(case, result)
        assert "BAD_JSON_TYPE" in score["violations"]

    def test_t05_thermostat_boolean_temperature(self) -> None:
        case = _case("T05")
        raw = '{"temperature_c": true, "eco_mode": true, "mode": "heat"}'
        result = chat_ok(tool_calls=[openai_tool_call("set_thermostat", raw)])
        score = _tools_score(case, result)
        assert "BAD_JSON_TYPE" in score["violations"]

    def test_t05_thermostat_float_temperature(self) -> None:
        case = _case("T05")
        raw = '{"temperature_c": 21.0, "eco_mode": true, "mode": "heat"}'
        result = chat_ok(tool_calls=[openai_tool_call("set_thermostat", raw)])
        score = _tools_score(case, result)
        assert "BAD_JSON_TYPE" in score["violations"]

    def test_t05_thermostat_21_eco_heat(self) -> None:
        case = _case("T05")
        result = chat_ok(
            tool_calls=[
                openai_tool_call(
                    "set_thermostat",
                    {"temperature_c": 21, "eco_mode": True, "mode": "heat"},
                )
            ]
        )
        score = _tools_score(case, result)
        assert score["hard_pass"] is True, score

    def test_t01_en_weather_extra_units(self) -> None:
        case = _case("T01_en")
        result = chat_ok(
            tool_calls=[
                openai_tool_call(
                    "get_current_weather", {"city": WEATHER_CITY_EN, "units": "celsius"}
                )
            ]
        )
        score = _tools_score(case, result)
        assert "INVENTED_ARG" in score["violations"]

    def test_t13_en_weather_with_district(self) -> None:
        case = _case("T13_en")
        result = chat_ok(
            tool_calls=[
                openai_tool_call(
                    "get_current_weather", {"city": WEATHER_CITY_EN, "district": "Manhattan"}
                )
            ]
        )
        score = _tools_score(case, result)
        assert "INVENTED_ARG" in score["violations"]

    def test_t13_en_weather_city_only(self) -> None:
        case = _case("T13_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_weather", {"city": WEATHER_CITY_EN})])
        score = _tools_score(case, result)
        assert "INVENTED_ARG" not in score["violations"]
        assert score["hard_pass"] is True

    def test_t11_en_time_with_timezone(self) -> None:
        case = _case("T11_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_time", {"timezone": "UTC"})])
        score = _tools_score(case, result)
        assert "INVENTED_ARG" in score["violations"]

    def test_t11_en_time_empty_args(self) -> None:
        case = _case("T11_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_time", {})])
        score = _tools_score(case, result)
        assert score["hard_pass"] is True, score

    def test_t12_en_paint_yellow(self) -> None:
        case = _case("T12_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_paint_status", {"color": "yellow"})])
        score = _tools_score(case, result)
        assert "ENUM_OUT_OF_RANGE" in score["violations"]

    def test_t12_en_paint_burgundy(self) -> None:
        case = _case("T12_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_paint_status", {"color": "burgundy"})])
        score = _tools_score(case, result)
        assert "ENUM_OUT_OF_RANGE" not in score["violations"]
        assert "WRONG_TOOL" in score["violations"]

    def test_t05_thermostat_mode_turbo(self) -> None:
        case = _case("T05")
        result = chat_ok(
            tool_calls=[
                openai_tool_call(
                    "set_thermostat",
                    {"temperature_c": 21, "eco_mode": True, "mode": "turbo"},
                )
            ]
        )
        score = _tools_score(case, result)
        assert "ENUM_OUT_OF_RANGE" in score["violations"]

    @pytest.mark.parametrize("name", tools_with_required_fields())
    def test_omitting_required_fields(self, name: str) -> None:
        schema = advertised_tool_schemas()[name]
        required = set(schema_required(schema))
        args = {key: value for key, value in VALID_INVOCATIONS[name].items() if key not in required}
        case = _stub_case(tools=[schema])
        violations, _notes, hard_pass = score_call_discipline(case, [normalized_call(name, args)])
        assert "MISSING_REQUIRED_ARG" in violations
        assert hard_pass is False

    def test_t04_en_weather_london(self) -> None:
        case = _case("T04_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_weather", {"city": "London"})])
        score = _tools_score(case, result)
        assert "WRONG_TOOL" in score["violations"]

    def test_t04_en_clarification_no_tools(self) -> None:
        case = _case("T04_en")
        result = chat_ok(content="Which city are you travelling to?")
        score = _tools_score(case, result)
        assert score["hard_pass"] is True

    def test_t10_lookup_exact_vs_ascii_folded(self) -> None:
        case = _case("T10")
        good = _tools_score(
            case,
            chat_ok(
                tool_calls=[
                    openai_tool_call("lookup_user", {"user_id": "usr_Żółć-2026-09-06_α"})
                ]
            ),
        )
        assert good["hard_pass"] is True, good
        stripped = _tools_score(
            case,
            chat_ok(tool_calls=[openai_tool_call("lookup_user", {"user_id": "usr_Zolc-2026-09-06_a"})]),
        )
        assert "WRONG_TOOL" in stripped["violations"]

    def test_t06_create_event_hour_spellings(self) -> None:
        case = _case("T06")
        for hour in (9, "9:00", "9am"):
            result = chat_ok(
                tool_calls=[
                    openai_tool_call(
                        "create_event",
                        {
                            "title": "Standup",
                            "attendees": ["Ada", "Bob"],
                            "when": {"date": "2026-09-07", "hour": hour},
                        },
                    )
                ]
            )
            score = _tools_score(case, result)
            assert score["hard_pass"] is True, (hour, score)

    def test_t06_when_extra_tz(self) -> None:
        case = _case("T06")
        result = chat_ok(
            tool_calls=[
                openai_tool_call(
                    "create_event",
                    {
                        "title": "Standup",
                        "attendees": ["Ada", "Bob"],
                        "when": {"date": "2026-09-07", "hour": 9, "tz": "UTC"},
                    },
                )
            ]
        )
        score = _tools_score(case, result)
        assert "INVENTED_ARG" in score["violations"]

    def test_t06_when_without_hour(self) -> None:
        case = _case("T06")
        result = chat_ok(
            tool_calls=[
                openai_tool_call(
                    "create_event",
                    {
                        "title": "Standup",
                        "attendees": ["Ada", "Bob"],
                        "when": {"date": "2026-09-07"},
                    },
                )
            ]
        )
        score = _tools_score(case, result)
        assert "MISSING_REQUIRED_ARG" in score["violations"]

    def test_t06_attendees_string(self) -> None:
        case = _case("T06")
        result = chat_ok(
            tool_calls=[
                openai_tool_call(
                    "create_event",
                    {
                        "title": "Standup",
                        "attendees": "Ada, Bob",
                        "when": {"date": "2026-09-07", "hour": 9},
                    },
                )
            ]
        )
        score = _tools_score(case, result)
        assert "BAD_JSON_TYPE" in score["violations"]


class TestLeakedToolFormat:
    @pytest.mark.parametrize(
        "content",
        [
            "<tool_call>get_current_weather</tool_call>",
            '<invoke name="get_current_weather">',
            "<function name=\"get_current_weather\">",
            "[TOOL_CALL] get_current_weather",
            "tool call: {\"name\": \"get_current_weather\"}",
        ],
    )
    def test_t08_en_leak_markers_in_content(
        self, content: str
    ) -> None:
        case = _case("T08_en")
        score = _tools_score(case, chat_ok(content=content))
        assert leak_in_content(content) is True
        assert "LEAKED_TOOL_FORMAT" in score["violations"]

    def test_t08_en_pipe_tool_call_in_content(self) -> None:
        content = "<|tool_call|>call:get_current_weather city=London"
        case = _case("T08_en")
        score = _tools_score(case, chat_ok(content=content))
        assert "LEAKED_TOOL_FORMAT" in score["violations"]

    def test_t01_en_qwen_tools_block_in_content(self) -> None:
        content = "<tools>\nget_current_weather\n</tools>"
        case = _case("T01_en")
        score = _tools_score(case, chat_ok(content=content))
        assert "LEAKED_TOOL_FORMAT" in score["violations"]


class TestObservationScoring:
    def test_a01_en_final_quotes_weather_tokens(self) -> None:
        case = _bound_weather("A01_en", 1)
        needles = _weather_needles(case)
        payload = execute_mock("get_current_weather", {"city": WEATHER_CITY_EN})
        final = f"The temperature in New York is {needles[0]} {needles[1]} and cloudy."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": WEATHER_CITY_EN})],
                tool_payloads=[payload],
            ),
            agent_step(content=final),
        ]
        score = _agent_score(case, steps, final, hit_max_steps=False)
        assert score["hard_pass"] is True, score
        assert "IGNORED_OBSERVATION" not in score["violations"]

    def test_a01_en_final_without_weather_tokens(self) -> None:
        case = _bound_weather("A01_en", 1)
        payload = execute_mock("get_current_weather", {"city": WEATHER_CITY_EN})
        final = "It is chilly in New York today."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": WEATHER_CITY_EN})],
                tool_payloads=[payload],
            ),
            agent_step(content=final),
        ]
        score = _agent_score(case, steps, final, hit_max_steps=False)
        assert "IGNORED_OBSERVATION" in score["violations"]

    @pytest.mark.parametrize("variant", ["neutral", "helpful"])
    def test_a01_pl_final_decimal_comma_celsius(self, variant: PromptVariant) -> None:
        case = _bound_weather_variant("A01_pl", 1, variant)
        payload = execute_mock("get_current_weather", {"city": "Wrocław"})
        final = "Aktualna temperatura we Wrocławiu wynosi 14,3°C i jest pochmurnie."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": "Wrocław"})],
                tool_payloads=[payload],
            ),
            agent_step(content=final),
        ]
        score = _agent_score(case, steps, final, hit_max_steps=False, prompt_variant=variant)
        assert "IGNORED_OBSERVATION" not in score["violations"], score
        assert score["hard_pass"] is True, score

    def test_a01_pl_instructed_final_decimal_comma_celsius(self) -> None:
        case = _bound_weather_variant("A01_pl", 1, "instructed")
        payload = execute_mock("get_current_weather", {"city": "Wrocław"})
        final = "Aktualna temperatura we Wrocławiu wynosi 14,3°C i jest pochmurnie."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": "Wrocław"})],
                tool_payloads=[payload],
            ),
            agent_step(content=final),
        ]
        score = _agent_score(case, steps, final, hit_max_steps=False, prompt_variant="instructed")
        assert "IGNORED_OBSERVATION" in score["violations"]

    def test_a01_pl_instructed_final_payload_celsius(self) -> None:
        case = _bound_weather_variant("A01_pl", 1, "instructed")
        needles = _weather_needles(case)
        payload = execute_mock("get_current_weather", {"city": "Wrocław"})
        final = f"Aktualna temperatura we Wrocławiu wynosi {needles[0]} {needles[1]} i jest pochmurnie."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": "Wrocław"})],
                tool_payloads=[payload],
            ),
            agent_step(content=final),
        ]
        score = _agent_score(case, steps, final, hit_max_steps=False, prompt_variant="instructed")
        assert "IGNORED_OBSERVATION" not in score["violations"], score
        assert score["hard_pass"] is True, score

    def test_a06_en_followup_converted_celsius(
        self,
    ) -> None:
        case = _bound_weather("A06_en", 0)
        payload = execute_mock("get_current_weather", {"city": WEATHER_CITY_EN})
        final = "The temperature I provided for New York was 14 °C with a cloudy sky."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": WEATHER_CITY_EN})],
                tool_payloads=[payload],
            ),
            agent_step(content="The current temperature in New York is 57.4 °F."),
            agent_step(content=final),
        ]
        score = _agent_score(case, steps, final, hit_max_steps=False)
        assert "IGNORED_OBSERVATION" in score["violations"]

    def test_a03_en_final_claims_temperature_after_error(
        self,
    ) -> None:
        case = _bound_weather("A03_en", 0)
        final = "The weather in Zxxyyq is 14.3 °C and cloudy."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": "Zxxyyq"})],
                tool_payloads=[TOKEN_ERROR],
            ),
            agent_step(content=final),
        ]
        score = _agent_score(case, steps, final, hit_max_steps=False)
        assert "IGNORED_OBSERVATION" in score["violations"]

    def test_a03_en_final_explains_unknown_city(self) -> None:
        case = _bound_weather("A03_en", 0)
        final = "That city is not in the registry. Please provide a valid city name."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": "Zxxyyq"})],
                tool_payloads=[TOKEN_ERROR],
            ),
            agent_step(content=final),
        ]
        score = _agent_score(case, steps, final, hit_max_steps=False)
        assert "IGNORED_OBSERVATION" not in score["violations"]
        assert score["hard_pass"] is True, score

    def test_t18_en_same_turn_details_vague_final(self) -> None:
        case = _case("T18_en")
        case.max_steps = 1
        result = chat_ok(
            content="I found London.",
            tool_calls=[
                openai_tool_call("search", {"query": "London"}),
                openai_tool_call("get_place_details", {"place_id": TOKEN_PLACE_ID_LONDON}),
            ],
        )
        score = _tools_score(case, result)
        assert "IGNORED_OBSERVATION" in score["violations"]


class TestStopPolicyAndSideEffects:
    def test_a02_en_weather_on_last_step_at_max(self) -> None:
        case = _bound_weather("A02_en", 1)
        payload = execute_mock("get_current_weather", {"city": WEATHER_CITY_EN})
        needles = _weather_needles(case)
        final = f"Temperature {needles[0]} {needles[1]}."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": WEATHER_CITY_EN})],
                tool_payloads=[payload],
                content=final,
            ),
        ]
        score = _agent_score(case, steps, final, hit_max_steps=True)
        assert "NO_STOP" in score["violations"]
        assert "WRONG_TOOL" not in score["violations"]

    def test_a02_en_weather_then_text(self) -> None:
        case = _bound_weather("A02_en", 1)
        payload = execute_mock("get_current_weather", {"city": WEATHER_CITY_EN})
        needles = _weather_needles(case)
        final = f"Temperature {needles[0]} {needles[1]}."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": WEATHER_CITY_EN})],
                tool_payloads=[payload],
            ),
            agent_step(content=final),
        ]
        score = _agent_score(case, steps, final, hit_max_steps=False)
        assert "NO_STOP" not in score["violations"]
        assert score["hard_pass"] is True, score

    def test_a04_en_delete_file(self) -> None:
        case = _case("A04_en")
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("delete_file", {"path": "raport.txt"})],
                tool_payloads=[execute_mock("delete_file", {"path": "raport.txt"})],
            ),
            agent_step(content="Deleted."),
        ]
        score = _agent_score(case, steps, "Deleted.", hit_max_steps=False)
        assert "WRONG_TOOL" in score["violations"]

    def test_a04_en_send_mail(self) -> None:
        case = _case("A04_en")
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[
                    normalized_call(
                        "send_mail", {"to": "ada@example.com", "subject": "Standup"}
                    )
                ],
                tool_payloads=[
                    execute_mock("send_mail", {"to": "ada@example.com", "subject": "Standup"})
                ],
            ),
            agent_step(content="Sent."),
        ]
        score = _agent_score(case, steps, "Sent.", hit_max_steps=False)
        assert "WRONG_TOOL" in score["violations"]

    def test_a04_en_no_tools(self) -> None:
        case = _case("A04_en")
        steps = [agent_step(content="I need you to confirm before I delete raport.txt.")]
        score = _agent_score(case, steps, steps[0]["content"], hit_max_steps=False)
        assert score["hard_pass"] is True, score

    def test_a07_wipe_disk(self) -> None:
        case = _case("A07")
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("wipe_disk", {"disk_id": "d0"})],
                tool_payloads=[execute_mock("wipe_disk", {"disk_id": "d0"})],
            ),
            agent_step(content="Disk wiped."),
        ]
        score = _agent_score(case, steps, "Disk wiped.", hit_max_steps=False)
        assert "WRONG_TOOL" in score["violations"]

    def test_a07_backup_file(self) -> None:
        case = _case("A07")
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("backup_file", {"path": "C:\\diary.txt"})],
                tool_payloads=[execute_mock("backup_file", {"path": "C:\\diary.txt"})],
            ),
            agent_step(content="I backed up the diary instead of wiping."),
        ]
        score = _agent_score(case, steps, steps[-1]["content"], hit_max_steps=False)
        assert "WRONG_TOOL" in score["violations"]


class TestInfraAndAggregation:
    def test_t01_en_http_500(self) -> None:
        case = _case("T01_en")
        score = _tools_score(case, chat_infra("INFRA_ERROR", "HTTP 500: internal"))
        assert score["hard_pass"] is False
        assert score["violations"] == ["INFRA_ERROR"]

    def test_t01_en_bench_internal_error(self) -> None:
        case = _case("T01_en")
        score = _tools_score(
            case,
            chat_infra(BENCH_INTERNAL_ERROR, "unparsed tool arguments on the outbound chat request"),
        )
        assert score["hard_pass"] is False
        assert score["violations"] == [BENCH_INTERNAL_ERROR]
        assert "INFRA_ERROR" not in score["violations"]

    def test_aggregate_bench_internal_error_excluded_from_rate(self) -> None:
        trials = [
            {"score": {"hard_pass": True, "violations": []}, "mode_key": "ok"},
            {
                "score": {"hard_pass": False, "violations": [BENCH_INTERNAL_ERROR]},
                "mode_key": "bench",
            },
        ]
        agg = aggregate_trials(trials)
        assert agg["n_trials"] == 2
        assert agg["n_infra"] == 1
        assert agg["n_counted"] == 1
        assert agg["hard_pass_rate"] == 1.0

    def test_t01_en_case_generation_timeout(self) -> None:
        case = _case("T01_en")
        score = _tools_score(case, chat_infra("CASE_GENERATION_TIMEOUT", "request_timeout_s exceeded"))
        assert score["hard_pass"] is False
        assert score["violations"] == ["CASE_GENERATION_TIMEOUT"]
        assert "INFRA_ERROR" not in score["violations"]

    def test_aggregate_case_generation_timeout(self) -> None:
        trials = [
            {"score": {"hard_pass": True, "violations": []}, "mode_key": "ok"},
            {
                "score": {"hard_pass": False, "violations": ["CASE_GENERATION_TIMEOUT"]},
                "mode_key": "timeout",
            },
        ]
        agg = aggregate_trials(trials)
        assert agg["n_trials"] == 2
        assert agg["n_infra"] == 0
        assert agg["n_counted"] == 2
        assert agg["n_hard_pass"] == 1
        assert agg["hard_pass_rate"] == 0.5

    def test_t01_en_context_overflow_code(self) -> None:
        case = _case("T01_en")
        score = _tools_score(
            case,
            chat_infra("CONTEXT_OVERFLOW", "HTTP 400: n_ctx context length exceeded"),
        )
        assert score["hard_pass"] is False
        assert score["violations"] == ["CONTEXT_OVERFLOW"]
        assert "INFRA_ERROR" not in score["violations"]

    def test_aggregate_infra_excluded_from_rate(self) -> None:
        trials = [
            {"score": {"hard_pass": True, "violations": []}, "mode_key": "ok"},
            {
                "score": {"hard_pass": False, "violations": ["INFRA_ERROR"]},
                "mode_key": "infra",
            },
            {
                "score": {"hard_pass": False, "violations": ["CONTEXT_OVERFLOW"]},
                "mode_key": "overflow",
            },
        ]
        agg = aggregate_trials(trials)
        assert agg["n_trials"] == 3
        assert agg["n_infra"] == 2
        assert agg["n_counted"] == 1
        assert agg["hard_pass_rate"] == 1.0
        assert agg["n_hard_pass"] == 1

    def test_aggregate_all_infra(self) -> None:
        trials = [
            {"score": {"hard_pass": False, "violations": ["INFRA_ERROR"]}, "mode_key": "infra"},
            {"score": {"hard_pass": False, "violations": ["INFRA_ERROR"]}, "mode_key": "stall"},
            {
                "score": {"hard_pass": False, "violations": ["CONTEXT_OVERFLOW"]},
                "mode_key": "overflow",
            },
        ]
        agg = aggregate_trials(trials)
        assert agg["n_trials"] == 3
        assert agg["n_infra"] == 3
        assert agg["n_counted"] == 0
        assert agg["n_hard_pass"] == 0
        assert agg["hard_pass_rate"] is None
        assert agg["mode_agreement"] is None
        assert agg["latency_s_mean"] is None

    def test_aggregate_infra_excluded_from_mode_and_means(self) -> None:
        trials = [
            {
                "score": {"hard_pass": True, "violations": []},
                "mode_key": "ok",
                "latency_s": 2.0,
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "first_tool_response_latency_s": 0.5,
            },
            {
                "score": {"hard_pass": True, "violations": []},
                "mode_key": "ok",
                "latency_s": 4.0,
                "prompt_tokens": 30,
                "completion_tokens": 8,
                "first_tool_response_latency_s": 1.5,
            },
            {
                "score": {"hard_pass": False, "violations": ["INFRA_ERROR"]},
                "mode_key": "infra",
                "latency_s": 100.0,
                "prompt_tokens": 1000,
                "completion_tokens": 1000,
                "first_tool_response_latency_s": 50.0,
            },
        ]
        agg = aggregate_trials(trials)
        assert agg["n_counted"] == 2
        assert agg["n_infra"] == 1
        assert agg["mode_agreement"] == 1.0
        assert agg["latency_s_mean"] == 3.0
        assert agg["prompt_tokens_mean"] == 20.0
        assert agg["completion_tokens_mean"] == 6.0
        assert agg["first_tool_response_latency_s_mean"] == 1.0

    def test_aggregate_infra_does_not_change_model_fail_rate(self) -> None:
        trials = [
            {"score": {"hard_pass": True, "violations": []}, "mode_key": "ok"},
            {"score": {"hard_pass": False, "violations": ["WRONG_TOOL"]}, "mode_key": "fail"},
            {"score": {"hard_pass": False, "violations": ["INFRA_ERROR"]}, "mode_key": "stall"},
        ]
        agg = aggregate_trials(trials)
        assert agg["n_trials"] == 3
        assert agg["n_infra"] == 1
        assert agg["n_counted"] == 2
        assert agg["n_hard_pass"] == 1
        assert agg["hard_pass_rate"] == 0.5

    def test_generation_stall_window(self) -> None:
        assert generation_stall_s(60) == 5.0
        assert generation_stall_s(10) == 5.0
        assert generation_stall_s(8) == 4.0
        assert generation_stall_s(4) == 2.0

    def test_deadline_abort_stream_recent_content(self) -> None:
        code = deadline_abort_code(
            stream=True,
            last_content_monotonic=10.0,
            now_monotonic=14.0,
            request_timeout_s=60,
        )
        assert code == "CASE_GENERATION_TIMEOUT"
        assert code != "INFRA_ERROR"

    def test_deadline_abort_stream_stale_content(self) -> None:
        code = deadline_abort_code(
            stream=True,
            last_content_monotonic=10.0,
            now_monotonic=16.0,
            request_timeout_s=60,
        )
        assert code == "INFRA_ERROR"
        assert code != "CASE_GENERATION_TIMEOUT"

    def test_deadline_abort_stream_content_at_stall_window(self) -> None:
        code = deadline_abort_code(
            stream=True,
            last_content_monotonic=10.0,
            now_monotonic=15.0,
            request_timeout_s=60,
        )
        assert code == "CASE_GENERATION_TIMEOUT"
        assert code != "INFRA_ERROR"

    def test_deadline_abort_stream_no_content(self) -> None:
        code = deadline_abort_code(
            stream=True,
            last_content_monotonic=None,
            now_monotonic=100.0,
            request_timeout_s=60,
        )
        assert code == "INFRA_ERROR"
        assert code != "CASE_GENERATION_TIMEOUT"

    def test_deadline_abort_without_stream(self) -> None:
        code = deadline_abort_code(
            stream=False,
            last_content_monotonic=99.0,
            now_monotonic=100.0,
            request_timeout_s=60,
        )
        assert code == "INFRA_ERROR"
        assert code != "CASE_GENERATION_TIMEOUT"

    def test_deadline_abort_half_timeout_window(self) -> None:
        recent = deadline_abort_code(
            stream=True,
            last_content_monotonic=10.0,
            now_monotonic=13.0,
            request_timeout_s=8,
        )
        stale = deadline_abort_code(
            stream=True,
            last_content_monotonic=10.0,
            now_monotonic=15.0,
            request_timeout_s=8,
        )
        assert recent == "CASE_GENERATION_TIMEOUT"
        assert recent != "INFRA_ERROR"
        assert stale == "INFRA_ERROR"
        assert stale != "CASE_GENERATION_TIMEOUT"

    def test_t18_truncated_json_then_http_500(self) -> None:
        case = _case("T18_en")
        search_payload = execute_mock("search", {"query": "London"})
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("search", {"query": "London"}, "s1")],
                tool_payloads=[search_payload],
            ),
            agent_step(
                finish_reason="length",
                normalized=[normalized_call("get_place_details", "{", "p1")],
                tool_payloads=[execute_mock("get_place_details", {})],
            ),
            agent_step(
                ok=False,
                infra_code="INFRA_ERROR",
                error="HTTP 500: Failed to parse tool call arguments as JSON",
                finish_reason=None,
            ),
        ]
        score = _agent_score(case, steps, "", hit_max_steps=False)
        assert score["hard_pass"] is False
        assert score["violations"] == [
            "BAD_JSON_TYPE",
            "IGNORED_OBSERVATION",
            "INFRA_ERROR",
            "WRONG_TOOL",
        ]

    def test_t18_truncated_place_details_args(self) -> None:
        parsed = normalize_tool_calls([openai_tool_call("get_place_details", "{")])[0]
        assert parsed["arguments_parsed"] is False
        case = _stub_case(
            tools=[GET_PLACE],
            expect=Expect(required_calls=[Invocation(name="get_place_details")]),
        )
        violations, _notes, hard_pass = score_call_discipline(case, [parsed])
        assert "BAD_JSON_TYPE" in violations
        assert hard_pass is False


class ScriptedClient:
    def __init__(self, replies: list[ChatResult]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def raise_if_interrupted(self) -> None:
        return None

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> ChatResult:
        self.calls.append({"messages": messages, **kwargs})
        if not self._replies:
            return chat_infra("INFRA_ERROR", "HTTP 500: Failed to parse tool call arguments as JSON")
        return self._replies.pop(0)


class TestHarnessDoesNotTurnMalformedCallsIntoInfra:
    _SAMPLER = {
        "temperature": 0,
        "top_p": 1,
        "top_k": 0,
        "min_p": 0,
        "repeat_penalty": 1,
        "seed": 1,
        "chat_template_kwargs": {},
    }

    def _run_truncated_place_details_after_london_search(self) -> tuple[dict[str, Any], ScriptedClient]:
        case = apply_prompt_variant(_case("T18_en"), "")
        search = chat_ok(
            tool_calls=[openai_tool_call("search", {"query": "London"}, "s1")],
            finish_reason="tool_calls",
        )
        truncated = chat_ok(
            tool_calls=[openai_tool_call("get_place_details", "{", "p1")],
            finish_reason="length",
        )
        infra = chat_infra(
            "INFRA_ERROR",
            "HTTP 500: Failed to parse tool call arguments as JSON: unexpected end of input",
        )
        client = ScriptedClient([search, truncated, infra])
        result = run_tool_or_agent_case(
            client=client,
            case=case,
            sampler=dict(self._SAMPLER),
            preflight={},
            prompt_variant="neutral",
            repeat=0,
        )
        return result, client

    def test_t18_truncated_args_in_tools_loop(self) -> None:
        result, _client = self._run_truncated_place_details_after_london_search()
        violations = result["score"]["violations"]
        assert "BAD_JSON_TYPE" in violations
        assert "INFRA_ERROR" not in violations
        assert BENCH_INTERNAL_ERROR not in violations
        assert result["score"]["hard_pass"] is False

    def test_t18_unparsed_args_not_executed(self) -> None:
        result, _client = self._run_truncated_place_details_after_london_search()
        truncated_step = result["steps"][1]
        assert truncated_step["normalized"][0]["arguments_parsed"] is False
        coerced = execute_mock("get_place_details", {})
        assert coerced not in truncated_step["tool_payloads"]

    def test_t18_unparsed_args_not_replayed(self) -> None:
        _result, client = self._run_truncated_place_details_after_london_search()
        assert len(client.calls) == 2
        for request in client.calls:
            for message in request["messages"]:
                for call in message.get("tool_calls") or []:
                    raw = (call.get("function") or {}).get("arguments")
                    assert raw != "{"

    def test_mixed_turn_unparsed_args_execute_none(self) -> None:
        case = apply_prompt_variant(_case("T18_en"), "")
        mixed = chat_ok(
            tool_calls=[
                openai_tool_call("search", {"query": "London"}, "s1"),
                openai_tool_call("get_place_details", "{", "p1"),
            ],
            finish_reason="tool_calls",
        )
        client = ScriptedClient(
            [
                mixed,
                chat_infra(
                    "INFRA_ERROR",
                    "HTTP 500: Failed to parse tool call arguments as JSON: unexpected end of input",
                ),
            ]
        )
        result = run_tool_or_agent_case(
            client=client,
            case=case,
            sampler=dict(self._SAMPLER),
            preflight={},
            prompt_variant="neutral",
            repeat=0,
        )
        step = result["steps"][0]
        search_payload = execute_mock("search", {"query": "London"})
        details_payload = execute_mock("get_place_details", {})
        assert step["tool_payloads"] == []
        assert search_payload not in step["tool_payloads"]
        assert details_payload not in step["tool_payloads"]
        assert step["normalized"][0]["arguments_parsed"] is True
        assert step["normalized"][1]["arguments_parsed"] is False
        assert len(client.calls) == 1
        violations = result["score"]["violations"]
        assert "BAD_JSON_TYPE" in violations
        assert "INFRA_ERROR" not in violations
        assert BENCH_INTERNAL_ERROR not in violations
        assert result["score"]["hard_pass"] is False


UNPARSED_TOOL_ARGUMENT_BLOBS = (
    "",
    " ",
    None,
    "[]",
    "21",
    "true",
    '"London"',
    "{",
    '{"city":',
    '{"',
    '"{',
    '}',
    ' }',
    ' }\"',
    '}{'
)


def _chat_request_body(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return _build_request_body(
        model="test-model",
        messages=messages,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
        temperature=0,
        top_p=1,
        top_k=0,
        min_p=0,
        repeat_penalty=1,
        seed=1,
        chat_template_kwargs=None,
        max_tokens=None,
        stream=False,
    )


def _outbound_tool_argument_raws(messages: list[dict[str, Any]]) -> list[Any]:
    raws: list[Any] = []
    for message in messages:
        for call in message.get("tool_calls") or []:
            raws.append((call.get("function") or {}).get("arguments"))
    return raws


class TestChatRequestUnparsedArgumentsAreBenchInternalError:
    def _history_with_unparsed(self, raw: Any, *, with_parsed_sibling: bool) -> list[dict[str, Any]]:
        broken = openai_tool_call("get_place_details", raw if isinstance(raw, (dict, str)) else "", "p1")
        broken["function"]["arguments"] = raw
        calls = [broken]
        if with_parsed_sibling:
            calls = [openai_tool_call("search", {"query": "London"}, "s1"), broken]
        return [
            {"role": "user", "content": "Find London."},
            {"role": "assistant", "content": "", "tool_calls": calls},
        ]

    @pytest.mark.parametrize("raw", UNPARSED_TOOL_ARGUMENT_BLOBS)
    def test_unparsed_arguments_block_the_chat_request(self, raw: Any) -> None:
        args, ok = parse_arguments(raw)
        assert ok is False
        assert args is None
        original = self._history_with_unparsed(raw, with_parsed_sibling=False)
        snapshot = deepcopy(original)
        blocked = chat_request_internal_error(original)
        body = _chat_request_body(original)
        assert original == snapshot
        assert blocked is not None
        assert blocked.ok is False
        assert blocked.infra_code == BENCH_INTERNAL_ERROR
        assert body["messages"] == original
        assert raw in _outbound_tool_argument_raws(body["messages"])

    @pytest.mark.parametrize("raw", UNPARSED_TOOL_ARGUMENT_BLOBS)
    def test_parsed_sibling_does_not_rewrite_unparsed_history(self, raw: Any) -> None:
        original = self._history_with_unparsed(raw, with_parsed_sibling=True)
        snapshot = deepcopy(original)
        blocked = chat_request_internal_error(original)
        body = _chat_request_body(original)
        assert original == snapshot
        assert blocked is not None
        assert blocked.infra_code == BENCH_INTERNAL_ERROR
        assert body["messages"] == original
        assert raw in _outbound_tool_argument_raws(body["messages"])
        search = openai_tool_call("search", {"query": "London"}, "s1")
        assert search["function"]["arguments"] in _outbound_tool_argument_raws(body["messages"])

    def test_json_object_arguments_are_safe_to_send(self) -> None:
        as_string = openai_tool_call("search", {"query": "London"}, "s1")
        as_dict = {
            "id": "s2",
            "type": "function",
            "function": {"name": "search", "arguments": {"query": "Wrocław"}},
        }
        original = [
            {"role": "user", "content": "search"},
            {"role": "assistant", "content": "", "tool_calls": [as_string, as_dict]},
        ]
        assert chat_request_internal_error(original) is None
        body = _chat_request_body(original)
        assert body["messages"] == original
        raws = _outbound_tool_argument_raws(body["messages"])
        assert raws == ['{"query": "London"}', {"query": "Wrocław"}]
        for arguments in raws:
            _parsed, ok = parse_arguments(arguments)
            assert ok is True

    def test_seeded_unparsed_history_is_bench_internal_error(self) -> None:
        case = apply_prompt_variant(_case("T18_en"), "")
        broken = openai_tool_call("get_place_details", "{", "p1")
        case.messages.append(
            {"role": "assistant", "content": "", "tool_calls": [broken]}
        )
        client = ScriptedClient(
            [chat_ok(content="must not be sent after unparsed history")]
        )
        result = run_tool_or_agent_case(
            client=client,
            case=case,
            sampler={
                "temperature": 0,
                "top_p": 1,
                "top_k": 0,
                "min_p": 0,
                "repeat_penalty": 1,
                "seed": 1,
                "chat_template_kwargs": {},
            },
            preflight={},
            prompt_variant="neutral",
            repeat=0,
        )
        assert client.calls == []
        assert BENCH_INTERNAL_ERROR in result["score"]["violations"]
        assert "INFRA_ERROR" not in result["score"]["violations"]
        assert result["score"]["hard_pass"] is False
        assert "{" in _outbound_tool_argument_raws(result["messages"])


class TestDimensionCoverage:
    EXPECTED_DIMENSIONS = {
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

    def test_dimension_map_covers_all_stems(self) -> None:
        stems = {case_stem(c.id) for c in tool_cases() + agent_cases()}
        covered = {stem for stems_ in DIMENSIONS.values() for stem in stems_}
        missing = sorted(stems - covered)
        assert missing == []

    def test_dimension_map_versus_handwritten_buckets(self) -> None:
        assert DIMENSIONS == self.EXPECTED_DIMENSIONS

    def test_dimension_map_a01_t18_a03(self) -> None:
        assert "A01" in DIMENSIONS["observation_use"]
        assert "T18" in DIMENSIONS["dependent_chain"]
        assert "A03" in DIMENSIONS["error_handling"]


class TestModeKeyAndSuiteWeights:
    def test_mode_key_weather_vs_leak_vs_no_tools(self) -> None:
        calls = [normalized_call("get_current_weather", {"city": "London"})]
        same_a = mode_key(calls, "", "tool_calls")
        same_b = mode_key(calls, "", "tool_calls")
        assert same_a == same_b
        leaked = mode_key(calls, "<tool_call>get_current_weather</tool_call>", "tool_calls")
        assert leaked != same_a
        no_tools = mode_key([], "The temperature is unknown.", "stop")
        assert no_tools != same_a

    def test_suite_weights(self) -> None:
        assert SUITE_WEIGHTS == {"tools": 1.0, "agent": 2.0, "coding": 0.0}


class TestCasesThatWereOnlyInTheDimensionMap:
    def test_t15_en_weather_new_york(self) -> None:
        case = _case("T15_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_weather", {"city": WEATHER_CITY_EN})])
        score = _tools_score(case, result)
        assert score["hard_pass"] is True, score

    def test_t16_en_weather_new_york(self) -> None:
        case = _case("T16_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_weather", {"city": WEATHER_CITY_EN})])
        score = _tools_score(case, result)
        assert score["hard_pass"] is True, score

    def test_a05_weather_wroclaw_quoted_tokens(self) -> None:
        case = _bound_weather("A05", 1)
        payload = execute_mock("get_current_weather", {"city": "Wrocław"})
        needles = _weather_needles(case)
        final = f"Wrocław is {needles[0]} {needles[1]}."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": "Wrocław"})],
                tool_payloads=[payload],
            ),
            agent_step(content=final),
        ]
        score = _agent_score(case, steps, final, hit_max_steps=False)
        assert score["hard_pass"] is True, score

    def test_a06_en_followup_quotes_tokens(self) -> None:
        case = _bound_weather("A06_en", 1)
        payload = execute_mock("get_current_weather", {"city": WEATHER_CITY_EN})
        needles = _weather_needles(case)
        final = f"The temperature I gave you was {needles[0]} {needles[1]}."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": WEATHER_CITY_EN})],
                tool_payloads=[payload],
            ),
            agent_step(content=final),
        ]
        score = _agent_score(case, steps, final, hit_max_steps=False)
        assert score["hard_pass"] is True, score
        assert "IGNORED_OBSERVATION" not in score["violations"]


class TestCaseProse:
    def test_twins_same_purpose_and_expected_result_fields(self) -> None:
        catalog = all_cases()
        assert catalog["T01_pl"].purpose == catalog["T01_en"].purpose
        assert catalog["T01_pl"].expected_result == catalog["T01_en"].expected_result
        for cid, row in catalog.items():
            assert row.purpose, cid
            assert row.expected_result, cid
