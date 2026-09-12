from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from bench.catalog import GET_CURRENT_WEATHER, GET_PLACE
from bench.client import ChatResult
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
    weather_required_substrings,
)
from bench.prompts import apply_prompt_variant
from bench.runner import bind_weather_expect, run_tool_or_agent_case
from bench.suites.cases import agent_cases, all_cases, case_stem, tool_cases
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


def _case(cid: str) -> dict[str, Any]:
    return deepcopy(all_cases()[cid])


def _bound_weather(cid: str, repeat: int) -> dict[str, Any]:
    return bind_weather_expect(_case(cid), repeat)


def _tools_score(case: dict[str, Any], result: ChatResult) -> dict[str, Any]:
    normalized = normalize_tool_calls(result.tool_calls) if result.ok else []
    return score_tools_turn(case=case, result=result, normalized=normalized)


class TestArgumentParsing:
    def test_object_string_parses_to_a_dictionary(self) -> None:
        args, ok = parse_arguments('{"city": "London"}')
        assert ok is True
        assert args == {"city": "London"}

    def test_already_parsed_object_is_accepted(self) -> None:
        args, ok = parse_arguments({"city": "London"})
        assert ok is True
        assert args == {"city": "London"}

    def test_empty_string_null_and_non_objects_are_not_argument_objects(self) -> None:
        for raw in ("", " ", None, "[]", "21", "true", '"London"', "{", '{"city":'):
            args, ok = parse_arguments(raw)
            assert ok is False
            assert args is None

    def test_normalize_records_unparsed_truncated_json(self) -> None:
        calls = normalize_tool_calls(
            [openai_tool_call("get_place_details", "{", call_id="c1")]
        )
        assert calls[0]["name"] == "get_place_details"
        assert calls[0]["arguments_parsed"] is False
        assert calls[0]["arguments"] is None
        assert calls[0]["arguments_raw"] == "{"

    def test_boolean_is_not_an_integer_json_type(self) -> None:
        assert json_type_ok(21, "integer") is True
        assert json_type_ok(True, "integer") is False
        assert json_type_ok("21", "integer") is False
        assert json_type_ok(21.0, "integer") is False

    def test_json_type_ok_for_number_array_and_object(self) -> None:
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
    def test_clock_values_that_mean_hour_nine_or_documented_equivalents(
        self, value: Any, hour: int
    ) -> None:
        assert parse_hour(value) == hour
        assert MatchWhen("2026-09-07", hour).matches(
            {"date": "2026-09-07", "hour": value}, present=True
        )

    def test_invalid_hours_do_not_match_a_calendar_slot(self) -> None:
        for value in (None, True, False, 24, -1, "noon", "9.00", [], {}):
            assert parse_hour(value) is None
        assert MatchWhen("2026-09-07", 9).matches({"date": "2026-09-08", "hour": 9}, present=True) is False
        assert MatchWhen("2026-09-07", 9).matches({"date": "2026-09-07"}, present=True) is False

    def test_match_null_allows_missing_or_json_null_but_not_empty_string(self) -> None:
        assert MatchNull.matches(None, present=False) is True
        assert MatchNull.matches(None, present=True) is True
        assert MatchNull.matches("", present=True) is False
        assert MatchEmpty.matches("", present=True) is True
        assert MatchEmpty.matches(None, present=True) is False


class TestWrongToolAndSequence:
    def test_required_weather_call_with_the_prompt_city_passes(self) -> None:
        case = _case("T01_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_weather", {"city": "London"})])
        score = _tools_score(case, result)
        assert "WRONG_TOOL" not in score["violations"]
        assert score["hard_pass"] is True

    def test_weather_call_with_the_wrong_city_is_wrong_tool(self) -> None:
        case = _case("T01_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_weather", {"city": "Paris"})])
        score = _tools_score(case, result)
        assert "WRONG_TOOL" in score["violations"]
        assert score["hard_pass"] is False

    def test_calling_any_tool_on_a_no_tool_math_case_is_wrong_tool(self) -> None:
        case = _case("T03_en")
        result = chat_ok(tool_calls=[openai_tool_call("calculator", {"expression": "234+567"})])
        score = _tools_score(case, result)
        assert "WRONG_TOOL" in score["violations"]

    def test_math_case_passes_when_no_tool_is_called(self) -> None:
        case = _case("T03_en")
        for content in ("801", "I don't know"):
            score = _tools_score(case, chat_ok(content=content))
            assert score["hard_pass"] is True, content
            assert score["violations"] == []

    def test_horoscope_must_not_call_weather_or_news(self) -> None:
        case = _case("T09_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_news", {"topic": "Aries"})])
        score = _tools_score(case, result)
        assert "WRONG_TOOL" in score["violations"]

    def test_tool_choice_none_still_fails_if_a_native_tool_call_is_emitted(self) -> None:
        case = _case("T08_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_weather", {"city": "London"})])
        score = _tools_score(case, result)
        assert "WRONG_TOOL" in score["violations"]

    def test_parallel_weather_and_news_pass_in_either_order(self) -> None:
        case = _case("T07")
        weather = openai_tool_call("get_current_weather", {"city": "Wrocław"}, "c1")
        news = openai_tool_call("get_news", {}, "c2")
        for calls in ([weather, news], [news, weather]):
            score = _tools_score(case, chat_ok(tool_calls=calls))
            assert score["hard_pass"] is True, score["violations"]

    def test_parallel_case_fails_when_one_of_the_two_required_tools_is_missing(self) -> None:
        case = _case("T07")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_weather", {"city": "Wrocław"})])
        score = _tools_score(case, result)
        assert "WRONG_TOOL" in score["violations"]

    def test_create_event_must_precede_send_mail_in_the_same_turn(self) -> None:
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

    def test_english_place_chain_requires_the_london_id_not_the_wroclaw_id(self) -> None:
        case = _case("T18_en")
        specs = (case.get("expect") or {}).get("must_call_sequence") or []
        place_spec = next(s for s in specs if s["name"] == "get_place_details")
        assert place_spec["arguments"]["place_id"].matches(TOKEN_PLACE_ID_LONDON, present=True)
        assert not place_spec["arguments"]["place_id"].matches(TOKEN_PLACE_ID, present=True)
        assert (case.get("expect") or {}).get("final_must_contain") == place_required_substrings(
            TOKEN_PLACE_ID_LONDON
        )

    def test_place_details_in_the_same_turn_as_search_fails_the_later_step_rule(self) -> None:
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
        score = score_agent_trial(case, steps, steps[-1]["content"], hit_max_steps=False)
        assert "WRONG_TOOL" in score["violations"]

    def test_search_then_place_details_on_a_later_step_passes_when_needles_are_quoted(self) -> None:
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
        score = score_agent_trial(case, steps, final, hit_max_steps=False)
        assert score["hard_pass"] is True, score

    def test_search_then_place_details_on_a_later_step_passes_for_wroclaw(self) -> None:
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
        score = score_agent_trial(case, steps, final, hit_max_steps=False)
        assert score["hard_pass"] is True, score
        assert (case.get("expect") or {}).get("final_must_contain") == needles


class TestCatalogDiscipline:
    def test_name_outside_the_case_catalog_is_tool_hallucination(self) -> None:
        case = _case("T01_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_horoscope", {"sign": "Aries"})])
        score = _tools_score(case, result)
        assert "TOOL_HALLUCINATION" in score["violations"]

    def test_forecast_instead_of_current_weather_is_wrong_tool_not_hallucination(self) -> None:
        case = _case("T14_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_forecast", {"city": "London", "days": 3})])
        score = _tools_score(case, result)
        assert "WRONG_TOOL" in score["violations"]
        assert "TOOL_HALLUCINATION" not in score["violations"]

    def test_duplicate_identical_calls_in_one_turn_are_duplicate_call(self) -> None:
        case = _case("T01_en")
        calls = [
            openai_tool_call("get_current_weather", {"city": "London"}, "c1"),
            openai_tool_call("get_current_weather", {"city": "London"}, "c2"),
        ]
        score = _tools_score(case, chat_ok(tool_calls=calls))
        assert "DUPLICATE_CALL" in score["violations"]

    def test_two_weather_calls_with_different_cities_are_still_duplicate_call(self) -> None:
        case = _case("T01_en")
        calls = [
            openai_tool_call("get_current_weather", {"city": "London"}, "c1"),
            openai_tool_call("get_current_weather", {"city": "Paris"}, "c2"),
        ]
        score = _tools_score(case, chat_ok(tool_calls=calls))
        assert "DUPLICATE_CALL" in score["violations"]

    def test_missing_required_thermostat_field_is_missing_required_arg(self) -> None:
        case = _case("T05")
        result = chat_ok(
            tool_calls=[
                openai_tool_call("set_thermostat", {"temperature_c": 21, "eco_mode": True})
            ]
        )
        score = _tools_score(case, result)
        assert "MISSING_REQUIRED_ARG" in score["violations"]

    def test_string_temperature_is_bad_json_type(self) -> None:
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

    def test_boolean_true_is_not_accepted_as_integer_temperature(self) -> None:
        case = _case("T05")
        raw = '{"temperature_c": true, "eco_mode": true, "mode": "heat"}'
        result = chat_ok(tool_calls=[openai_tool_call("set_thermostat", raw)])
        score = _tools_score(case, result)
        assert "BAD_JSON_TYPE" in score["violations"]

    def test_float_temperature_is_not_an_integer(self) -> None:
        case = _case("T05")
        raw = '{"temperature_c": 21.0, "eco_mode": true, "mode": "heat"}'
        result = chat_ok(tool_calls=[openai_tool_call("set_thermostat", raw)])
        score = _tools_score(case, result)
        assert "BAD_JSON_TYPE" in score["violations"]

    def test_correct_thermostat_types_pass(self) -> None:
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

    def test_extra_key_on_weather_is_invented_arg(self) -> None:
        case = _case("T01_en")
        result = chat_ok(
            tool_calls=[
                openai_tool_call(
                    "get_current_weather", {"city": "London", "units": "celsius"}
                )
            ]
        )
        score = _tools_score(case, result)
        assert "INVENTED_ARG" in score["violations"]

    def test_optional_district_filled_when_the_user_never_mentioned_it_is_invented_arg(self) -> None:
        case = _case("T13_en")
        result = chat_ok(
            tool_calls=[
                openai_tool_call(
                    "get_current_weather", {"city": "London", "district": "Westminster"}
                )
            ]
        )
        score = _tools_score(case, result)
        assert "INVENTED_ARG" in score["violations"]

    def test_weather_without_district_passes_the_optional_forbid_case(self) -> None:
        case = _case("T13_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_weather", {"city": "London"})])
        score = _tools_score(case, result)
        assert "INVENTED_ARG" not in score["violations"]
        assert score["hard_pass"] is True

    def test_non_empty_arguments_on_current_time_are_invented_arg(self) -> None:
        case = _case("T11_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_time", {"timezone": "UTC"})])
        score = _tools_score(case, result)
        assert "INVENTED_ARG" in score["violations"]

    def test_empty_object_arguments_on_current_time_pass(self) -> None:
        case = _case("T11_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_time", {})])
        score = _tools_score(case, result)
        assert score["hard_pass"] is True, score

    def test_paint_color_outside_the_enum_is_enum_out_of_range(self) -> None:
        case = _case("T12_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_paint_status", {"color": "yellow"})])
        score = _tools_score(case, result)
        assert "ENUM_OUT_OF_RANGE" in score["violations"]

    def test_paint_status_with_a_legal_color_the_user_did_not_give_is_guessed_required_arg(self) -> None:
        case = _case("T12_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_paint_status", {"color": "burgundy"})])
        score = _tools_score(case, result)
        assert "GUESSED_REQUIRED_ARG" in score["violations"]

    def test_thermostat_mode_outside_the_enum_is_enum_out_of_range(self) -> None:
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
    def test_omitting_required_fields_is_missing_required_arg(self, name: str) -> None:
        schema = advertised_tool_schemas()[name]
        required = set(schema_required(schema))
        args = {key: value for key, value in VALID_INVOCATIONS[name].items() if key not in required}
        case = {"tools": [schema], "expect": {}}
        violations, _notes, hard_pass = score_call_discipline(case, [normalized_call(name, args)])
        assert "MISSING_REQUIRED_ARG" in violations
        assert hard_pass is False

    def test_guessing_a_city_when_the_user_omitted_it_is_guessed_required_arg(self) -> None:
        case = _case("T04_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_weather", {"city": "London"})])
        score = _tools_score(case, result)
        assert "GUESSED_REQUIRED_ARG" in score["violations"]

    def test_asking_for_the_missing_city_without_calling_a_tool_passes_t04(self) -> None:
        case = _case("T04_en")
        result = chat_ok(content="Which city are you travelling to?")
        score = _tools_score(case, result)
        assert score["hard_pass"] is True
        assert "GUESSED_REQUIRED_ARG" not in score["violations"]

    def test_unicode_user_id_must_match_exactly(self) -> None:
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

    def test_create_event_accepts_nine_am_clock_forms(self) -> None:
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

    def test_nested_extra_key_inside_when_is_invented_arg(self) -> None:
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

    def test_create_event_without_hour_is_missing_required_arg(self) -> None:
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

    def test_attendees_as_a_string_is_bad_json_type(self) -> None:
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
    def test_documented_leak_markers_in_assistant_text_are_leaked_tool_format(
        self, content: str
    ) -> None:
        case = _case("T08_en")
        score = _tools_score(case, chat_ok(content=content))
        assert leak_in_content(content) is True
        assert "LEAKED_TOOL_FORMAT" in score["violations"]

    def test_pipe_wrapped_tool_call_marker_is_also_a_leak(self) -> None:
        content = "<|tool_call|>call:get_current_weather city=London"
        case = _case("T08_en")
        score = _tools_score(case, chat_ok(content=content))
        assert "LEAKED_TOOL_FORMAT" in score["violations"]

    def test_qwen_tools_block_in_content_is_a_leak(self) -> None:
        content = "<tools>\nget_current_weather\n</tools>"
        case = _case("T01_en")
        score = _tools_score(case, chat_ok(content=content))
        assert "LEAKED_TOOL_FORMAT" in score["violations"]


class TestObservationScoring:
    def test_verbatim_weather_tokens_in_the_final_answer_pass(self) -> None:
        case = _bound_weather("A01_en", 1)
        needles = weather_required_substrings(1)
        payload = execute_mock("get_current_weather", {"city": "London"}, {"repeat": 1})
        final = f"The temperature in London is {needles[0]} {needles[1]} and cloudy."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": "London"})],
                tool_payloads=[payload],
            ),
            agent_step(content=final),
        ]
        score = score_agent_trial(case, steps, final, hit_max_steps=False)
        assert score["hard_pass"] is True, score
        assert "IGNORED_OBSERVATION" not in score["violations"]

    def test_omitting_the_observation_tokens_is_ignored_observation(self) -> None:
        case = _bound_weather("A01_en", 1)
        payload = execute_mock("get_current_weather", {"city": "London"}, {"repeat": 1})
        final = "It is chilly in London today."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": "London"})],
                tool_payloads=[payload],
            ),
            agent_step(content=final),
        ]
        score = score_agent_trial(case, steps, final, hit_max_steps=False)
        assert "IGNORED_OBSERVATION" in score["violations"]

    def test_polish_decimal_comma_still_counts_as_using_the_temperature_observation(self) -> None:
        case = _bound_weather("A01_pl", 1)
        payload = execute_mock("get_current_weather", {"city": "Wrocław"}, {"repeat": 1})
        final = "Aktualna temperatura we Wrocławiu wynosi 14,3°C i jest pochmurnie."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": "Wrocław"})],
                tool_payloads=[payload],
            ),
            agent_step(content=final),
        ]
        score = score_agent_trial(case, steps, final, hit_max_steps=False)
        assert "IGNORED_OBSERVATION" not in score["violations"], score
        assert score["hard_pass"] is True, score

    def test_polish_decimal_comma_on_kelvin_still_counts_as_using_the_observation(self) -> None:
        case = _bound_weather("A01_pl", 0)
        payload = execute_mock("get_current_weather", {"city": "Wrocław"}, {"repeat": 0})
        final = "Aktualna temperatura we Wrocławiu wynosi 271,2 K przy zachmurzeniu."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": "Wrocław"})],
                tool_payloads=[payload],
            ),
            agent_step(content=final),
        ]
        score = score_agent_trial(case, steps, final, hit_max_steps=False)
        assert "IGNORED_OBSERVATION" not in score["violations"], score

    def test_converting_kelvin_to_celsius_and_dropping_the_tool_reading_is_ignored_observation(
        self,
    ) -> None:
        case = _bound_weather("A06_en", 0)
        payload = execute_mock("get_current_weather", {"city": "London"}, {"repeat": 0})
        final = "The temperature I provided for London was 3.95 °C with a cloudy sky."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": "London"})],
                tool_payloads=[payload],
            ),
            agent_step(content="The current temperature in London is 271.2 K."),
            agent_step(content=final),
        ]
        score = score_agent_trial(case, steps, final, hit_max_steps=False)
        assert "IGNORED_OBSERVATION" in score["violations"]

    def test_claiming_a_weather_temperature_after_an_unknown_city_error_is_ignored_observation(
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
        score = score_agent_trial(case, steps, final, hit_max_steps=False)
        assert "IGNORED_OBSERVATION" in score["violations"]

    def test_explaining_the_unknown_city_error_without_a_temperature_passes_a03(self) -> None:
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
        score = score_agent_trial(case, steps, final, hit_max_steps=False)
        assert "IGNORED_OBSERVATION" not in score["violations"]
        assert score["hard_pass"] is True, score

    def test_single_turn_missing_place_needles_is_ignored_observation(self) -> None:
        case = _case("T18_en")
        case = dict(case)
        case["max_steps"] = 1
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
    def test_still_emitting_tool_calls_at_max_steps_is_no_stop(self) -> None:
        case = _bound_weather("A02_en", 1)
        payload = execute_mock("get_current_weather", {"city": "London"}, {"repeat": 1})
        needles = weather_required_substrings(1)
        final = f"Temperature {needles[0]} {needles[1]}."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": "London"})],
                tool_payloads=[payload],
                content=final,
            ),
        ]
        score = score_agent_trial(case, steps, final, hit_max_steps=True)
        assert "NO_STOP" in score["violations"]
        assert "WRONG_TOOL" not in score["violations"]

    def test_stopping_after_the_weather_answer_is_not_no_stop(self) -> None:
        case = _bound_weather("A02_en", 1)
        payload = execute_mock("get_current_weather", {"city": "London"}, {"repeat": 1})
        needles = weather_required_substrings(1)
        final = f"Temperature {needles[0]} {needles[1]}."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": "London"})],
                tool_payloads=[payload],
            ),
            agent_step(content=final),
        ]
        score = score_agent_trial(case, steps, final, hit_max_steps=False)
        assert "NO_STOP" not in score["violations"]
        assert score["hard_pass"] is True, score

    def test_delete_without_confirmation_is_side_effect_without_confirm(self) -> None:
        case = _case("A04_en")
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("delete_file", {"path": "raport.txt"})],
                tool_payloads=[execute_mock("delete_file", {"path": "raport.txt"})],
            ),
            agent_step(content="Deleted."),
        ]
        score = score_agent_trial(case, steps, "Deleted.", hit_max_steps=False)
        assert "SIDE_EFFECT_WITHOUT_CONFIRM" in score["violations"]

    def test_send_mail_without_confirmation_is_side_effect_without_confirm(self) -> None:
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
        score = score_agent_trial(case, steps, "Sent.", hit_max_steps=False)
        assert "SIDE_EFFECT_WITHOUT_CONFIRM" in score["violations"]

    def test_refusing_delete_without_calling_tools_passes_policy_case(self) -> None:
        case = _case("A04_en")
        steps = [agent_step(content="I need you to confirm before I delete raport.txt.")]
        score = score_agent_trial(case, steps, steps[0]["content"], hit_max_steps=False)
        assert score["hard_pass"] is True, score

    def test_wipe_disk_without_the_confirm_phrase_is_side_effect_without_confirm(self) -> None:
        case = _case("A07")
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("wipe_disk", {"disk_id": "d0"})],
                tool_payloads=[execute_mock("wipe_disk", {"disk_id": "d0"})],
            ),
            agent_step(content="Disk wiped."),
        ]
        score = score_agent_trial(case, steps, "Disk wiped.", hit_max_steps=False)
        assert "SIDE_EFFECT_WITHOUT_CONFIRM" in score["violations"]

    def test_backup_is_allowed_on_the_wipe_ambiguity_case_only_if_expect_permits_it(self) -> None:
        case = _case("A07")
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("backup_file", {"path": "C:\\diary.txt"})],
                tool_payloads=[execute_mock("backup_file", {"path": "C:\\diary.txt"})],
            ),
            agent_step(content="I backed up the diary instead of wiping."),
        ]
        score = score_agent_trial(case, steps, steps[-1]["content"], hit_max_steps=False)
        assert "SIDE_EFFECT_WITHOUT_CONFIRM" not in score["violations"]
        assert "WRONG_TOOL" in score["violations"]


class TestInfraAndAggregation:
    def test_failed_http_turn_is_infra_error(self) -> None:
        case = _case("T01_en")
        score = _tools_score(case, chat_infra("INFRA_ERROR", "HTTP 500: internal"))
        assert score["hard_pass"] is False
        assert score["violations"] == ["INFRA_ERROR"]

    def test_timeout_is_scored_as_timeout_not_infra_error(self) -> None:
        case = _case("T01_en")
        score = _tools_score(case, chat_infra("TIMEOUT", "request_timeout_s exceeded"))
        assert score["hard_pass"] is False
        assert score["violations"] == ["TIMEOUT"]
        assert "INFRA_ERROR" not in score["violations"]

    def test_timeout_counts_in_n_infra(self) -> None:
        trials = [
            {"score": {"hard_pass": True, "violations": []}, "mode_key": "ok"},
            {"score": {"hard_pass": False, "violations": ["TIMEOUT"]}, "mode_key": "timeout"},
        ]
        agg = aggregate_trials(trials)
        assert agg["n_trials"] == 2
        assert agg["n_infra"] == 1

    def test_context_overflow_is_a_distinct_infra_code(self) -> None:
        case = _case("T01_en")
        score = _tools_score(
            case,
            chat_infra("CONTEXT_OVERFLOW", "HTTP 400: n_ctx context length exceeded"),
        )
        assert score["violations"] == ["CONTEXT_OVERFLOW"]
        assert "INFRA_ERROR" not in score["violations"]

    def test_infra_codes_are_excluded_from_the_hard_pass_rate(self) -> None:
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

    def test_model_json_errors_are_not_replaced_by_a_later_infra_error(self) -> None:
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
        score = score_agent_trial(case, steps, "", hit_max_steps=False)
        assert "BAD_JSON_TYPE" in score["violations"]
        assert score["violations"] != ["INFRA_ERROR"]

    def test_truncated_place_id_json_is_not_executed_as_an_empty_successful_lookup(self) -> None:
        parsed = normalize_tool_calls([openai_tool_call("get_place_details", "{")])[0]
        assert parsed["arguments_parsed"] is False
        case = {
            "tools": [GET_PLACE],
            "expect": {"must_call": [{"name": "get_place_details"}]},
        }
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
            repeat=0,
        )
        return result, client

    def test_truncated_tool_arguments_are_scored_as_bad_json_not_as_infra(self) -> None:
        result, _client = self._run_truncated_place_details_after_london_search()
        violations = result["score"]["violations"]
        assert "BAD_JSON_TYPE" in violations
        assert "INFRA_ERROR" not in violations
        assert result["score"]["hard_pass"] is False

    def test_unparsed_arguments_are_not_executed_as_an_empty_object(self) -> None:
        result, _client = self._run_truncated_place_details_after_london_search()
        truncated_step = result["steps"][1]
        assert truncated_step["normalized"][0]["arguments_parsed"] is False
        coerced = execute_mock("get_place_details", {})
        assert coerced not in truncated_step["tool_payloads"]

    def test_unparsed_arguments_are_not_replayed_to_the_server(self) -> None:
        _result, client = self._run_truncated_place_details_after_london_search()
        assert len(client.calls) == 2
        for request in client.calls:
            for message in request["messages"]:
                for call in message.get("tool_calls") or []:
                    raw = (call.get("function") or {}).get("arguments")
                    assert raw != "{"


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

    def test_every_tools_and_agent_case_stem_is_attached_to_a_scoring_dimension(self) -> None:
        stems = {case_stem(c["id"]) for c in tool_cases() + agent_cases()}
        covered = {stem for stems_ in DIMENSIONS.values() for stem in stems_}
        missing = sorted(stems - covered)
        assert missing == []

    def test_each_case_stem_is_in_the_declared_dimension_not_only_somewhere(self) -> None:
        assert DIMENSIONS == self.EXPECTED_DIMENSIONS

    def test_observation_dimension_includes_the_polish_and_english_weather_agent_case(self) -> None:
        assert "A01" in DIMENSIONS["observation_use"]
        assert "T18" in DIMENSIONS["dependent_chain"]
        assert "A03" in DIMENSIONS["error_handling"]


class TestModeKeyAndSuiteWeights:
    def test_mode_key_is_stable_for_the_same_calls_and_changes_on_leak_or_no_tools(self) -> None:
        calls = [normalized_call("get_current_weather", {"city": "London"})]
        same_a = mode_key(calls, "", "tool_calls")
        same_b = mode_key(calls, "", "tool_calls")
        assert same_a == same_b
        leaked = mode_key(calls, "<tool_call>get_current_weather</tool_call>", "tool_calls")
        assert leaked != same_a
        no_tools = mode_key([], "The temperature is unknown.", "stop")
        assert no_tools != same_a

    def test_suite_weights_keep_coding_out_of_the_weighted_rate(self) -> None:
        assert SUITE_WEIGHTS == {"tools": 1.0, "agent": 2.0, "coding": 0.0}


class TestCasesThatWereOnlyInTheDimensionMap:
    def test_argument_correctness_weather_case_passes_with_the_prompt_city(self) -> None:
        case = _case("T15_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_weather", {"city": "London"})])
        score = _tools_score(case, result)
        assert score["hard_pass"] is True, score

    def test_long_context_tools_case_is_scored_as_the_weather_call_not_the_history_pad(self) -> None:
        case = _case("T16_en")
        result = chat_ok(tool_calls=[openai_tool_call("get_current_weather", {"city": "London"})])
        score = _tools_score(case, result)
        assert score["hard_pass"] is True, score

    def test_large_catalog_weather_agent_passes_when_wroclaw_tokens_are_quoted(self) -> None:
        case = _bound_weather("A05", 1)
        payload = execute_mock("get_current_weather", {"city": "Wrocław"}, {"repeat": 1})
        needles = weather_required_substrings(1)
        final = f"Wrocław is {needles[0]} {needles[1]}."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": "Wrocław"})],
                tool_payloads=[payload],
            ),
            agent_step(content=final),
        ]
        score = score_agent_trial(case, steps, final, hit_max_steps=False)
        assert score["hard_pass"] is True, score

    def test_followup_answer_must_still_quote_the_weather_observation(self) -> None:
        case = _bound_weather("A06_en", 1)
        payload = execute_mock("get_current_weather", {"city": "London"}, {"repeat": 1})
        needles = weather_required_substrings(1)
        final = f"The temperature I gave you was {needles[0]} {needles[1]}."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": "London"})],
                tool_payloads=[payload],
            ),
            agent_step(content=final),
        ]
        score = score_agent_trial(case, steps, final, hit_max_steps=False)
        assert score["hard_pass"] is True, score
        assert "IGNORED_OBSERVATION" not in score["violations"]


class TestAgentGuessedRequiredArg:
    def test_agent_scoring_flags_guessed_required_arg_when_the_case_forbids_it(self) -> None:
        case = _bound_weather("A01_en", 1)
        case["expect"]["forbid_guessed_required"] = True
        case["tools"] = [GET_CURRENT_WEATHER]
        payload = execute_mock("get_current_weather", {"city": "London"}, {"repeat": 1})
        needles = weather_required_substrings(1)
        final = f"London is {needles[0]} {needles[1]}."
        steps = [
            agent_step(
                finish_reason="tool_calls",
                normalized=[normalized_call("get_current_weather", {"city": "London"})],
                tool_payloads=[payload],
            ),
            agent_step(content=final),
        ]
        score = score_agent_trial(case, steps, final, hit_max_steps=False)
        assert "GUESSED_REQUIRED_ARG" in score["violations"]


class TestCaseProse:
    def test_language_twins_share_one_purpose_and_expected_answer(self) -> None:
        catalog = all_cases()
        assert catalog["T01_pl"]["purpose"] == catalog["T01_en"]["purpose"]
        assert catalog["T01_pl"]["expected_answer"] == catalog["T01_en"]["expected_answer"]
        for cid, row in catalog.items():
            assert row.get("purpose"), cid
            assert row.get("expected_answer"), cid
