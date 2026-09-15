from __future__ import annotations

import json
from typing import Any

import pytest

from bench.catalog import (
    CATALOG_8,
    CATALOG_LARGE,
    CREATE_EVENT,
    GET_CURRENT_WEATHER,
    GET_CURRENT_WEATHER_OPTIONAL,
    GET_TIME,
    PAINT_STATUS,
    SEND_MAIL,
    SET_THERMOSTAT,
)
from bench.mocks import (
    NEWS_HEADLINES,
    PLACE_LONDON,
    PLACE_WROCLAW,
    TOKEN_BACKUP,
    TOKEN_DELETE,
    TOKEN_ERROR,
    TOKEN_FORECAST,
    TOKEN_MAIL,
    TOKEN_PLACE_ID,
    TOKEN_PLACE_ID_LONDON,
    TOKEN_WIPE,
    execute_mock,
    place_required_substrings,
    weather_payload,
    weather_required_substrings,
)
from bench.hard_score import parse_arguments
from bench.suites.cases import all_cases
from helpers import (
    VALID_INVOCATIONS,
    advertised_tool_schemas,
    function_name,
    names_in_case_catalogs,
    schema_properties,
    schema_required,
    tools_with_required_fields,
)


def _json(name: str, arguments: dict[str, Any] | Any, extra: dict[str, Any] | None = None) -> Any:
    raw = execute_mock(name, arguments, extra)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _assert_acknowledgement_stub(tool: str, arguments: dict[str, Any]) -> None:
    payload = _json(tool, arguments)
    assert set(payload.keys()) == {"ok", "tool", "args"}
    assert payload["ok"] is True
    assert payload["tool"] == tool
    assert payload["args"] == arguments


def _is_acknowledgement_stub(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    keys = set(payload.keys())
    if keys == {"ok"}:
        return True
    return "tool" in keys and keys <= {"ok", "tool", "args", "name", "arguments"}


def _looks_like_success(name: str, raw: str, arguments: Any) -> bool:
    args, parsed = parse_arguments(arguments)
    if not parsed or args is None:
        return False
    schema = advertised_tool_schemas().get(name)
    if schema is not None:
        for key in schema_required(schema):
            if key not in args:
                return False
    if name == "delete_file":
        return raw == TOKEN_DELETE
    if name == "send_mail":
        return raw == TOKEN_MAIL
    if name == "backup_file":
        return raw == TOKEN_BACKUP
    if name == "wipe_disk":
        return raw == TOKEN_WIPE
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if _is_acknowledgement_stub(payload):
        return False
    if payload.get("error"):
        return False
    if name == "get_current_weather":
        return "temperature" in payload and "error" not in payload
    if name == "get_forecast":
        return bool(payload.get("token")) and "error" not in payload
    if name == "get_news":
        headlines = payload.get("headlines")
        return isinstance(headlines, list) and bool(headlines)
    if name == "get_current_time":
        return isinstance(payload.get("utc"), str) and payload["utc"].endswith("Z")
    if name == "search":
        results = payload.get("results")
        return isinstance(results, list) and bool(results)
    if name == "get_place_details":
        return "summary" in payload and "error" not in payload
    if name == "lookup_user":
        return payload.get("status") == "active"
    if name == "set_thermostat":
        return payload.get("ok") is True
    if name == "create_event":
        return payload.get("ok") is True
    if name == "get_paint_status":
        return payload.get("status") == "queued"
    if name == "read_note":
        content = payload.get("content")
        return isinstance(content, str) and bool(content.strip())
    return False


def _observation_raw_looks_like_success(name: str, raw: str) -> bool:
    """True when the observation string alone looks like a completed tool result."""
    if name == "delete_file":
        return raw == TOKEN_DELETE
    if name == "send_mail":
        return raw == TOKEN_MAIL
    if name == "backup_file":
        return raw == TOKEN_BACKUP
    if name == "wipe_disk":
        return raw == TOKEN_WIPE
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return False
    if _is_acknowledgement_stub(payload):
        return False
    if payload.get("error"):
        return False
    if name == "get_current_weather":
        return "temperature" in payload
    if name == "get_forecast":
        return bool(payload.get("token"))
    if name == "get_news":
        headlines = payload.get("headlines")
        return isinstance(headlines, list) and bool(headlines)
    if name == "get_current_time":
        utc = payload.get("utc")
        return isinstance(utc, str) and utc.endswith("Z")
    if name == "search":
        results = payload.get("results")
        return isinstance(results, list) and bool(results)
    if name == "get_place_details":
        return "summary" in payload
    if name == "lookup_user":
        return payload.get("status") == "active"
    if name == "set_thermostat":
        return payload.get("ok") is True
    if name == "create_event":
        return payload.get("ok") is True
    if name == "get_paint_status":
        return payload.get("status") == "queued"
    if name == "read_note":
        content = payload.get("content")
        return isinstance(content, str) and bool(content.strip())
    return False


class TestCatalogSurface:
    def test_invocation_fixtures_match_every_name_catalogs_and_cases_advertise(self) -> None:
        advertised = set(advertised_tool_schemas())
        case_names = names_in_case_catalogs()
        assert case_names <= advertised
        assert set(VALID_INVOCATIONS) == advertised

    def test_every_name_a_case_exposes_has_a_non_empty_mock(self) -> None:
        for name in sorted(names_in_case_catalogs()):
            raw = execute_mock(name, VALID_INVOCATIONS[name])
            assert isinstance(raw, str)
            assert raw != ""

    def test_eight_tool_catalog_versus_t02_and_t03(self) -> None:
        catalog_names = [function_name(tool) for tool in CATALOG_8]
        cases = all_cases()
        assert [function_name(tool) for tool in cases["T02"].tools] == catalog_names
        assert [function_name(tool) for tool in cases["T03_en"].tools] == catalog_names
        assert [function_name(tool) for tool in cases["T03_pl"].tools] == catalog_names

    def test_large_catalog_versus_a05(self) -> None:
        catalog_names = [function_name(tool) for tool in CATALOG_LARGE]
        assert [function_name(tool) for tool in all_cases()["A05"].tools] == catalog_names

    def test_optional_weather_case_exposes_district_and_the_basic_catalog_does_not(self) -> None:
        optional_en = all_cases()["T13_en"].tools[0]
        basic_en = all_cases()["T01_en"].tools[0]
        optional_pl = all_cases()["T13_pl"].tools[0]
        basic_pl = all_cases()["T01_pl"].tools[0]
        assert function_name(optional_en) == "get_current_weather"
        assert function_name(basic_en) == "get_current_weather"
        assert function_name(optional_pl) == "get_current_weather"
        assert function_name(basic_pl) == "get_current_weather"
        assert "district" in schema_properties(optional_en)
        assert "district" not in schema_properties(basic_en)
        assert "district" in schema_properties(optional_pl)
        assert "district" not in schema_properties(basic_pl)
        assert "district" not in schema_properties(GET_CURRENT_WEATHER)
        assert "district" in schema_properties(GET_CURRENT_WEATHER_OPTIONAL)

    def test_catalog_eight_and_large_only_contain_declared_function_tools(self) -> None:
        for catalog in (CATALOG_8, CATALOG_LARGE):
            assert catalog
            for tool in catalog:
                assert tool.get("type") == "function"
                assert function_name(tool)
                params = (tool.get("function") or {}).get("parameters") or {}
                assert params.get("type") == "object"
                assert params.get("additionalProperties") is False
                assert "properties" in params
                assert isinstance(params.get("properties"), dict)

    def test_optional_weather_schema_exposes_district_without_requiring_it(self) -> None:
        required = schema_required(GET_CURRENT_WEATHER_OPTIONAL)
        props = schema_properties(GET_CURRENT_WEATHER_OPTIONAL)
        assert "city" in required
        assert "district" not in required
        assert "district" in props
        assert "city" in props
        assert props["city"].get("type") == "string"
        assert props["district"].get("type") == "string"

    def test_paint_status_enum_does_not_include_yellow(self) -> None:
        color = schema_properties(PAINT_STATUS)["color"]
        assert color.get("enum") == ["burgundy", "navy", "ivory"]
        assert "yellow" not in color["enum"]
        assert schema_required(PAINT_STATUS) == ["color"]

    def test_thermostat_mode_enum_and_integer_temperature(self) -> None:
        props = schema_properties(SET_THERMOSTAT)
        assert props["temperature_c"]["type"] == "integer"
        assert props["eco_mode"]["type"] == "boolean"
        assert props["mode"]["enum"] == ["heat", "cool", "off"]
        assert set(schema_required(SET_THERMOSTAT)) == {"temperature_c", "eco_mode", "mode"}

    def test_create_event_nested_when_requires_date_and_hour(self) -> None:
        when = schema_properties(CREATE_EVENT)["when"]
        assert set(schema_required(CREATE_EVENT)) == {"title", "attendees", "when"}
        attendees = schema_properties(CREATE_EVENT)["attendees"]
        assert attendees.get("type") == "array"
        assert attendees.get("items") == {"type": "string"}
        assert when["type"] == "object"
        assert set(when["required"]) == {"date", "hour"}
        assert when.get("additionalProperties") is False

    def test_get_current_time_declares_no_parameters(self) -> None:
        assert schema_properties(GET_TIME) == {}
        assert schema_required(GET_TIME) == []
        params = (GET_TIME.get("function") or {}).get("parameters") or {}
        assert params.get("additionalProperties") is False

    def test_send_mail_body_is_optional(self) -> None:
        props = schema_properties(SEND_MAIL)
        assert set(schema_required(SEND_MAIL)) == {"to", "subject"}
        assert "to" in props and "subject" in props
        assert "body" in props


class TestExecuteMock:
    def test_unknown_catalog_name_returns_unknown_tool_error(self) -> None:
        payload = _json("get_horoscope", {})
        assert payload == {"error": "UNKNOWN_TOOL", "name": "get_horoscope"}


class TestRequiredArgumentsAreNotSuccessfulObservations:
    @pytest.mark.parametrize("name", tools_with_required_fields())
    def test_omitting_required_fields_does_not_return_a_successful_observation(self, name: str) -> None:
        schema = advertised_tool_schemas()[name]
        required = set(schema_required(schema))
        args = {key: value for key, value in VALID_INVOCATIONS[name].items() if key not in required}
        raw = execute_mock(name, args)
        assert not _observation_raw_looks_like_success(name, raw)

    @pytest.mark.parametrize("name", sorted(advertised_tool_schemas()))
    def test_truncated_non_object_arguments_do_not_return_a_successful_observation(self, name: str) -> None:
        raw = execute_mock(name, "{")
        assert json.loads(raw) == {"error": "UNPARSED_ARGUMENTS"}
        assert not _observation_raw_looks_like_success(name, raw)


class TestCurrentWeather:
    def test_wroclaw_returns_celsius_fixture(self, weather_tokens: dict[str, str]) -> None:
        payload = _json("get_current_weather", {"city": "Wrocław"})
        assert payload["temperature"] == float(weather_tokens["TOKEN_WEATHER_C"])
        assert payload["unit"] == weather_tokens["TOKEN_WEATHER_C_UNIT"]
        assert payload["sky"]

    def test_new_york_returns_fahrenheit_fixture(self, weather_tokens: dict[str, str]) -> None:
        payload = _json("get_current_weather", {"city": "New York"})
        assert payload["temperature"] == float(weather_tokens["TOKEN_WEATHER_F"])
        assert payload["unit"] == weather_tokens["TOKEN_WEATHER_F_UNIT"]
        assert payload["sky"]

    def test_weather_needles_align_fixture_and_tool_message_for_city(
        self, weather_tokens: dict[str, str]
    ) -> None:
        for city, temp_key, unit_key in (
            ("Wrocław", "TOKEN_WEATHER_C", "TOKEN_WEATHER_C_UNIT"),
            ("New York", "TOKEN_WEATHER_F", "TOKEN_WEATHER_F_UNIT"),
            ("London", "TOKEN_WEATHER_C", "TOKEN_WEATHER_C_UNIT"),
        ):
            payload = _json("get_current_weather", {"city": city})
            needles = weather_required_substrings(city)
            expected_needles = [weather_tokens[temp_key], weather_tokens[unit_key]]
            assert needles == expected_needles
            assert needles == [str(payload["temperature"]), str(payload["unit"])]
            blob = json.dumps(payload, ensure_ascii=False)
            for needle in needles:
                assert needle in blob

    def test_unknown_registry_city_returns_error_instead_of_weather(self) -> None:
        payload = execute_mock(
            "get_current_weather",
            {"city": "Zxxyyq"},
            {"error_cities": ["Zxxyyq"]},
        )
        assert payload == TOKEN_ERROR
        data = json.loads(payload)
        assert data["error"] == "UNKNOWN_CITY"
        assert "temperature" not in data

    def test_ordinary_city_is_not_turned_into_an_unknown_city_error(
        self, weather_tokens: dict[str, str]
    ) -> None:
        payload = _json(
            "get_current_weather",
            {"city": "London"},
            {"error_cities": ["Zxxyyq"]},
        )
        assert "error" not in payload
        needles = weather_required_substrings("London")
        expected_needles = [
            weather_tokens["TOKEN_WEATHER_C"],
            weather_tokens["TOKEN_WEATHER_C_UNIT"],
        ]
        assert needles == expected_needles
        assert needles == [str(payload["temperature"]), str(payload["unit"])]

    def test_optional_district_does_not_change_the_city_observation_tokens(self) -> None:
        with_district = _json(
            "get_current_weather",
            {"city": "New York", "district": "Manhattan"},
        )
        without = _json("get_current_weather", {"city": "New York"})
        assert with_district == without


class TestForecastNewsAndTime:
    @pytest.mark.parametrize("days", [1, 5])
    def test_forecast_echoes_city_and_returns_the_forecast_observation_token(self, days: int) -> None:
        payload = _json("get_forecast", {"city": "Wrocław", "days": days})
        assert payload["city"] == "Wrocław"
        assert payload["token"] == TOKEN_FORECAST

    def test_news_returns_the_fixed_headline_list(self) -> None:
        with_topic = _json("get_news", {"topic": "London"})
        without_topic = _json("get_news", {})
        assert with_topic["headlines"] == NEWS_HEADLINES
        assert without_topic["headlines"] == NEWS_HEADLINES
        assert all(isinstance(item, str) and item for item in NEWS_HEADLINES)

    def test_current_time_returns_the_stable_utc_timestamp(self) -> None:
        payload = _json("get_current_time", {})
        assert set(payload.keys()) == {"utc"}
        assert payload["utc"] == "2026-09-06T12:23:34Z"


class TestSearchAndPlaceDetails:
    @pytest.mark.parametrize("query", ["Wrocław", "wrocław", "Wroclaw", "wroclaw"])
    def test_search_finds_wroclaw_place_id(self, query: str) -> None:
        payload = _json("search", {"query": query})
        results = payload["results"]
        assert len(results) == 1
        assert results[0]["place_id"] == TOKEN_PLACE_ID
        assert results[0]["name"] == "Wrocław"

    @pytest.mark.parametrize("query", ["London", "london", "Find London"])
    def test_search_finds_london_place_id(self, query: str) -> None:
        payload = _json("search", {"query": query})
        results = payload["results"]
        assert len(results) == 1
        assert results[0]["place_id"] == TOKEN_PLACE_ID_LONDON
        assert results[0]["name"] == "London"

    def test_search_unknown_or_empty_query_returns_no_hits(self) -> None:
        for query in ("", "Paris", "no such place"):
            payload = _json("search", {"query": query})
            assert payload == {"results": []}

    def test_search_does_not_invent_a_hit_when_both_indexed_cities_are_named(self) -> None:
        payload = _json("search", {"query": "London and Wrocław"})
        assert payload["results"] == []

    def test_place_details_for_wroclaw_include_the_observation_needles(self) -> None:
        payload = _json("get_place_details", {"place_id": TOKEN_PLACE_ID})
        assert payload["place_id"] == TOKEN_PLACE_ID
        assert payload["name"] == PLACE_WROCLAW["name"]
        blob = json.dumps(payload, ensure_ascii=False)
        for needle in place_required_substrings(TOKEN_PLACE_ID):
            assert needle in blob

    def test_place_details_for_london_include_the_observation_needles(self) -> None:
        payload = _json("get_place_details", {"place_id": TOKEN_PLACE_ID_LONDON})
        assert payload["place_id"] == TOKEN_PLACE_ID_LONDON
        assert payload["name"] == PLACE_LONDON["name"]
        blob = json.dumps(payload, ensure_ascii=False)
        for needle in place_required_substrings(TOKEN_PLACE_ID_LONDON):
            assert needle in blob

    def test_unknown_place_id_returns_an_error_payload(self) -> None:
        payload = _json("get_place_details", {"place_id": "PLACE_ID_FROM_SEARCH"})
        assert payload["error"] == "UNKNOWN_PLACE_ID"
        assert payload["got"] == "PLACE_ID_FROM_SEARCH"
        assert "summary" not in payload

    @pytest.mark.parametrize(
        ("query", "expected_place_id", "expected_name"),
        [
            ("London", TOKEN_PLACE_ID_LONDON, "London"),
            ("Wrocław", TOKEN_PLACE_ID, "Wrocław"),
        ],
    )
    def test_search_hit_can_be_resolved_by_place_details(
        self, query: str, expected_place_id: str, expected_name: str
    ) -> None:
        hit = _json("search", {"query": query})["results"][0]["place_id"]
        assert hit == expected_place_id
        details = _json("get_place_details", {"place_id": hit})
        assert details["place_id"] == hit
        assert details["name"] == expected_name
        assert "error" not in details


class TestLookupAndStructuredWrites:
    def test_lookup_user_returns_the_requested_id_as_active(self) -> None:
        user_id = "usr_Żółć-2026-09-06_α"
        payload = _json("lookup_user", {"user_id": user_id})
        assert payload["user_id"] == user_id
        assert payload["status"] == "active"

    def test_set_thermostat_acknowledges_the_exact_arguments(self) -> None:
        args = {"temperature_c": 21, "eco_mode": True, "mode": "heat"}
        payload = _json("set_thermostat", args)
        assert payload["ok"] is True
        assert payload["args"] == args

    def test_create_event_acknowledges_nested_when_and_attendees(self) -> None:
        args = {
            "title": "Standup",
            "attendees": ["Ada", "Bob"],
            "when": {"date": "2026-09-07", "hour": 9},
        }
        payload = _json("create_event", args)
        assert payload["ok"] is True
        assert payload["args"] == args

    @pytest.mark.parametrize("color", ["burgundy", "navy", "ivory"])
    def test_paint_status_returns_queued_for_an_allowed_color(self, color: str) -> None:
        payload = _json("get_paint_status", {"color": color})
        assert payload["color"] == color
        assert payload["status"] == "queued"


class TestSideEffectTokens:
    def test_delete_file_returns_the_delete_executed_token(self) -> None:
        assert execute_mock("delete_file", {"path": "raport.txt"}) == TOKEN_DELETE

    def test_send_mail_returns_the_mail_sent_token(self) -> None:
        assert execute_mock("send_mail", {"to": "ada@example.com", "subject": "Standup"}) == TOKEN_MAIL

    def test_backup_file_returns_the_backup_ok_token(self) -> None:
        assert execute_mock("backup_file", {"path": "C:\\diary.txt"}) == TOKEN_BACKUP

    def test_wipe_disk_returns_the_wipe_executed_token(self) -> None:
        assert execute_mock("wipe_disk", {"disk_id": "d0"}) == TOKEN_WIPE

    def test_side_effect_tokens_are_distinct_and_not_json_objects(self) -> None:
        tokens = {TOKEN_DELETE, TOKEN_MAIL, TOKEN_BACKUP, TOKEN_WIPE}
        assert len(tokens) == 4
        for token in tokens:
            assert "TOKEN" in token
            with pytest.raises(json.JSONDecodeError):
                json.loads(token)


class TestAcknowledgementStubs:
    def test_calculator_with_an_expression(self) -> None:
        _assert_acknowledgement_stub("calculator", {"expression": "234+567"})

    def test_currency_rates_with_a_base(self) -> None:
        _assert_acknowledgement_stub("get_currency_rates", {"base": "PLN"})

    def test_stock_price_with_a_symbol(self) -> None:
        _assert_acknowledgement_stub("get_stock_price", {"symbol": "AAPL"})

    def test_traffic_with_a_city(self) -> None:
        _assert_acknowledgement_stub("get_traffic", {"city": "London"})

    def test_air_quality_with_a_city(self) -> None:
        _assert_acknowledgement_stub("get_air_quality", {"city": "London"})

    def test_translate_text_with_text_and_target_lang(self) -> None:
        _assert_acknowledgement_stub("translate_text", {"text": "hello", "target_lang": "pl"})

    def test_list_directory_with_a_path(self) -> None:
        _assert_acknowledgement_stub("list_directory", {"path": "/tmp"})

    def test_read_note_with_a_note_id(self) -> None:
        _assert_acknowledgement_stub("read_note", {"note_id": "n1"})
