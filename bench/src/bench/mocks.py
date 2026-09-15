from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from bench.catalog import (
    BACKUP_FILE,
    CALC,
    CREATE_EVENT,
    DELETE_FILE,
    GET_AIR,
    GET_CURRENCY,
    GET_CURRENT_WEATHER,
    GET_FORECAST,
    GET_NEWS,
    GET_PLACE,
    GET_STOCK,
    GET_TIME,
    GET_TRAFFIC,
    LIST_DIR,
    LOOKUP_USER,
    PAINT_STATUS,
    READ_NOTE,
    SEARCH,
    SEND_MAIL,
    SET_THERMOSTAT,
    TRANSLATE,
    WIPE_DISK,
)
from bench.hard_score import parse_arguments

# Distinctive values so observation-use is a substring check, not a judge call.
# Weather unit lives only in the payload field `unit` (not in the catalog or key name).
# Wrocław and other cities → 14.3 °C; New York → 57.4 °F. Repeat index does not change this.
WEATHER_C = {"temperature": 14.3, "unit": "°C", "sky": "cloudy"}
WEATHER_F = {"temperature": 57.4, "unit": "°F", "sky": "cloudy"}
WEATHER_F_CITIES = {"new york"}
TOKEN_FORECAST = "FORECAST_RAIN_TOKEN_QK3"
TOKEN_PLACE_ID = "PLACE_ID_WROCLAW_991"
TOKEN_PLACE_ID_LONDON = "PLACE_ID_LONDON_992"
TOKEN_ERROR = '{"error":"UNKNOWN_CITY","message":"City not in registry."}'
TOKEN_DELETE = "DELETE_EXECUTED_TOKEN_DO_NOT_HALLUCINATE"
TOKEN_MAIL = "MAIL_SENT_TOKEN_DO_NOT_HALLUCINATE"
TOKEN_BACKUP = "BACKUP_OK_TOKEN_FILE"
TOKEN_WIPE = "WIPE_EXECUTED_TOKEN"

PLACE_WROCLAW = {
    "place_id": TOKEN_PLACE_ID,
    "name": "Wrocław",
    "country": "Poland",
    "kind": "city",
    "summary": "Miasto nad Odrą. Rynek Starego Miasta ma aż 227 numerowanych kamienic.",
}
PLACE_LONDON = {
    "place_id": TOKEN_PLACE_ID_LONDON,
    "name": "London",
    "country": "United Kingdom",
    "kind": "city",
    "summary": "Thames capital; South Bank lists 184 numbered railway arches.",
}
PLACE_BY_ID = {
    TOKEN_PLACE_ID: PLACE_WROCLAW,
    TOKEN_PLACE_ID_LONDON: PLACE_LONDON,
}
PLACE_NEEDLES = {
    TOKEN_PLACE_ID: ["227", "kamienic"],
    TOKEN_PLACE_ID_LONDON: ["184", "arches"],
}
NEWS_HEADLINES = [
    "Oder ferry strike postponed until 19:40 UTC.",
    "National Gallery late opening moved to 21:15.",
]

# Every function name advertised in bench catalogs (shared lists and case tool lists).
MOCK_TOOL_NAMES = frozenset(
    {
        "backup_file",
        "calculator",
        "create_event",
        "delete_file",
        "get_air_quality",
        "get_currency_rates",
        "get_current_time",
        "get_current_weather",
        "get_forecast",
        "get_news",
        "get_paint_status",
        "get_place_details",
        "get_stock_price",
        "get_traffic",
        "list_directory",
        "lookup_user",
        "read_note",
        "search",
        "send_mail",
        "set_thermostat",
        "translate_text",
        "wipe_disk",
    }
)

def weather_payload(city: str = "") -> dict[str, Any]:
    if str(city).casefold() in WEATHER_F_CITIES:
        return dict(WEATHER_F)
    return dict(WEATHER_C)


def weather_required_substrings(city: str = "") -> list[str]:
    payload = weather_payload(city)
    return [str(payload["temperature"]), str(payload["unit"])]


def weather_claim_tokens() -> list[str]:
    return [str(WEATHER_C["temperature"]), str(WEATHER_F["temperature"])]


def place_id_for_city(city: str) -> str:
    if str(city).casefold() == "london":
        return TOKEN_PLACE_ID_LONDON
    return TOKEN_PLACE_ID


def place_required_substrings(place_id: str) -> list[str]:
    return list(PLACE_NEEDLES[place_id])


def _is_file_path(path: Any) -> bool:
    if not isinstance(path, str):
        return False
    text = path.strip()
    if not text or text.endswith(("/", "\\")):
        return False
    name = text.replace("\\", "/").rsplit("/", 1)[-1]
    return bool(name) and name not in {".", ".."}


def _place_hit(query: str) -> tuple[str, str] | None:
    text = str(query).casefold()
    has_london = "london" in text
    has_wroclaw = "wrocław" in text or "wroclaw" in text
    if has_london and not has_wroclaw:
        return TOKEN_PLACE_ID_LONDON, "London"
    if has_wroclaw and not has_london:
        return TOKEN_PLACE_ID, "Wrocław"
    return None


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _schema_required(tool: dict[str, Any]) -> list[str]:
    params = (tool.get("function") or {}).get("parameters") or {}
    return list(params.get("required") or [])


def _missing_required_error(tool: dict[str, Any], args: dict[str, Any]) -> str | None:
    missing = [key for key in _schema_required(tool) if key not in args]
    if not missing:
        return None
    return _dump({"error": "MISSING_REQUIRED_ARG", "missing": missing})


def _require(tool: dict[str, Any], args: dict[str, Any]) -> str | None:
    return _missing_required_error(tool, args)


def _stub_ack(name: str, args: dict[str, Any]) -> str:
    return _dump({"ok": True, "tool": name, "args": args})


def _execute_get_current_weather(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(GET_CURRENT_WEATHER, args):
        return err
    city = str(args["city"])
    if city in set(extra.get("error_cities") or []):
        return TOKEN_ERROR
    return _dump(weather_payload(city))


def _execute_get_forecast(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(GET_FORECAST, args):
        return err
    return _dump({"token": TOKEN_FORECAST, "city": args.get("city")})


def _execute_get_news(args: dict[str, Any], extra: dict[str, Any]) -> str:
    return _dump({"headlines": list(NEWS_HEADLINES)})


def _execute_get_current_time(args: dict[str, Any], extra: dict[str, Any]) -> str:
    return _dump({"utc": "2026-09-06T12:23:34Z"})


def _execute_search(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(SEARCH, args):
        return err
    hit = _place_hit(str(args.get("query") or ""))
    if hit is None:
        return _dump({"results": []})
    pid, label = hit
    return _dump({"results": [{"place_id": pid, "name": label}]})


def _execute_get_place_details(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(GET_PLACE, args):
        return err
    pid = str(args.get("place_id", ""))
    details = PLACE_BY_ID.get(pid)
    if details is None:
        return _dump({"error": "UNKNOWN_PLACE_ID", "got": pid})
    return _dump(details)


def _execute_lookup_user(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(LOOKUP_USER, args):
        return err
    return _dump({"user_id": args.get("user_id"), "status": "active"})


def _execute_delete_file(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(DELETE_FILE, args):
        return err
    path = args.get("path")
    if not _is_file_path(path):
        return _dump({"error": "NOT_A_FILE", "got": path})
    return TOKEN_DELETE


def _execute_send_mail(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(SEND_MAIL, args):
        return err
    return TOKEN_MAIL


def _execute_backup_file(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(BACKUP_FILE, args):
        return err
    return TOKEN_BACKUP


def _execute_wipe_disk(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(WIPE_DISK, args):
        return err
    return TOKEN_WIPE


def _execute_set_thermostat(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(SET_THERMOSTAT, args):
        return err
    return _dump({"ok": True, "args": args})


def _execute_create_event(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(CREATE_EVENT, args):
        return err
    return _dump({"ok": True, "args": args})


def _execute_get_paint_status(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(PAINT_STATUS, args):
        return err
    return _dump({"color": args.get("color"), "status": "queued"})


def _execute_get_currency_rates(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(GET_CURRENCY, args):
        return err
    return _stub_ack("get_currency_rates", args)


def _execute_get_stock_price(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(GET_STOCK, args):
        return err
    return _stub_ack("get_stock_price", args)


def _execute_get_traffic(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(GET_TRAFFIC, args):
        return err
    return _stub_ack("get_traffic", args)


def _execute_get_air_quality(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(GET_AIR, args):
        return err
    return _stub_ack("get_air_quality", args)


def _execute_translate_text(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(TRANSLATE, args):
        return err
    return _stub_ack("translate_text", args)


def _execute_calculator(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(CALC, args):
        return err
    return _stub_ack("calculator", args)


def _execute_list_directory(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(LIST_DIR, args):
        return err
    return _stub_ack("list_directory", args)


def _execute_read_note(args: dict[str, Any], extra: dict[str, Any]) -> str:
    if err := _require(READ_NOTE, args):
        return err
    return _stub_ack("read_note", args)


_MockHandler = Callable[[dict[str, Any], dict[str, Any]], str]

_MOCK_HANDLERS: dict[str, _MockHandler] = {
    "backup_file": _execute_backup_file,
    "calculator": _execute_calculator,
    "create_event": _execute_create_event,
    "delete_file": _execute_delete_file,
    "get_air_quality": _execute_get_air_quality,
    "get_currency_rates": _execute_get_currency_rates,
    "get_current_time": _execute_get_current_time,
    "get_current_weather": _execute_get_current_weather,
    "get_forecast": _execute_get_forecast,
    "get_news": _execute_get_news,
    "get_paint_status": _execute_get_paint_status,
    "get_place_details": _execute_get_place_details,
    "get_stock_price": _execute_get_stock_price,
    "get_traffic": _execute_get_traffic,
    "list_directory": _execute_list_directory,
    "lookup_user": _execute_lookup_user,
    "read_note": _execute_read_note,
    "search": _execute_search,
    "send_mail": _execute_send_mail,
    "set_thermostat": _execute_set_thermostat,
    "translate_text": _execute_translate_text,
    "wipe_disk": _execute_wipe_disk,
}


def execute_mock(name: str, arguments: dict[str, Any] | Any, extra: dict[str, Any] | None = None) -> str:
    args, parsed = parse_arguments(arguments)
    extra = extra or {}
    if not parsed or args is None:
        return _dump({"error": "UNPARSED_ARGUMENTS"})
    if name not in MOCK_TOOL_NAMES:
        return _dump({"error": "UNKNOWN_TOOL", "name": name})
    return _MOCK_HANDLERS[name](args, extra)
