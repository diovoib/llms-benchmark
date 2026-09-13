from __future__ import annotations

import json
from typing import Any

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


def execute_mock(name: str, arguments: dict[str, Any] | Any, extra: dict[str, Any] | None = None) -> str:
    args, parsed = parse_arguments(arguments)
    extra = extra or {}
    if not parsed or args is None:
        return _dump({"error": "UNPARSED_ARGUMENTS"})
    if name == "get_current_weather":
        city = str(args.get("city", ""))
        if city in set(extra.get("error_cities") or []):
            return TOKEN_ERROR
        return _dump(weather_payload(city))
    if name == "get_forecast":
        return _dump({"token": TOKEN_FORECAST, "city": args.get("city")})
    if name == "get_current_time":
        return _dump({"utc": "2026-09-06T12:00:00Z"})
    if name == "get_news":
        return _dump({"headlines": list(NEWS_HEADLINES)})
    if name == "search":
        hit = _place_hit(str(args.get("query") or ""))
        if hit is None:
            return _dump({"results": []})
        pid, label = hit
        return _dump({"results": [{"place_id": pid, "name": label}]})
    if name == "get_place_details":
        pid = str(args.get("place_id", ""))
        details = PLACE_BY_ID.get(pid)
        if details is None:
            return _dump({"error": "UNKNOWN_PLACE_ID", "got": pid})
        return _dump(details)
    if name == "lookup_user":
        return _dump({"user_id": args.get("user_id"), "status": "active"})
    if name == "delete_file":
        path = args.get("path")
        if not _is_file_path(path):
            return _dump({"error": "NOT_A_FILE", "got": path})
        return TOKEN_DELETE
    if name == "send_mail":
        return TOKEN_MAIL
    if name == "backup_file":
        return TOKEN_BACKUP
    if name == "wipe_disk":
        return TOKEN_WIPE
    if name == "set_thermostat":
        return _dump({"ok": True, "args": args})
    if name == "create_event":
        return _dump({"ok": True, "args": args})
    if name == "get_paint_status":
        return _dump({"color": args.get("color"), "status": "queued"})
    return _dump({"ok": True, "tool": name, "args": args})
