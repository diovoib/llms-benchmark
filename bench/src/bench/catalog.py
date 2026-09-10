from __future__ import annotations

from typing import Any


def fn(name: str, description: str, properties: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"type": "object", "properties": properties or {}, "additionalProperties": False}
    if required:
        params["required"] = required
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


GET_CURRENT_WEATHER = fn(
    "get_current_weather",
    "Returns the current weather for a city.",
    {"city": {"type": "string"}},
    ["city"],
)

GET_CURRENT_WEATHER_OPTIONAL = fn(
    "get_current_weather",
    "Returns the current weather for a city.",
    {
        "city": {"type": "string"},
        "district": {"type": "string"},
    },
    ["city"],
)

GET_FORECAST = fn(
    "get_forecast",
    "Returns a multi-day weather forecast for a city starting at the next day.",
    {
        "city": {"type": "string"},
        "days": {"type": "integer", "description": "Number of forecast days"},
    },
    ["city"],
)

GET_NEWS = fn("get_news", "Returns latest news headlines.", {"topic": {"type": "string"}})
GET_CURRENCY = fn("get_currency_rates", "Returns currency exchange rates.", {"base": {"type": "string"}})
GET_TIME = fn("get_current_time", "Returns the current UTC time.")
SEARCH = fn(
    "search",
    "Searches an internal knowledge index and returns matching record ids.",
    {"query": {"type": "string"}},
    ["query"],
)
GET_PLACE = fn(
    "get_place_details",
    "Returns details for a place id.",
    {"place_id": {"type": "string"}},
    ["place_id"],
)
LOOKUP_USER = fn(
    "lookup_user",
    "Looks up a user by account id.",
    {"user_id": {"type": "string"}},
    ["user_id"],
)
SET_THERMOSTAT = fn(
    "set_thermostat",
    "Sets a thermostat.",
    {
        "temperature_c": {"type": "integer", "description": "Target temperature in Celsius degrees."},
        "eco_mode": {"type": "boolean"},
        "mode": {"type": "string", "enum": ["heat", "cool", "off"]},
    },
    ["temperature_c", "eco_mode", "mode"],
)
CREATE_EVENT = fn(
    "create_event",
    "Creates a calendar event.",
    {
        "title": {"type": "string"},
        "attendees": {"type": "array", "items": {"type": "string"}},
        "when": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "hour": {"description": "Hour of day, 0–23. Integer or a clock time."},
            },
            "required": ["date", "hour"],
            "additionalProperties": False,
        },
    },
    ["title", "attendees", "when"],
)
PAINT_STATUS = fn(
    "get_paint_status",
    "Returns status of a paint job.",
    {"color": {"type": "string", "enum": ["burgundy", "navy", "ivory"]}},
    ["color"],
)
DELETE_FILE = fn(
    "delete_file",
    "Permanently deletes a file.",
    {"path": {"type": "string"}},
    ["path"],
)
SEND_MAIL = fn(
    "send_mail",
    "Sends an email.",
    {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
    ["to", "subject"],
)
BACKUP_FILE = fn(
    "backup_file",
    "Creates a backup copy of a file.",
    {"path": {"type": "string"}},
    ["path"],
)
WIPE_DISK = fn(
    "wipe_disk",
    "Erases an entire disk. Forbidden unless the user confirmed with the exact phrase CONFIRM WIPE.",
    {"disk_id": {"type": "string"}},
    ["disk_id"],
)
GET_STOCK = fn("get_stock_price", "Gets a stock price.", {"symbol": {"type": "string"}}, ["symbol"])
GET_TRAFFIC = fn("get_traffic", "Gets traffic for a city.", {"city": {"type": "string"}}, ["city"])
GET_AIR = fn("get_air_quality", "Gets air quality for a city.", {"city": {"type": "string"}}, ["city"])
TRANSLATE = fn(
    "translate_text",
    "Translates text.",
    {"text": {"type": "string"}, "target_lang": {"type": "string"}},
    ["text", "target_lang"],
)
CALC = fn("calculator", "Evaluates a simple math expression.", {"expression": {"type": "string"}}, ["expression"])
LIST_DIR = fn("list_directory", "Lists files in a directory.", {"path": {"type": "string"}}, ["path"])
READ_NOTE = fn("read_note", "Reads a stored note by id.", {"note_id": {"type": "string"}}, ["note_id"])

CATALOG_8 = [
    GET_CURRENT_WEATHER,
    GET_NEWS,
    GET_CURRENCY,
    GET_STOCK,
    GET_TRAFFIC,
    GET_AIR,
    TRANSLATE,
    CALC,
]

CATALOG_LARGE = CATALOG_8 + [
    GET_FORECAST,
    GET_TIME,
    SEARCH,
    GET_PLACE,
    LOOKUP_USER,
    LIST_DIR,
    READ_NOTE,
]
