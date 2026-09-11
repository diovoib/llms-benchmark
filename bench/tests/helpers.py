from __future__ import annotations

import json
from typing import Any

from bench.catalog import CATALOG_8, CATALOG_LARGE
from bench.client import ChatResult
from bench.hard_score import parse_arguments
from bench.mocks import TOKEN_PLACE_ID_LONDON
from bench.suites.cases import all_cases


def openai_tool_call(name: str, arguments: Any, call_id: str = "call_1") -> dict[str, Any]:
    if isinstance(arguments, dict):
        raw = json.dumps(arguments, ensure_ascii=False)
    elif arguments is None:
        raw = ""
    else:
        raw = str(arguments)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": raw},
    }


def normalized_call(name: str, arguments: Any, call_id: str = "call_1") -> dict[str, Any]:
    if isinstance(arguments, dict):
        raw = json.dumps(arguments, ensure_ascii=False)
        parsed_args, parsed = arguments, True
    else:
        raw = arguments
        parsed_args, parsed = parse_arguments(arguments)
    return {
        "id": call_id,
        "name": name,
        "arguments": parsed_args,
        "arguments_parsed": parsed,
        "arguments_raw": raw,
    }


def chat_ok(
    *,
    content: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> ChatResult:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        if not content and finish_reason == "stop":
            finish_reason = "tool_calls"
    return ChatResult(ok=True, message=message, finish_reason=finish_reason)


def chat_infra(code: str = "INFRA_ERROR", error: str = "HTTP 500: server_error") -> ChatResult:
    return ChatResult(ok=False, infra_code=code, error=error)


def agent_step(
    *,
    ok: bool = True,
    infra_code: str | None = None,
    error: str | None = None,
    finish_reason: str | None = "stop",
    content: str = "",
    normalized: list[dict[str, Any]] | None = None,
    tool_payloads: list[str] | None = None,
    score: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "result": {
            "ok": ok,
            "infra_code": infra_code,
            "error": error,
            "finish_reason": finish_reason,
        },
        "content": content,
        "normalized": list(normalized or []),
        "score": dict(score or {}),
        "tool_payloads": list(tool_payloads or []),
    }


def function_name(tool: dict[str, Any]) -> str:
    return str((tool.get("function") or {}).get("name") or "")


def schema_required(tool: dict[str, Any]) -> list[str]:
    params = (tool.get("function") or {}).get("parameters") or {}
    return list(params.get("required") or [])


def schema_properties(tool: dict[str, Any]) -> dict[str, Any]:
    params = (tool.get("function") or {}).get("parameters") or {}
    return dict(params.get("properties") or {})


def advertised_tool_schemas() -> dict[str, dict[str, Any]]:
    """Unique function schemas the model can see: shared catalogs plus every case list."""
    out: dict[str, dict[str, Any]] = {}
    for tool in [*CATALOG_8, *CATALOG_LARGE]:
        name = function_name(tool)
        if name:
            out[name] = tool
    for case in all_cases().values():
        for tool in case.get("tools") or []:
            name = function_name(tool)
            if name and name not in out:
                out[name] = tool
    return out


def names_in_case_catalogs() -> set[str]:
    names: set[str] = set()
    for case in all_cases().values():
        for tool in case.get("tools") or []:
            name = function_name(tool)
            if name:
                names.add(name)
    return names


def tools_with_required_fields() -> list[str]:
    return sorted(
        name
        for name, schema in advertised_tool_schemas().items()
        if schema_required(schema)
    )


VALID_INVOCATIONS: dict[str, dict[str, Any]] = {
    "get_current_weather": {"city": "London"},
    "get_forecast": {"city": "London", "days": 3},
    "get_news": {"topic": "London"},
    "get_currency_rates": {"base": "PLN"},
    "get_current_time": {},
    "search": {"query": "London"},
    "get_place_details": {"place_id": TOKEN_PLACE_ID_LONDON},
    "lookup_user": {"user_id": "usr_1"},
    "set_thermostat": {"temperature_c": 21, "eco_mode": True, "mode": "heat"},
    "create_event": {
        "title": "Standup",
        "attendees": ["Ada", "Bob"],
        "when": {"date": "2026-09-07", "hour": 9},
    },
    "get_paint_status": {"color": "navy"},
    "delete_file": {"path": "raport.txt"},
    "send_mail": {"to": "ada@example.com", "subject": "Standup"},
    "backup_file": {"path": "C:\\diary.txt"},
    "wipe_disk": {"disk_id": "d0"},
    "get_stock_price": {"symbol": "AAPL"},
    "get_traffic": {"city": "London"},
    "get_air_quality": {"city": "London"},
    "translate_text": {"text": "hello", "target_lang": "pl"},
    "calculator": {"expression": "234+567"},
    "list_directory": {"path": "/tmp"},
    "read_note": {"note_id": "n1"},
}
