"""Reference Chat Completions client with three local tools."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import urllib.request
from datetime import datetime, timezone


def get_current_time() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def calculate(a: int, b: int, operation: str) -> str:
    if operation == "+":
        return str(a + b)
    if operation == "-":
        return str(a - b)
    if operation == "*":
        return str(a * b)
    if operation == "/":
        if b == 0:
            return "division by zero"
        return str(a / b)
    return f"unknown operation {operation}"


def count_words(text: str) -> str:
    return str(len(text.split()))


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return the current UTC time as ISO-8601 with a trailing Z.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Integer a, integer b, operation one of + - * /.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                    "operation": {"type": "string", "enum": ["+", "-", "*", "/"]},
                },
                "required": ["a", "b", "operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_words",
            "description": "Count whitespace-separated words in text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
]

REGISTRY = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "count_words": count_words,
}


def run_tool_chat(base_url, model, messages, tools, registry) -> str:
    url = str(base_url).rstrip("/") + "/chat/completions"

    def post(msgs, include_tools: bool):
        body = {"model": model, "messages": msgs}
        if include_tools:
            body["tools"] = tools if isinstance(tools, list) else []
        req = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def response_text(data) -> str:
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            return ""
        first = choices[0] if isinstance(choices[0], dict) else {}
        msg = first.get("message")
        if not isinstance(msg, dict):
            return ""
        content = msg.get("content")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return str(content)

    msgs = copy.deepcopy(list(messages))
    data = post(msgs, True)
    choices = data.get("choices") if isinstance(data, dict) else None
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    msg = first.get("message") if isinstance(first.get("message"), dict) else {}
    calls = msg.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return response_text(data)
    msgs.append(msg)
    for call in calls:
        if not isinstance(call, dict):
            msgs.append({"role": "tool", "tool_call_id": None, "content": "arguments is not a JSON object"})
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = fn.get("name")
        call_id = call.get("id")
        parse_error = None
        args = None
        if "arguments" not in fn:
            args = {}
        else:
            raw = fn["arguments"]
            if raw is None:
                parse_error = "arguments is not a JSON object"
            elif isinstance(raw, str):
                try:
                    args = json.loads(raw)
                except json.JSONDecodeError as exc:
                    parse_error = str(exc)
            elif isinstance(raw, dict):
                args = raw
            else:
                parse_error = "arguments is not a JSON object"
        if parse_error is None and not isinstance(args, dict):
            parse_error = "arguments is not a JSON object"
        if parse_error is not None:
            content = parse_error
        elif not isinstance(registry, dict) or name not in registry:
            content = f"unknown tool {name}"
        else:
            try:
                content = registry[name](**args)
            except Exception as exc:
                content = str(exc)
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": str(content),
            }
        )
    data2 = post(msgs, False)
    return response_text(data2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--message",
        default="What is the current UTC time? What is 7 * 6?",
    )
    args = parser.parse_args()
    messages = [{"role": "user", "content": args.message}]
    try:
        text = run_tool_chat(args.base_url, args.model, messages, TOOLS, REGISTRY)
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    print(text)


if __name__ == "__main__":
    main()
