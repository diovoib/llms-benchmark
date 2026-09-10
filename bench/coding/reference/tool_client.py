"""Reference OpenAI-compatible tool client."""

from __future__ import annotations

import json
import urllib.request


def run_tool_chat(base_url, model, messages, tools, registry) -> str:
    url = str(base_url).rstrip("/") + "/chat/completions"

    def post(msgs, include_tools: bool):
        body = {"model": model, "messages": msgs}
        if include_tools and tools:
            body["tools"] = tools
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    data = post(list(messages), True)
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    calls = msg.get("tool_calls") or []
    if not calls:
        return msg.get("content") or ""
    msgs = list(messages)
    msgs.append(msg)
    for call in calls:
        fn = call.get("function") or {}
        name = fn.get("name")
        raw = fn.get("arguments") or "{}"
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        impl = (registry or {}).get(name)
        try:
            content = impl(**args) if impl else f"unknown tool {name}"
        except Exception as exc:
            content = str(exc)
        msgs.append({"role": "tool", "tool_call_id": call.get("id"), "content": str(content)})
    data2 = post(msgs, False)
    return ((data2.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
