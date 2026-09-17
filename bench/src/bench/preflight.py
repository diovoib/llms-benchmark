from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from bench.catalog import GET_NEWS, GET_TIME
from bench.client import BenchClient
from bench.template_dialect import (
    RenderOutcome,
    TemplateRenderer,
    probe_template_dialect,
)


LONG_TOOL = {
    "type": "function",
    "function": {
        "name": "preflight_marker_tool",
        "description": (
            "PREFLIGHT_ONLY " + ("MARKER_TOKEN_TOOLS_MUST_AFFECT_PROMPT " * 40)
        ),
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}


def _root(base_url: str) -> str:
    u = base_url.rstrip("/")
    if u.endswith("/v1"):
        return u[: -len("/v1")]
    return u


def _get_json(
    url: str,
    api_key: str | None = None,
    *,
    client: BenchClient | None = None,
    timeout_s: float,
) -> tuple[int | None, Any]:
    if client is not None:
        client.raise_if_interrupted()
    try:
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                status, body = resp.status, json.loads(raw)
            except json.JSONDecodeError:
                status, body = resp.status, raw
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except Exception as exc:
        if client is not None:
            client.raise_if_interrupted()
        return None, str(exc)
    if client is not None:
        client.raise_if_interrupted()
    return status, body


def _post_json(
    url: str,
    body: dict[str, Any],
    api_key: str | None = None,
    *,
    client: BenchClient | None = None,
    timeout_s: float,
) -> tuple[int | None, Any]:
    if client is not None:
        client.raise_if_interrupted()
    data = json.dumps(body).encode("utf-8")
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url, data=data, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                status, payload = resp.status, json.loads(raw)
            except json.JSONDecodeError:
                status, payload = resp.status, raw
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except HTTPError as exc:
        if client is not None:
            client.raise_if_interrupted()
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw
    except URLError as exc:
        if client is not None:
            client.raise_if_interrupted()
        return None, str(exc.reason if getattr(exc, "reason", None) else exc)
    except Exception as exc:
        if client is not None:
            client.raise_if_interrupted()
        return None, str(exc)
    if client is not None:
        client.raise_if_interrupted()
    return status, payload


def _error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
        if payload.get("message"):
            return str(payload["message"])
    return str(payload)


def _default_generation_temperature(props: Any) -> float | None:
    if not isinstance(props, dict):
        return None
    dgs = props.get("default_generation_settings")
    if not isinstance(dgs, dict):
        return None
    params = dgs.get("params")
    blob = params if isinstance(params, dict) else dgs
    raw = blob.get("temperature")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def collect_server_info(
    base_url: str,
    api_key: str | None = None,
    *,
    client: BenchClient | None = None,
    timeout_s: float,
) -> dict[str, Any]:
    root = _root(base_url)
    health_status, health = _get_json(f"{root}/health", api_key, client=client, timeout_s=timeout_s)
    props_status, props = _get_json(f"{root}/props", api_key, client=client, timeout_s=timeout_s)
    models_status, models = _get_json(f"{base_url.rstrip('/')}/models", api_key, client=client, timeout_s=timeout_s)
    chat_format = None
    build = None
    if isinstance(props, dict):
        build = props.get("build_info") or props.get("version")
        dgs = props.get("default_generation_settings") or {}
        params = dgs.get("params") or dgs
        chat_format = (
            params.get("chat_template")
            or params.get("chat_format")
            or props.get("chat_template")
            or props.get("chat_format")
        )
        if isinstance(chat_format, str) and len(chat_format) > 200:
            chat_format = chat_format[:200] + "…"
    return {
        "server_root": root,
        "health": {"status": health_status, "body": health},
        "props": {"status": props_status, "body": props if not isinstance(props, dict) else {k: props[k] for k in list(props)[:20]}},
        "models": {"status": models_status, "body": models},
        "chat_format": chat_format,
        "build": build,
        "default_temperature": _default_generation_temperature(props),
        "host": urlparse(root).netloc,
    }


def _make_renderer(client: BenchClient, common: dict[str, Any]) -> TemplateRenderer:
    root = _root(client.base_url)
    apply_urls = [
        f"{root}/apply-template",
        f"{client.base_url.rstrip('/')}/apply-template",
    ]

    def apply_template(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> RenderOutcome:
        body: dict[str, Any] = {
            "model": client.model,
            "messages": messages,
            "tools": tools,
        }
        kwargs = common.get("chat_template_kwargs")
        if kwargs:
            body["chat_template_kwargs"] = kwargs
        last_error = None
        saw_missing = False
        for url in apply_urls:
            status, payload = _post_json(
                url,
                body,
                client.api_key,
                client=client,
                timeout_s=float(common["request_timeout_s"]),
            )
            if status in (404, 405) or status is None:
                saw_missing = True
                last_error = _error_message(payload)
                continue
            if isinstance(payload, dict) and isinstance(payload.get("prompt"), str):
                return RenderOutcome(ok=True, prompt=payload["prompt"], via="apply-template")
            if status == 200:
                return RenderOutcome(ok=True, prompt=str(payload), via="apply-template")
            err = _error_message(payload)
            low = err.lower()
            if status in (400, 422) and ("not found" in low or "unknown path" in low or "no such" in low):
                saw_missing = True
                last_error = err
                continue
            return RenderOutcome(ok=False, error=err, via="apply-template")
        return RenderOutcome(ok=False, error=last_error, via="apply-template", missing=saw_missing)

    def chat(messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, max_tokens: int) -> RenderOutcome:
        result = client.chat(messages, tools=tools, max_tokens=max_tokens, **common)
        if result.ok:
            return RenderOutcome(
                ok=True,
                prompt=None,
                prompt_tokens=result.prompt_tokens,
                via="chat",
            )
        return RenderOutcome(ok=False, error=result.error, prompt_tokens=result.prompt_tokens, via="chat")

    return TemplateRenderer(apply_template=apply_template, chat=chat)


def run_preflight(
    client: BenchClient,
    sampler: dict[str, Any],
    *,
    request_timeout_s: float,
    max_tokens: int,
) -> dict[str, Any]:
    info = collect_server_info(
        client.base_url,
        getattr(client, "api_key", None),
        client=client,
        timeout_s=request_timeout_s,
    )
    client.raise_if_interrupted()
    messages = [{"role": "user", "content": "Reply with the single word pong."}]
    common = dict(
        temperature=0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repeat_penalty=1.0,
        seed=1,
        chat_template_kwargs=sampler.get("chat_template_kwargs"),
        stream=False,
        request_timeout_s=request_timeout_s,
    )
    without = client.chat(messages, max_tokens=max_tokens, **common)
    with_tools = client.chat(messages, tools=[LONG_TOOL, GET_TIME], max_tokens=max_tokens, **common)
    jinja_required = False
    tools_dropped = False
    usage_missing = False
    ok = True
    reasons: list[str] = []

    for label, res in (("without_tools", without), ("with_tools", with_tools)):
        if not res.ok and res.http_status == 500 and res.error and "jinja" in res.error.lower():
            jinja_required = True
            ok = False
            reasons.append(f"{label}: tools require --jinja")
        elif not res.ok:
            ok = False
            reasons.append(f"{label}: {res.infra_code} {res.error}")

    pt_wo = without.prompt_tokens
    pt_w = with_tools.prompt_tokens
    if without.ok and with_tools.ok:
        if pt_wo is None or pt_w is None:
            usage_missing = True
            ok = False
            reasons.append("usage.prompt_tokens missing; cannot detect silent tools drop")
        elif int(pt_w) <= int(pt_wo):
            tools_dropped = True
            ok = False
            reasons.append(f"silent tools drop: prompt_tokens with={pt_w} without={pt_wo}")

    parallel_confirmed = False
    parallel_note = None
    par = client.chat(
        [{"role": "user", "content": "I need the current time and the latest news, both in this same reply."}],
        tools=[GET_TIME, GET_NEWS],
        parallel_tool_calls=True,
        max_tokens=max_tokens,
        **common,
    )
    if not par.ok:
        parallel_note = par.error
    else:
        n_calls = len(par.tool_calls)
        parallel_confirmed = n_calls >= 2
        parallel_note = f"preflight parallel tool_calls count={n_calls}"

    issued_ids = [str(c.get("id")) for c in par.tool_calls if c.get("id")]
    dialect: dict[str, Any]
    if not ok or tools_dropped:
        dialect = {
            "ok": False,
            "skipped": True,
            "note": "tools not injected; history dialect not probed",
        }
    else:
        dialect = probe_template_dialect(
            _make_renderer(client, common),
            tools=[GET_NEWS],
            issued_ids=issued_ids,
        )

    return {
        "ok": ok,
        "reasons": reasons,
        "jinja_required": jinja_required,
        "tools_dropped": tools_dropped,
        "usage_missing": usage_missing,
        "prompt_tokens_without_tools": pt_wo,
        "prompt_tokens_with_tools": pt_w,
        "parallel_tool_calls_supported": parallel_confirmed,
        "parallel_tool_calls_confirmed": parallel_confirmed,
        "parallel_note": parallel_note,
        "template_dialect": dialect,
        "server": info,
        "thinking": sampler.get("thinking"),
        "chat_template_kwargs": sampler.get("chat_template_kwargs"),
    }
