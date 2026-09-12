from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx


INFRA_CODES = frozenset({"INFRA_ERROR", "CONTEXT_OVERFLOW"})
GENERATION_STALL_CAP_S = 5.0


def generation_stall_s(request_timeout_s: float) -> float:
    return min(float(request_timeout_s) / 2.0, GENERATION_STALL_CAP_S)


def deadline_abort_code(
    *,
    stream: bool,
    last_content_monotonic: float | None,
    now_monotonic: float,
    request_timeout_s: float,
) -> str:
    if not stream:
        return "INFRA_ERROR"
    if last_content_monotonic is None:
        return "INFRA_ERROR"
    if (now_monotonic - last_content_monotonic) > generation_stall_s(request_timeout_s):
        return "INFRA_ERROR"
    return "CASE_GENERATION_TIMEOUT"

OnWire = Callable[[str, bytes], None]


@dataclass
class ChatResult:
    ok: bool
    infra_code: str | None = None
    error: str | None = None
    message: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    raw: dict[str, Any] | None = None
    http_status: int | None = None

    @property
    def content(self) -> str:
        value = self.message.get("content")
        return value if isinstance(value, str) else ("" if value is None else str(value))

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        calls = self.message.get("tool_calls") or []
        return list(calls) if isinstance(calls, list) else []

    def to_message(self) -> dict[str, Any]:
        msg = {"role": "assistant", "content": self.message.get("content")}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


def _overflow_blob(status: int | None, body: str) -> bool:
    blob = f"{status} {body}".lower()
    return any(
        s in blob
        for s in ("context size", "n_ctx", "context length", "too many tokens", "context overflow")
    )


def _chat_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _build_request_body(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    parallel_tool_calls: bool | None,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repeat_penalty: float,
    seed: int | None,
    chat_template_kwargs: dict[str, Any] | None,
    max_tokens: int | None,
    stream: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repeat_penalty": repeat_penalty,
    }
    if seed is not None:
        body["seed"] = seed
    if tools:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if parallel_tool_calls is not None:
        body["parallel_tool_calls"] = parallel_tool_calls
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if chat_template_kwargs:
        body["chat_template_kwargs"] = chat_template_kwargs
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return body


def _new_sse_state() -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [],
        "finish_reason": None,
        "usage": {},
        "last_content_monotonic": None,
    }


def _apply_chunk(
    state: dict[str, Any],
    obj: dict[str, Any],
    *,
    now: float | None = None,
) -> None:
    usage = obj.get("usage")
    if isinstance(usage, dict) and usage:
        state["usage"] = usage
    choices = obj.get("choices") or []
    if not choices:
        return
    ch = choices[0] or {}
    finish = ch.get("finish_reason")
    if finish:
        state["finish_reason"] = finish
    delta = ch.get("delta") or ch.get("message") or {}
    if not isinstance(delta, dict):
        return
    piece = delta.get("content")
    stamped = False
    if isinstance(piece, str) and piece:
        state["content"] += piece
        stamped = True
    for tc in delta.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        idx = int(tc.get("index") or 0)
        slots: list[dict[str, Any]] = state["tool_calls"]
        while len(slots) <= idx:
            slots.append(
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            )
        slot = slots[idx]
        if tc.get("id"):
            slot["id"] = tc["id"]
            stamped = True
        if tc.get("type"):
            slot["type"] = tc["type"]
        fn = tc.get("function") or {}
        if isinstance(fn, dict):
            if fn.get("name"):
                slot["function"]["name"] = (slot["function"].get("name") or "") + str(fn["name"])
                stamped = True
            if fn.get("arguments") is not None:
                arg = str(fn["arguments"])
                slot["function"]["arguments"] = (slot["function"].get("arguments") or "") + arg
                if arg:
                    stamped = True
    if stamped:
        state["last_content_monotonic"] = time.monotonic() if now is None else now


def _message_from_state(state: dict[str, Any]) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": state.get("content") or None}
    calls = []
    for slot in state.get("tool_calls") or []:
        fn = slot.get("function") or {}
        if not slot.get("id") and not fn.get("name") and not fn.get("arguments"):
            continue
        calls.append(
            {
                "id": slot.get("id"),
                "type": slot.get("type") or "function",
                "function": {
                    "name": fn.get("name") or "",
                    "arguments": fn.get("arguments") or "",
                },
            }
        )
    if calls:
        msg["tool_calls"] = calls
    return msg


def _result_from_state(
    state: dict[str, Any],
    *,
    latency_s: float,
    ok: bool = True,
    infra_code: str | None = None,
    error: str | None = None,
    http_status: int | None = None,
    raw: dict[str, Any] | None = None,
) -> ChatResult:
    usage = state.get("usage") or {}
    return ChatResult(
        ok=ok,
        infra_code=infra_code,
        error=error,
        message=_message_from_state(state),
        finish_reason=state.get("finish_reason"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        latency_s=latency_s,
        raw=raw,
        http_status=http_status,
    )


def _result_from_completion(
    obj: dict[str, Any],
    *,
    latency_s: float,
    http_status: int | None,
) -> ChatResult:
    choice = (obj.get("choices") or [{}])[0] or {}
    usage = obj.get("usage") or {}
    return ChatResult(
        ok=True,
        message=choice.get("message") or {},
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        latency_s=latency_s,
        raw=obj,
        http_status=http_status,
    )


def _pop_sse_objects(buf: bytearray) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    while True:
        sep_lf = buf.find(b"\n\n")
        sep_cr = buf.find(b"\r\n\r\n")
        if sep_lf < 0 and sep_cr < 0:
            break
        if sep_cr < 0 or (sep_lf >= 0 and sep_lf <= sep_cr):
            block = bytes(buf[:sep_lf])
            del buf[: sep_lf + 2]
        else:
            block = bytes(buf[:sep_cr])
            del buf[: sep_cr + 4]
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


class BenchClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        connect_timeout_s: float,
        interrupt_event: threading.Event | None = None,
        on_live_armed: Callable[[], None] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.connect_timeout_s = float(connect_timeout_s)
        self.interrupt_event = interrupt_event
        self._on_live_armed = on_live_armed
        self._live_lock = threading.Lock()
        self._live_close: Callable[[], None] | None = None
        self._http: httpx.Client | None = None

    def _client(self) -> httpx.Client:
        if self._http is None:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._http = httpx.Client(
                timeout=httpx.Timeout(
                    connect=self.connect_timeout_s,
                    read=None,
                    write=None,
                    pool=self.connect_timeout_s,
                ),
                headers=headers,
            )
        return self._http

    def close_live(self) -> None:
        with self._live_lock:
            closer = self._live_close
            self._live_close = None
        if closer is None:
            return
        try:
            closer()
        except Exception:
            pass

    def raise_if_interrupted(self) -> None:
        if self.interrupt_event is not None and self.interrupt_event.is_set():
            self.close_live()
            raise KeyboardInterrupt

    def _arm_live(self, closer: Callable[[], None]) -> None:
        with self._live_lock:
            self._live_close = closer
        if self._on_live_armed is not None:
            self._on_live_armed()
        if self.interrupt_event is not None and self.interrupt_event.is_set():
            self.close_live()
            raise KeyboardInterrupt

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        parallel_tool_calls: bool | None = None,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repeat_penalty: float,
        seed: int | None,
        chat_template_kwargs: dict[str, Any] | None,
        max_tokens: int | None = None,
        stream: bool = False,
        request_timeout_s: float,
        on_wire: OnWire | None = None,
        on_progress: Callable[[ChatResult], None] | None = None,
        progress_every_bytes: int = 256,
    ) -> ChatResult:
        body = _build_request_body(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repeat_penalty,
            seed=seed,
            chat_template_kwargs=chat_template_kwargs,
            max_tokens=max_tokens,
            stream=stream,
        )
        raw_req = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if on_wire is not None:
            on_wire("request", raw_req)

        self.raise_if_interrupted()
        started = time.perf_counter()
        deadline = time.monotonic() + float(request_timeout_s)
        url = _chat_url(self.base_url)
        http = self._client()
        stream_cm = http.stream("POST", url, content=raw_req)
        resp: httpx.Response | None = None
        entered = False
        stop_watch = threading.Event()
        try:
            resp = stream_cm.__enter__()
            entered = True

            def _abort_live() -> None:
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
                if entered:
                    try:
                        stream_cm.__exit__(None, None, None)
                    except Exception:
                        pass

            self._arm_live(_abort_live)

            def _watch_deadline() -> None:
                while not stop_watch.is_set():
                    if time.monotonic() >= deadline:
                        self.close_live()
                        return
                    stop_watch.wait(0.2)

            threading.Thread(target=_watch_deadline, name="bench-http-deadline", daemon=True).start()
            self.raise_if_interrupted()
            if time.monotonic() >= deadline:
                return self._timeout(
                    started,
                    stream=stream,
                    state=_new_sse_state(),
                    http_status=resp.status_code,
                    request_timeout_s=request_timeout_s,
                )
            return self._read_response(
                resp,
                stream=stream,
                deadline=deadline,
                started=started,
                on_wire=on_wire,
                on_progress=on_progress,
                progress_every_bytes=progress_every_bytes,
                request_timeout_s=request_timeout_s,
            )
        except (httpx.ConnectTimeout, httpx.ConnectError) as exc:
            self.raise_if_interrupted()
            return ChatResult(
                ok=False,
                infra_code="INFRA_ERROR",
                error=f"connection: {exc}",
                latency_s=time.perf_counter() - started,
            )
        except KeyboardInterrupt:
            self.close_live()
            raise
        except SystemExit:
            self.close_live()
            raise
        except Exception as exc:
            self.raise_if_interrupted()
            if time.monotonic() >= deadline:
                return self._timeout(
                    started,
                    stream=stream,
                    state=_new_sse_state(),
                    http_status=None,
                    request_timeout_s=request_timeout_s,
                )
            return ChatResult(
                ok=False,
                infra_code="INFRA_ERROR",
                error=str(exc),
                latency_s=time.perf_counter() - started,
            )
        finally:
            stop_watch.set()
            self.close_live()

    def _timeout(
        self,
        started: float,
        *,
        stream: bool,
        state: dict[str, Any],
        http_status: int | None,
        request_timeout_s: float,
    ) -> ChatResult:
        self.close_live()
        code = deadline_abort_code(
            stream=stream,
            last_content_monotonic=state.get("last_content_monotonic"),
            now_monotonic=time.monotonic(),
            request_timeout_s=request_timeout_s,
        )
        if stream:
            return _result_from_state(
                state,
                latency_s=time.perf_counter() - started,
                ok=False,
                infra_code=code,
                error="request_timeout_s exceeded",
                http_status=http_status,
            )
        return ChatResult(
            ok=False,
            infra_code=code,
            error="request_timeout_s exceeded",
            latency_s=time.perf_counter() - started,
            http_status=http_status,
        )

    def _read_response(
        self,
        resp: httpx.Response,
        *,
        stream: bool,
        deadline: float,
        started: float,
        on_wire: OnWire | None,
        on_progress: Callable[[ChatResult], None] | None = None,
        progress_every_bytes: int = 256,
        request_timeout_s: float,
    ) -> ChatResult:
        state = _new_sse_state()
        buf = bytearray()
        since_progress = 0
        try:
            for chunk in resp.iter_bytes():
                self.raise_if_interrupted()
                if time.monotonic() >= deadline:
                    if chunk and on_wire is not None:
                        on_wire("response", chunk)
                    return self._timeout(
                        started,
                        stream=stream,
                        state=state,
                        http_status=resp.status_code,
                        request_timeout_s=request_timeout_s,
                    )
                if chunk:
                    if on_wire is not None:
                        on_wire("response", chunk)
                    buf.extend(chunk)
                    since_progress += len(chunk)
                    if stream:
                        for obj in _pop_sse_objects(buf):
                            _apply_chunk(state, obj)
                    if on_progress is not None and since_progress >= max(1, int(progress_every_bytes)):
                        since_progress = 0
                        on_progress(
                            _result_from_state(
                                state,
                                latency_s=time.perf_counter() - started,
                                http_status=resp.status_code,
                            )
                        )
        except KeyboardInterrupt:
            raise
        except SystemExit:
            raise
        except (httpx.ConnectTimeout, httpx.ConnectError) as exc:
            self.raise_if_interrupted()
            return ChatResult(
                ok=False,
                infra_code="INFRA_ERROR",
                error=f"connection: {exc}",
                latency_s=time.perf_counter() - started,
                http_status=resp.status_code,
            )
        except Exception as exc:
            self.raise_if_interrupted()
            if time.monotonic() >= deadline:
                return self._timeout(
                    started,
                    stream=stream,
                    state=state,
                    http_status=resp.status_code,
                    request_timeout_s=request_timeout_s,
                )
            return ChatResult(
                ok=False,
                infra_code="INFRA_ERROR",
                error=str(exc),
                latency_s=time.perf_counter() - started,
                http_status=resp.status_code,
            )

        latency = time.perf_counter() - started
        raw_text = bytes(buf).decode("utf-8", errors="replace")
        if resp.status_code >= 400:
            code = "CONTEXT_OVERFLOW" if _overflow_blob(resp.status_code, raw_text) else "INFRA_ERROR"
            return ChatResult(
                ok=False,
                infra_code=code,
                error=f"HTTP {resp.status_code}: {raw_text[:2000]}",
                http_status=resp.status_code,
                latency_s=latency,
            )
        if stream:
            for obj in _pop_sse_objects(buf):
                _apply_chunk(state, obj)
            return _result_from_state(state, latency_s=latency, http_status=resp.status_code)
        try:
            obj = json.loads(raw_text) if raw_text.strip() else {}
        except json.JSONDecodeError as exc:
            return ChatResult(
                ok=False,
                infra_code="INFRA_ERROR",
                error=f"invalid JSON: {exc}",
                http_status=resp.status_code,
                latency_s=latency,
            )
        if not isinstance(obj, dict):
            return ChatResult(
                ok=False,
                infra_code="INFRA_ERROR",
                error="completion is not an object",
                http_status=resp.status_code,
                latency_s=latency,
            )
        return _result_from_completion(obj, latency_s=latency, http_status=resp.status_code)
