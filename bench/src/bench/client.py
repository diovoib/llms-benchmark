from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


INFRA_CODES = frozenset({"INFRA_ERROR", "CONTEXT_OVERFLOW"})


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


class BenchClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("api_key is required")
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=None)

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
    ) -> ChatResult:
        extra_body: dict[str, Any] = {
            "top_k": top_k,
            "min_p": min_p,
            "repeat_penalty": repeat_penalty,
        }
        if chat_template_kwargs:
            extra_body["chat_template_kwargs"] = chat_template_kwargs
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "extra_body": extra_body,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = parallel_tool_calls
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        started = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except APITimeoutError as exc:
            return ChatResult(ok=False, infra_code="INFRA_ERROR", error=f"timeout: {exc}", latency_s=time.perf_counter() - started)
        except APIConnectionError as exc:
            return ChatResult(ok=False, infra_code="INFRA_ERROR", error=f"connection: {exc}", latency_s=time.perf_counter() - started)
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            body = ""
            try:
                body = exc.response.text[:2000] if exc.response is not None else str(exc)
            except Exception:
                body = str(exc)
            code = "INFRA_ERROR"
            blob = f"{status} {body}".lower()
            if any(s in blob for s in ("context size", "n_ctx", "context length", "too many tokens", "context overflow")):
                code = "CONTEXT_OVERFLOW"
            return ChatResult(
                ok=False,
                infra_code=code,
                error=f"HTTP {status}: {body}",
                http_status=status,
                latency_s=time.perf_counter() - started,
            )
        except Exception as exc:
            return ChatResult(ok=False, infra_code="INFRA_ERROR", error=str(exc), latency_s=time.perf_counter() - started)

        latency = time.perf_counter() - started
        raw = resp.model_dump()
        choice = (raw.get("choices") or [{}])[0]
        usage = raw.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        result = ChatResult(
            ok=True,
            message=choice.get("message") or {},
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=prompt_tokens,
            completion_tokens=usage.get("completion_tokens"),
            latency_s=latency,
            raw=raw,
        )
        return result
