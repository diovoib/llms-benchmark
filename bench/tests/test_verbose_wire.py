from __future__ import annotations

import json
from pathlib import Path

from bench.client import _apply_chunk, _message_from_state, _new_sse_state
from bench.results_io import RawWireLog, _stdout_text, format_verbose_request


def test_format_verbose_request_shows_roles_tools_and_omits_sampler() -> None:
    text = format_verbose_request(
        {
            "model": "demo",
            "temperature": 0.7,
            "top_p": 0.95,
            "seed": 42,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "description": "math",
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
                }
            ],
        }
    )
    assert text.startswith("model: demo\n")
    assert "[user]\nhello" in text
    assert "name: calculate" in text
    assert "type: integer" in text
    assert "required:" in text
    assert "temperature" not in text
    assert "top_p" not in text
    assert "seed" not in text


def test_apply_chunk_complete_message_two_tool_calls_without_index() -> None:
    state = _new_sse_state()
    _apply_chunk(
        state,
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_a",
                                "type": "function",
                                "function": {
                                    "name": "get_current_time",
                                    "arguments": "{}",
                                },
                            },
                            {
                                "id": "call_b",
                                "type": "function",
                                "function": {
                                    "name": "calculate",
                                    "arguments": '{"a": 7, "b": 6, "operation": "*"}',
                                },
                            },
                        ],
                    },
                }
            ]
        },
    )
    calls = _message_from_state(state)["tool_calls"]
    assert [c["id"] for c in calls] == ["call_a", "call_b"]
    assert [c["function"]["name"] for c in calls] == ["get_current_time", "calculate"]
    assert calls[0]["function"]["arguments"] == "{}"
    assert json.loads(calls[1]["function"]["arguments"]) == {
        "a": 7,
        "b": 6,
        "operation": "*",
    }


def test_apply_chunk_sse_deltas_concat_per_index() -> None:
    state = _new_sse_state()
    _apply_chunk(
        state,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {"name": "get_", "arguments": ""},
                            }
                        ]
                    }
                }
            ]
        },
    )
    _apply_chunk(
        state,
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "current_time", "arguments": "{}"},
                            },
                            {
                                "index": 1,
                                "id": "call_b",
                                "function": {"name": "calculate", "arguments": '{"a":1}'},
                            },
                        ]
                    }
                }
            ]
        },
    )
    calls = _message_from_state(state)["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["get_current_time", "calculate"]
    assert calls[0]["function"]["arguments"] == "{}"
    assert calls[1]["function"]["arguments"] == '{"a":1}'


def test_verbose_sse_content_chunks_appear_on_stdout_in_order(
    tmp_path: Path, capsys
) -> None:
    log = RawWireLog(tmp_path / "conversation.raw.txt", verbose=True, flush_bytes=1)
    try:
        log.begin_http_turn(0)
        log.on_wire("request", b'{"model":"m","messages":[{"role":"user","content":"hi"}]}')
        capsys.readouterr()
        log.on_wire(
            "response",
            b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
        )
        first = capsys.readouterr().out
        log.on_wire(
            "response",
            b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
        )
        second = capsys.readouterr().out
        log.finish_http_turn()
    finally:
        log.close()
    assert first == "Hel"
    assert second == "lo"


def test_verbose_nonstream_json_two_tools_stdout_names_and_yaml(
    tmp_path: Path, capsys
) -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "get_current_time", "arguments": "{}"},
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": '{"a": 2}',
                            },
                        },
                    ]
                }
            }
        ]
    }
    log = RawWireLog(tmp_path / "conversation.raw.txt", verbose=True, flush_bytes=1)
    try:
        log.begin_http_turn(0)
        log.on_wire("request", b'{"model":"m","messages":[]}')
        capsys.readouterr()
        log.on_wire("response", json.dumps(payload).encode("utf-8"))
        log.finish_http_turn()
        out = capsys.readouterr().out
    finally:
        log.close()
    live_a = out.find("[tool_call]\nget_current_time")
    live_b = out.find("[tool_call]\ncalculate")
    yaml_at = out.find("[tool_calls]\n")
    assert live_a != -1
    assert live_b != -1
    assert live_a < live_b < yaml_at
    yaml_block = out[yaml_at:]
    assert "name: get_current_time" in yaml_block
    assert "name: calculate" in yaml_block
    assert yaml_block.find("name: get_current_time") < yaml_block.find("name: calculate")


def test_stdout_text_replaces_characters_outside_stdout_encoding(monkeypatch) -> None:
    class AsciiStdout:
        encoding = "ascii"

        def __init__(self) -> None:
            self.chunks: list[str] = []

        def write(self, text: str) -> int:
            text.encode("ascii")
            self.chunks.append(text)
            return len(text)

        def flush(self) -> None:
            return None

    fake = AsciiStdout()
    monkeypatch.setattr("bench.results_io.sys.stdout", fake)
    _stdout_text("zażółć")
    written = "".join(fake.chunks)
    written.encode("ascii")
    assert "z" in written
    assert "ż" not in written
