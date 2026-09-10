from __future__ import annotations

import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

CALL_ID = "call_bench_1"
TOKEN_NO_TOOL = "TOKEN_NO_TOOL_PATH_OK"


class StubState:
    def __init__(self, final_token: str) -> None:
        self.lock = Lock()
        self.requests: list[dict[str, Any]] = []
        self.final_token = final_token

    def record(self, method: str, path: str, body: Any) -> None:
        with self.lock:
            self.requests.append({"method": method, "path": path, "body": body})

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.requests)


def make_handler(state: StubState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _read_json(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if not raw:
                return None
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                return {"_unparsed": raw.decode("utf-8", errors="replace")}

        def do_POST(self) -> None:
            body = self._read_json()
            state.record("POST", self.path, body)
            messages = []
            if isinstance(body, dict):
                messages = body.get("messages") or []
            has_tool_role = any(isinstance(m, dict) and m.get("role") == "tool" for m in messages)
            if has_tool_role:
                payload = {
                    "choices": [{
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": f"Tool said done {state.final_token}"},
                    }]
                }
            else:
                payload = {
                    "choices": [{
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": CALL_ID,
                                "type": "function",
                                "function": {"name": "echo_token", "arguments": json.dumps({"q": "ping"})},
                            }],
                        },
                    }]
                }
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

    return Handler


class StubServer:
    def __init__(self, final_token: str | None = None) -> None:
        token = final_token or ("TOK_" + secrets.token_hex(12))
        self.state = StubState(token)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.state))
        self.port = self.httpd.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def final_token(self) -> str:
        return self.state.final_token

    def serve_forever(self) -> None:
        self.httpd.serve_forever()

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
