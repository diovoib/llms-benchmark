from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from bench.paths import bench_root


def _fsync_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())


def write_json(path: Path, data: Any) -> None:
    _fsync_write_text(path, json.dumps(data, ensure_ascii=False, indent=2, default=str))


def write_text(path: Path, text: str) -> None:
    _fsync_write_text(path, text)


def write_yaml(path: Path, data: Any) -> None:
    _fsync_write_text(path, yaml.safe_dump(data, allow_unicode=True, sort_keys=False))


def trial_txt(trial: dict[str, Any]) -> str:
    lines = [
        f"trial_id: {trial.get('trial_id')}",
        f"case_id: {trial.get('case_id')}",
        f"prompt_variant: {trial.get('prompt_variant')}",
        f"repeat: {trial.get('repeat')}",
        f"seed: {trial.get('seed')}",
        f"in_progress: {trial.get('in_progress')}",
        f"hard_pass: {trial.get('score', {}).get('hard_pass') if trial.get('score') else None}",
        f"violations: {trial.get('score', {}).get('violations') if trial.get('score') else None}",
        f"excluded_from_rate: {trial.get('score', {}).get('excluded_from_rate') if trial.get('score') else None}",
        f"invalidated: {trial.get('score', {}).get('invalidated') if trial.get('score') else None}",
        f"latency_s: {trial.get('latency_s')}",
        f"prompt_tokens: {trial.get('prompt_tokens')}",
        f"completion_tokens: {trial.get('completion_tokens')}",
        f"first_tool_response_latency_s: {trial.get('first_tool_response_latency_s')}",
        f"finish_reason: {trial.get('finish_reason')}",
        "-" * 50,
        "TRANSCRIPT",
        trial.get("transcript_text") or "",
    ]
    return "\n".join(str(x) for x in lines) + "\n"


def write_case_md(path: Path, *, case_id: str, purpose: str, expected_result: str) -> None:
    write_text(
        path,
        (
            f"# {case_id}\n\n"
            f"## Purpose\n\n{purpose}\n\n"
            f"## Expected result\n\n{expected_result}\n"
        ),
    )


RESULTS_README = """# Bench results (self-contained)

This directory is enough to judge the run. Do not open the git repository.

## What is here

- `summary.json` / `summary.txt` — headline number at this folder. Nested folders have their own summaries.
- `MANIFEST.json` — models, profiles, prompt variants, whether the run was interrupted.
- `config.snapshot.yaml` — config used for the run (API key redacted).
- `judge/` — judge prompt, criteria, violation codes, JSON schema.
- `prompts/` — system-prompt snapshots (`SYSTEM_<variant>.txt`) and `POLICY.txt`.
- `<model>/<profile>/<prompt_variant>/SYSTEM.txt` — the system text actually prepended for that variant (policy already appended on A04/A07 trials).
- `<model>/<profile>/<prompt_variant>/cases/<id>/CASE.md` — purpose and expected result for that case.
- `trial_*.json` / `trial_*.txt` — assembled transcript used for mechanical scoring.
- `trial_*.raw.txt` — unsparsed HTTP: request body as sent (messages, tools, sampler, max_tokens) and response body as received (SSE or error JSON). No Authorization header.
- C01: `coding/trial_*/conversation.json`, `conversation.txt`, `conversation.raw.txt`. C01 has no CASE.md.

## Layers of summary

1. Case folder — no summary (read the trials).
2. Prompt variant — `build_summary` on that folder’s completed trials.
3. Profile (`greedy` / `real`) — unweighted mean of the variants’ `weighted_suite_hard_pass_rate`. C01 is a block on the profile summary, not in that mean.
4. Model — unweighted mean of greedy/real headlines that exist.
5. This root — unweighted mean of model headlines. One headline number.

`python run.py summarize <dir>` rebuilds these files. It skips unparseable JSON and `in_progress` trials. It does not rewrite trial content or `*.raw.txt`.

## How to judge

Start from this folder’s `README.md`. Then:

1. `judge/JUDGE_PROMPT.md` and `judge/CRITERIA.md`
2. `CASE.md` next to tools/agent trials (purpose / expected result). C01 has no CASE.md — read `conversation.txt`, `python_checks.json`, and `attempts/`. Do not run the extracted code.
3. `trial_*.txt` (assembled chat)
4. `*.raw.txt` when the `.txt` is short but wall time or `completion_tokens` is large — generation is on the wire even if the assembled transcript looks empty
5. `trial_*.json` and the **prompt-variant** `summary.json` (`<model>/<profile>/<variant>/`). Those are mechanical ground truth. Profile (`greedy`/`real`), model, and this root `summary.json` are unweighted means of the child headlines, not a second scoring pass.

Do not open the git repository, `llama.bat` / `llama.sh`, or `bench/src`.

You may still comment that a mechanically passing trial did not complete the user’s task, or that a failing trial was close. That is a judge score, not a change to `hard_pass`.

How to correct a judge row: copy `evidence.quote` verbatim from that trial’s transcript; change the judge 0/1 (or violation) only. Do not edit trial `hard_pass` / `violations`. If the harness looks wrong, write a note; do not “fix” it in the judge JSON.

`CASE_GENERATION_TIMEOUT` means that HTTP call exceeded the suite `request_timeout_s` while the stream still produced content within `min(request_timeout_s / 2, 5s)` of the deadline — the same class of fail as max_tokens. A stream silent for longer than that window, a non-stream deadline, or a failed TCP/connect within `connect_timeout_s` (2s) is `INFRA_ERROR`.

If a later HTTP status is 5xx, `*.raw.txt` for that trial should still contain the previous response and the request that received the error.

## Example: pack a set for a judge

From `bench/`:

```text
mkdir results\\zestaw-do-sedziego
xcopy /E /I results\\20260908T045257Z\\gemma-4-12b-it-Q4_K_M results\\zestaw-do-sedziego\\gemma-4-12b-it-Q4_K_M
xcopy /E /I results\\20260909T065153Z\\mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M results\\zestaw-do-sedziego\\mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M
python run.py summarize results\\zestaw-do-sedziego
```
"""


def write_results_readme(root: Path) -> None:
    write_text(root / "README.md", RESULTS_README)


def copy_judge_bundle(dest: Path) -> None:
    src = bench_root() / "judge"
    target = dest / "judge"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src, target)


def _yaml_text(data: Any) -> str:
    return yaml.safe_dump(
        data,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).rstrip()


def _tools_for_display(tools: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return rows
    for item in tools:
        if not isinstance(item, dict):
            rows.append({"raw": item})
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else {}
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        row: dict[str, Any] = {
            "name": fn.get("name") or item.get("name"),
            "description": fn.get("description"),
        }
        if params.get("properties") is not None:
            row["parameters"] = params.get("properties")
        if params.get("required"):
            row["required"] = params.get("required")
        rows.append(row)
    return rows


def _parse_tool_arguments(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def _tool_calls_for_display(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for call in calls:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        rows.append(
            {
                "id": call.get("id"),
                "name": fn.get("name"),
                "arguments": _parse_tool_arguments(fn.get("arguments")),
            }
        )
    return rows


def _format_verbose_messages(messages: Any) -> str:
    parts: list[str] = []
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if not isinstance(msg, dict):
            parts.append(str(msg))
            continue
        role = msg.get("role") or ""
        parts.append(f"[{role}]")
        if msg.get("tool_call_id"):
            parts.append(f"tool_call_id: {msg.get('tool_call_id')}")
        calls = msg.get("tool_calls")
        if calls:
            parts.append(_yaml_text(_tool_calls_for_display(list(calls))))
        content = msg.get("content")
        if content not in (None, ""):
            parts.append(str(content))
        parts.append("")
    return "\n".join(parts).rstrip()


def format_verbose_request(body: dict[str, Any]) -> str:
    chunks = [f"model: {body.get('model')}"]
    messages = _format_verbose_messages(body.get("messages"))
    if messages:
        chunks.append(messages)
    tools = _tools_for_display(body.get("tools"))
    if tools:
        chunks.append("tools:")
        chunks.append(_yaml_text(tools))
    return "\n".join(chunks) + "\n"


def _stdout_text(text: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    sys.stdout.write(safe)
    sys.stdout.flush()


class RawWireLog:
    def __init__(
        self,
        path: Path,
        *,
        flush_bytes: int = 256,
        verbose: bool = False,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        if not path.exists():
            path.write_bytes(b"")
        self._fh = path.open("ab")
        self._pending = bytearray()
        self.flush_bytes = int(flush_bytes)
        self.verbose = verbose
        self._in_response = False
        self._t0 = time.monotonic()
        self._sse_buf = bytearray()
        self._sse_state: dict[str, Any] | None = None
        self._printed_content_len = 0
        self._printed_tool_calls = False
        self._tool_header_printed: list[bool] = []
        self._tool_name_len: list[int] = []
        self._tool_arg_len: list[int] = []

    def close(self) -> None:
        self.flush(force=True)
        self._fh.close()

    def flush(self, *, force: bool = False) -> None:
        if not self._pending and not force:
            return
        if self._pending:
            self._fh.write(self._pending)
            self._pending.clear()
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def _write(self, data: bytes, *, force: bool = False) -> None:
        self._pending.extend(data)
        if force or len(self._pending) >= self.flush_bytes:
            self.flush(force=True)

    def begin_http_turn(self, step: int) -> None:
        from bench.client import _new_sse_state

        elapsed = time.monotonic() - self._t0
        header = f"\n===== step {step} t={elapsed:.3f}s =====\n----- request -----\n".encode("ascii")
        self._write(header, force=True)
        self._in_response = False
        self._sse_buf.clear()
        self._sse_state = _new_sse_state()
        self._printed_content_len = 0
        self._printed_tool_calls = False
        self._tool_header_printed: list[bool] = []
        self._tool_name_len: list[int] = []
        self._tool_arg_len: list[int] = []

    def _emit_new_content(self) -> None:
        if self._sse_state is None:
            return
        content = str(self._sse_state.get("content") or "")
        extra = content[self._printed_content_len :]
        if extra:
            _stdout_text(extra)
            self._printed_content_len = len(content)
        self._emit_new_tool_tokens()

    def _emit_new_tool_tokens(self) -> None:
        if self._sse_state is None:
            return
        slots = self._sse_state.get("tool_calls") or []
        if not isinstance(slots, list):
            return
        while len(self._tool_header_printed) < len(slots):
            self._tool_header_printed.append(False)
            self._tool_name_len.append(0)
            self._tool_arg_len.append(0)
        for i, slot in enumerate(slots):
            if not isinstance(slot, dict):
                continue
            fn = slot.get("function") if isinstance(slot.get("function"), dict) else {}
            name = str(fn.get("name") or "")
            args = str(fn.get("arguments") or "")
            if not self._tool_header_printed[i] and (name or args):
                _stdout_text("\n[tool_call]\n")
                self._tool_header_printed[i] = True
            if len(name) > self._tool_name_len[i]:
                _stdout_text(name[self._tool_name_len[i] :])
                self._tool_name_len[i] = len(name)
            if len(args) > self._tool_arg_len[i]:
                if self._tool_arg_len[i] == 0 and self._tool_name_len[i]:
                    _stdout_text("\n")
                _stdout_text(args[self._tool_arg_len[i] :])
                self._tool_arg_len[i] = len(args)

    def _emit_tool_calls_once(self) -> None:
        if self._printed_tool_calls or self._sse_state is None:
            return
        from bench.client import _message_from_state

        calls = _message_from_state(self._sse_state).get("tool_calls") or []
        if not calls:
            return
        self._printed_tool_calls = True
        _stdout_text("\n\n[tool_calls]\n")
        _stdout_text(_yaml_text(_tool_calls_for_display(calls)))
        _stdout_text("\n")

    def _consume_verbose_response(self) -> None:
        from bench.client import _apply_chunk, _pop_sse_objects

        if self._sse_state is None:
            return
        for obj in _pop_sse_objects(self._sse_buf):
            _apply_chunk(self._sse_state, obj)
            self._emit_new_content()

    def _consume_verbose_json_body(self) -> None:
        from bench.client import _apply_chunk

        if self._sse_state is None or not self._sse_buf:
            return
        try:
            obj = json.loads(bytes(self._sse_buf))
        except json.JSONDecodeError:
            return
        self._sse_buf.clear()
        if isinstance(obj, dict):
            _apply_chunk(self._sse_state, obj)
            self._emit_new_content()

    def on_wire(self, part: str, data: bytes) -> None:
        if part == "request":
            self._write(data)
            self._write(b"\n----- response -----\n", force=True)
            self._in_response = True
            if self.verbose:
                try:
                    body = json.loads(data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    body = None
                _stdout_text("\n\nRequest:\n")
                if isinstance(body, dict):
                    _stdout_text(format_verbose_request(body))
                else:
                    _stdout_text(data.decode("utf-8", errors="replace") + "\n")
                _stdout_text("\nResponse:\n")
            return
        if part == "response":
            self._write(data)
            if self.verbose:
                self._sse_buf.extend(data)
                self._consume_verbose_response()

    def finish_http_turn(self, *, force: bool = True) -> None:
        self.flush(force=force)
        if self.verbose:
            self._consume_verbose_json_body()
            self._consume_verbose_response()
            self._emit_tool_calls_once()
            _stdout_text("\n")
        self._in_response = False
        self._sse_buf.clear()


def snapshot_trial_files(json_path: Path, txt_path: Path, trial: dict[str, Any]) -> None:
    payload = {k: trial[k] for k in trial if k != "messages"}
    if "messages" in trial:
        payload["messages"] = trial.get("messages")
    write_json(json_path, payload)
    write_text(txt_path, trial_txt(trial))


def make_trial_snapshot(
    json_path: Path,
    txt_path: Path,
    trial: dict[str, Any],
) -> Callable[[], None]:
    def snap() -> None:
        snapshot_trial_files(json_path, txt_path, trial)

    return snap
