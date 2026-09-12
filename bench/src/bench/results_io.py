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


def write_case_md(path: Path, *, case_id: str, purpose: str, expected_answer: str) -> None:
    write_text(
        path,
        (
            f"# {case_id}\n\n"
            f"## Purpose\n\n{purpose}\n\n"
            f"## Expected answer\n\n{expected_answer}\n"
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
- `<model>/<profile>/<prompt_variant>/cases/<id>/CASE.md` — purpose and expected answer for that case.
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
2. `CASE.md` next to tools/agent trials (purpose / expected answer). C01 has no CASE.md — read `conversation.txt`, `python_checks.json`, and `attempts/`. Do not run the extracted code.
3. `trial_*.txt` (assembled chat)
4. `*.raw.txt` when the `.txt` is short but wall time or `completion_tokens` is large — generation is on the wire even if the assembled transcript looks empty
5. `trial_*.json` and the **prompt-variant** `summary.json` (`<model>/<profile>/<variant>/`). Those are mechanical ground truth. Profile (`greedy`/`real`), model, and this root `summary.json` are unweighted means of the child headlines, not a second scoring pass.

Do not open the git repository, `llama.bat` / `llama.sh`, or `bench/src`.

You may still comment that a mechanically passing trial did not complete the user’s task, or that a failing trial was close. That is a judge score, not a change to `hard_pass`.

How to correct a judge row: copy `evidence.quote` verbatim from that trial’s transcript; change the judge 0/1 (or violation) only. Do not edit trial `hard_pass` / `violations`. If the harness looks wrong, write a note; do not “fix” it in the judge JSON.

`TIMEOUT` means that HTTP call exceeded the suite `request_timeout_s`. That can be a model that never finished generating, or a server that stopped responding. The bench does not distinguish those. A failed TCP/connect within `connect_timeout_s` (2s) is `INFRA_ERROR`, not `TIMEOUT`.

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
        self._resp_verbose = bytearray()
        self._in_response = False
        self._t0 = time.monotonic()

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
        elapsed = time.monotonic() - self._t0
        header = f"\n===== step {step} t={elapsed:.3f}s =====\n----- request -----\n".encode("ascii")
        self._write(header, force=True)
        self._in_response = False
        self._resp_verbose.clear()

    def on_wire(self, part: str, data: bytes) -> None:
        if part == "request":
            self._write(data)
            self._write(b"\n----- response -----\n", force=True)
            self._in_response = True
            self._resp_verbose.clear()
            if self.verbose:
                sys.stdout.buffer.write(b"\n\nRequest:\n")
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
            return
        if part == "response":
            self._write(data)
            if self._in_response:
                self._resp_verbose.extend(data)

    def finish_http_turn(self, *, force: bool = True) -> None:
        self.flush(force=force)
        if self.verbose:
            sys.stdout.buffer.write(b"\n\nResponse:\n")
            sys.stdout.buffer.write(bytes(self._resp_verbose))
            sys.stdout.buffer.flush()
        self._in_response = False


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
