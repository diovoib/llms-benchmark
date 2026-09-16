from __future__ import annotations

import re
import tabnanny
import tokenize
import shutil
from pathlib import Path
from typing import Any

from bench.client import BenchClient, ChatResult, INFRA_CODES
from bench.paths import coding_assets
from bench.results_io import RawWireLog, write_json, write_text

REVIEW_OK = "FINAL_REVIEW: OK"
REVIEW_NOK = "FINAL_REVIEW: NOK"
FENCE_RE = re.compile(r"```(?:python|py)?[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_blocks(text: str) -> list[str]:
    return [b.strip() + "\n" for b in FENCE_RE.findall(text or "") if b.strip()]


def check_python_file(path: Path, *, relative: str) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return {"file": relative, "ok": False, "errors": [{"check": "utf-8", "message": str(exc)}]}
    if not source.strip():
        errors.append({"check": "empty", "message": "file is empty"})
    try:
        compile(source, relative, "exec")
    except SyntaxError as exc:
        errors.append(
            {
                "check": "syntax",
                "lineno": exc.lineno,
                "offset": exc.offset,
                "message": exc.msg,
                "text": (exc.text or "").rstrip("\n"),
            }
        )
    try:
        with tokenize.open(path) as fh:
            tabnanny.process_tokens(tokenize.generate_tokens(fh.readline))
    except tabnanny.NannyNag as exc:
        errors.append(
            {
                "check": "tabnanny",
                "lineno": exc.get_lineno(),
                "message": exc.get_msg(),
                "text": (exc.get_line() or "").rstrip("\n"),
            }
        )
    except tokenize.TokenError as exc:
        errors.append({"check": "tokenize", "message": str(exc)})
    except IndentationError as exc:
        errors.append({"check": "indent", "lineno": exc.lineno, "message": exc.msg})
    return {"file": relative, "ok": not errors, "errors": errors}


def check_agent_python_files(trial_dir: Path, relative_paths: list[str]) -> dict[str, Any]:
    files = [check_python_file(trial_dir / rel, relative=rel) for rel in relative_paths]
    return {"files": files, "all_ok": all(item["ok"] for item in files) if files else True}


def parse_final_review(text: str) -> str | None:
    verdict = None
    for line in (text or "").splitlines():
        s = line.strip()
        if s == REVIEW_OK:
            verdict = "OK"
        elif s == REVIEW_NOK:
            verdict = "NOK"
    return verdict


def coding_user_prompt(*, spec: str, contract: str, max_rounds: int) -> str:
    return (
        "Write me a tool_client.py that talks to an OpenAI-compatible Chat Completions server "
        "(endpoint chat/completions), following the specification and API contract below. "
        "Do not add JSON fields or headers that the API contract does not list.\n"
        "You have no tools. You do not run any code. All work is this single response, in the chat.\n"
        "In this response, repeat the cycle of two steps. Step (1) as a software engineer, implement or improve the full program "
        "in a fenced block (file tool_client.py). The opening line of the block is exactly three backtick characters "
        "followed immediately by python and nothing else on that line. After the file, the closing line is exactly "
        "three backtick characters and nothing else on that line. The block must be closed. "
        "Step (2) switch role to software engineer reviewer "
        "and review that program. The review is required; FINAL_REVIEW text is not a review. "
        "As reviewer, list every requirement in SPEC.md as ok, or not ok with a short reason, "
        "then whether the program matches API_CONTRACT.md as ok, or not ok with a short reason, "
        "then whether the Python is correct - ok, or not ok with a short reason. "
        "One cycle is one full program plus the review that follows it. "
        "If anything is not ok, next cycle begins. The cycle contains both steps: "
        "(1) switch back to software engineer, write a complete new version of the code, "
        "and step (2) switch to the reviewer role and review that version too. "
        "If the review marks everything ok, do not write another version; go to FINAL_REVIEW.\n"
        f"At most {max_rounds} such engineer/reviewer cycles. Then stop. "
        f"If anything is still not ok when you stop, the last line must be {REVIEW_NOK}.\n"
        "End exactly with one of these two lines and nothing after it:\n"
        f"{REVIEW_OK}\n"
        "or\n"
        f"{REVIEW_NOK}\n\n"
        "=== SPEC.md ===\n"
        f"{spec}\n\n"
        "=== API_CONTRACT.md ===\n"
        f"{contract}\n"
    )


def run_coding_trial(
    *,
    client: BenchClient,
    sampler: dict[str, Any],
    trial_dir: Path,
    max_rounds: int,
    max_tokens: int,
    request_timeout_s: float,
    stream: bool = True,
    log_flush_bytes: int = 256,
    verbose: bool = False,
    on_progress_bytes: int = 256,
) -> dict[str, Any]:
    trial_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir = trial_dir / "attempts"
    if attempts_dir.exists():
        shutil.rmtree(attempts_dir)
    attempts_dir.mkdir()

    spec = (coding_assets() / "SPEC.md").read_text(encoding="utf-8")
    contract = (coding_assets() / "API_CONTRACT.md").read_text(encoding="utf-8")
    user = coding_user_prompt(spec=spec, contract=contract, max_rounds=max_rounds)

    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
    conv_json = trial_dir / "conversation.json"
    conv_txt = trial_dir / "conversation.txt"
    raw_path = trial_dir / "conversation.raw.txt"
    write_json(
        conv_json,
        {"in_progress": True, "messages": messages, "infra": None},
    )
    write_text(conv_txt, f"in_progress: true\n[user]\n{user}\n")
    raw_path.write_bytes(b"")
    wire = RawWireLog(raw_path, flush_bytes=log_flush_bytes, verbose=verbose)
    wire.begin_http_turn(0)

    def on_progress(partial: ChatResult) -> None:
        live = [*messages, partial.to_message()]
        write_json(
            conv_json,
            {
                "in_progress": True,
                "messages": live,
                "finish_reason": partial.finish_reason,
                "infra": None,
                "partial": True,
            },
        )
        parts = [f"in_progress: true\n"]
        for msg in live:
            parts.append(f"[{msg.get('role')}]\n{msg.get('content') or ''}\n")
        write_text(conv_txt, "".join(parts))

    client.raise_if_interrupted()
    try:
        result = client.chat(
            messages,
            tools=None,
            tool_choice=None,
            max_tokens=max_tokens,
            stream=stream,
            request_timeout_s=request_timeout_s,
            on_wire=wire.on_wire,
            on_progress=on_progress,
            progress_every_bytes=on_progress_bytes,
            **sampler,
        )
    finally:
        wire.finish_http_turn(force=True)
        wire.close()
    assistant = result.to_message()
    if result.error and not assistant.get("content"):
        assistant["content"] = ""
    messages = [*messages, assistant]
    content = result.content or ""

    conv = {
        "in_progress": False,
        "messages": messages,
        "finish_reason": result.finish_reason,
        "infra": None if result.ok else result.infra_code,
        "error": result.error,
        "http_status": result.http_status,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "latency_s": result.latency_s,
    }
    write_json(trial_dir / "conversation.json", conv)
    conv_txt_parts = []
    for msg in messages:
        conv_txt_parts.append(f"[{msg.get('role')}]\n{msg.get('content') or ''}\n")
    conv_txt_parts.append("---")
    conv_txt_parts.append(f"finish_reason: {result.finish_reason}")
    conv_txt_parts.append(f"ok: {result.ok}")
    conv_txt_parts.append(f"infra: {None if result.ok else result.infra_code}")
    if result.error:
        conv_txt_parts.append(f"error: {result.error}")
    write_text(trial_dir / "conversation.txt", "\n".join(conv_txt_parts) + "\n")

    finish = result.finish_reason
    truncated = (
        (not result.ok and result.infra_code in {"CONTEXT_OVERFLOW", "CASE_GENERATION_TIMEOUT"})
        or (result.ok and str(finish or "").lower() in {"length", "max_tokens"})
    )
    verdict = parse_final_review(content)
    infra = None if result.ok else result.infra_code
    if infra not in INFRA_CODES:
        infra = None

    blocks = extract_python_blocks(content)
    attempt_files: list[str] = []
    for i, block in enumerate(blocks, start=1):
        name = f"{i:02d}.py"
        write_text(attempts_dir / name, block)
        attempt_files.append(f"attempts/{name}")

    last = None
    for block in reversed(blocks):
        if "run_tool_chat" in block:
            last = block
            break
    if last is None and blocks:
        last = blocks[-1]
    if last:
        write_text(trial_dir / "tool_client.py", last)

    to_check = list(attempt_files)
    if last:
        to_check.append("tool_client.py")
    python_checks = check_agent_python_files(trial_dir, to_check)
    write_json(trial_dir / "python_checks.json", python_checks)

    meta = {
        "suite": "coding",
        "case_id": "C01",
        "final_review": verdict,
        "truncated": truncated,
        "finish_reason": finish,
        "infra": infra,
        "attempt_files": attempt_files,
        "n_attempts": len(attempt_files),
        "last_program": "tool_client.py" if last else None,
        "conversation": "conversation.json",
        "conversation_text": "conversation.txt",
        "python_checks": "python_checks.json",
        "python_checks_ok": python_checks["all_ok"],
        "latency_s": result.latency_s,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "max_review_cycles_in_prompt": max_rounds,
    }
    write_json(trial_dir / "meta.json", meta)
    return meta


def check_reference() -> dict[str, Any]:
    rel = "reference/tool_client.py"
    path = coding_assets() / rel
    return check_python_file(path, relative=rel)
