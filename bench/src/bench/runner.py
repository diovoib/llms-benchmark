from __future__ import annotations

import os
import hashlib
import json
import signal
import sys
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from bench.client import BenchClient, ChatClient, ChatResult, chat_request_internal_error
from bench.coding_loop import run_coding_trial
from bench.config import model_dir_name, resolve_temperature, snapshot_config
from bench.hard_score import (
    mode_key,
    normalize_tool_calls,
    score_agent_trial,
    score_tools_turn,
)
from bench.mocks import execute_mock
from bench.preflight import run_preflight
from bench.results_io import (
    RawWireLog,
    copy_judge_bundle,
    snapshot_trial_files,
    write_case_md,
    write_json,
    write_results_readme,
    write_text,
    write_yaml,
)
from bench.progress import (
    line_fail,
    line_interrupted,
    line_ok,
    line_start,
    print_header,
    print_skipping,
    print_totals,
    score_reasons,
    trial_label,
)
from bench.prompts import apply_prompt_variant, selected_prompt_variants
from bench.spec import Case, PromptVariant
from bench.suites.cases import all_cases
from bench.summary import snapshot_prompts, write_summary_tree
from bench.template_dialect import (
    adapt_messages,
    coalesce_history,
    dialect_skip_reason,
)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def opaque_id(*parts: str) -> str:
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return "t_" + h[:16]


def sampler_kwargs(profile: dict[str, Any], model: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "temperature": resolve_temperature(profile, model),
        "top_p": float(profile["top_p"]),
        "top_k": int(profile["top_k"]),
        "min_p": float(profile["min_p"]),
        "repeat_penalty": float(profile["repeat_penalty"]),
        "seed": seed,
        "chat_template_kwargs": profile.get("chat_template_kwargs") or {},
    }


def choose_seed(profile: dict[str, Any], case_id: str, repeat: int, variant: str = "") -> int:
    if profile["seed_policy"] == "fixed":
        return int(profile["seed"])
    raw = f"{case_id}:{repeat}:{profile.get('name', '')}:{variant}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16) % 1_000_000_007


def selected_case_ids(cfg: dict[str, Any], suites: list[str]) -> list[str]:
    catalog = all_cases()
    ids: list[str] = []
    seen: set[str] = set()
    for name in suites:
        spec = (cfg.get("suites") or {}).get(name) or {}
        for cid in spec.get("cases") or []:
            if cid in seen:
                continue
            if cid == "C01":
                if name == "coding":
                    seen.add(cid)
                    ids.append(cid)
                continue
            case = catalog.get(cid)
            if not case or case.suite != name:
                continue
            seen.add(cid)
            ids.append(cid)
    return ids


def repeats_for(cfg: dict[str, Any], suite_name: str, profile_name: str) -> int:
    repeats = ((cfg.get("suites") or {}).get(suite_name) or {}).get("repeats") or {}
    return int(repeats.get(profile_name, 1))


def format_messages(messages: list[dict[str, Any]]) -> str:
    chunks = []
    for m in messages:
        role = m.get("role")
        chunks.append(f"[{role}]")
        if m.get("tool_calls"):
            chunks.append(json.dumps(m.get("tool_calls"), ensure_ascii=False, indent=2))
        if m.get("tool_call_id"):
            chunks.append(f"tool_call_id={m.get('tool_call_id')}")
        content = m.get("content")
        if content:
            chunks.append(str(content))
        chunks.append("")
    return "\n".join(chunks)


def bind_weather_expect(case: Case, repeat: int = 0) -> Case:
    """Copy the case. Observation tokens are set on the case Expect, not bound here."""
    return deepcopy(case)


def mark_interrupted_if_present(paths: list[Path]) -> None:
    for path in paths:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        data["interrupted"] = True
        data["in_progress"] = False
        write_json(path, data)


def run_tool_or_agent_case(
    *,
    client: ChatClient,
    case: Case,
    sampler: dict[str, Any],
    preflight: dict[str, Any],
    prompt_variant: PromptVariant,
    repeat: int = 0,
    max_tokens: int | None = None,
    request_timeout_s: float | None = None,
    stream: bool = True,
    wire: RawWireLog | None = None,
    on_snapshot: Callable[[dict[str, Any]], None] | None = None,
    progress_every_bytes: int = 256,
) -> dict[str, Any]:
    case = bind_weather_expect(case, repeat)
    dialect = (preflight or {}).get("template_dialect")
    id_map: dict[str, str] = {}
    messages = coalesce_history(
        adapt_messages(json.loads(json.dumps(case.messages)), dialect, id_map),
        dialect,
    )
    extra_mock = {"error_cities": list(case.error_cities), "repeat": repeat}
    steps = []
    total_latency = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    ttf = None
    max_steps = int(case.max_steps or 1)
    followup = case.followup_user
    followup_sent = False

    def _chat(step_i: int):
        kwargs: dict[str, Any] = dict(sampler)
        kwargs["stream"] = stream
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if request_timeout_s is not None:
            kwargs["request_timeout_s"] = request_timeout_s
        if wire is not None:
            wire.begin_http_turn(step_i)
            kwargs["on_wire"] = wire.on_wire
        if on_snapshot is not None:
            kwargs["progress_every_bytes"] = progress_every_bytes

            def on_progress(partial: ChatResult) -> None:
                live_messages = list(messages) + [partial.to_message()]
                on_snapshot(
                    {
                        "in_progress": True,
                        "messages": live_messages,
                        "steps": steps,
                        "partial_assistant": partial.to_message(),
                        "latency_s": total_latency + partial.latency_s,
                        "transcript_text": format_messages(live_messages),
                    }
                )

            kwargs["on_progress"] = on_progress
        try:
            blocked = chat_request_internal_error(messages)
            if blocked is not None:
                return blocked
            return client.chat(
                messages,
                tools=case.tools,
                tool_choice=case.tool_choice,
                parallel_tool_calls=case.parallel_tool_calls,
                **kwargs,
            )
        finally:
            if wire is not None:
                wire.finish_http_turn(force=True)

    hit_max = False
    for step_i in range(max_steps):
        client.raise_if_interrupted()
        result = _chat(step_i)
        total_latency += result.latency_s
        prompt_tokens += result.prompt_tokens or 0
        completion_tokens += result.completion_tokens or 0
        normalized = normalize_tool_calls(result.tool_calls) if result.ok else []
        if normalized and ttf is None:
            ttf = result.latency_s
        turn_score = {}
        if case.suite == "tools" and max_steps <= 1:
            turn_score = score_tools_turn(
                case=case,
                result=result,
                normalized=normalized,
                prompt_variant=prompt_variant,
            )
        step_rec = {
            "result": {
                "ok": result.ok,
                "infra_code": result.infra_code,
                "error": result.error,
                "finish_reason": result.finish_reason,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
            "content": result.content,
            "normalized": normalized,
            "score": turn_score,
            "tool_payloads": [],
        }
        steps.append(step_rec)
        if on_snapshot is not None:
            on_snapshot(
                {
                    "in_progress": True,
                    "messages": messages,
                    "steps": steps,
                    "latency_s": total_latency,
                    "transcript_text": format_messages(messages),
                }
            )
        if not result.ok:
            if wire is not None:
                wire.flush(force=True)
            break
        if not normalized:
            messages.extend(adapt_messages([result.to_message()], dialect, id_map))
            messages = coalesce_history(messages, dialect)
            if followup and not followup_sent:
                messages.append({"role": "user", "content": followup})
                messages = coalesce_history(messages, dialect)
                followup_sent = True
                if on_snapshot is not None:
                    on_snapshot(
                        {
                            "in_progress": True,
                            "messages": messages,
                            "steps": steps,
                            "latency_s": total_latency,
                            "transcript_text": format_messages(messages),
                        }
                    )
                continue
            break
        turn_has_unparsed_arguments = any(
            not call.get("arguments_parsed") for call in normalized
        )
        if turn_has_unparsed_arguments:
            break
        chunk: list[dict[str, Any]] = [result.to_message()]
        for call in normalized:
            payload = execute_mock(call["name"] or "", call["arguments"] or {}, extra_mock)
            step_rec["tool_payloads"].append(payload)
            chunk.append({
                "role": "tool",
                "tool_call_id": call.get("id") or "",
                "content": payload,
            })
        messages.extend(adapt_messages(chunk, dialect, id_map))
        messages = coalesce_history(messages, dialect)
        if on_snapshot is not None:
            on_snapshot(
                {
                    "in_progress": True,
                    "messages": messages,
                    "steps": steps,
                    "latency_s": total_latency,
                    "transcript_text": format_messages(messages),
                }
            )

    hit_max = bool(steps and steps[-1].get("normalized")) and len(steps) >= max_steps

    final_content = ""
    for st in reversed(steps):
        if not st.get("normalized"):
            final_content = st.get("content") or ""
            break
    if case.suite == "agent" or max_steps > 1:
        score = score_agent_trial(
            case, steps, final_content, hit_max, prompt_variant=prompt_variant
        )
    else:
        score = steps[0]["score"] if steps else {
            "hard_pass": False,
            "violations": ["INFRA_ERROR"],
            "excluded_from_rate": False,
            "invalidated": False,
        }

    last = steps[-1] if steps else {}
    ncalls = last.get("normalized") or []
    mk = mode_key(ncalls if case.suite == "tools" else [
        c for st in steps for c in (st.get("normalized") or [])
    ], final_content if case.suite == "agent" else (steps[0].get("content") if steps else ""), last.get("result", {}).get("finish_reason"))

    return {
        "ok_run": True,
        "in_progress": False,
        "score": score,
        "messages": messages,
        "steps": steps,
        "latency_s": total_latency,
        "transcript_text": format_messages(messages),
        "normalized": ncalls,
        "mode_key": mk,
        "finish_reason": last.get("result", {}).get("finish_reason"),
        "prompt_tokens": prompt_tokens or None,
        "completion_tokens": completion_tokens or None,
        "first_tool_response_latency_s": ttf,
        "final_content": final_content,
    }


def _suite_timeout(cfg: dict[str, Any], suite: str) -> float:
    return float(((cfg.get("suites") or {}).get(suite) or {})["request_timeout_s"])


def _suite_max_tokens(cfg: dict[str, Any], suite: str) -> int:
    return int(((cfg.get("suites") or {}).get(suite) or {})["max_tokens"])


def _make_client(cfg: dict[str, Any], model: dict[str, Any], interrupt_event: threading.Event, on_live_armed: Callable[[], None]) -> BenchClient:
    return BenchClient(
        base_url=cfg["base_url"],
        api_key=str(cfg.get("api_key") or ""),
        model=str(model["name"]),
        connect_timeout_s=float(cfg["connect_timeout_s"]),
        interrupt_event=interrupt_event,
        on_live_armed=on_live_armed,
    )


def run_benchmark(
    cfg: dict[str, Any],
    *,
    profiles: list[str],
    suites: list[str],
    out_root: Path,
    prompt_variants: list[str] | None = None,
    verbose: bool = False,
) -> Path:
    run_dir = out_root / _now_stamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "config.snapshot.yaml", snapshot_config(cfg))
    copy_judge_bundle(run_dir)
    snapshot_prompts(run_dir)
    write_results_readme(run_dir)
    catalog = all_cases()
    all_trials: list[dict[str, Any]] = []
    coding_rows: list[dict[str, Any]] = []
    manifest_models = []
    case_ids = selected_case_ids(cfg, suites)
    variants = selected_prompt_variants(cfg, prompt_variants)
    interrupted = False
    interrupt_at = ""
    pending_line = False
    n_ok = 0
    n_fail = 0
    interrupt_event = threading.Event()
    live_clients: list[BenchClient] = []
    live_lock = threading.Lock()
    live_changed = threading.Event()
    pending_trial_jsons: list[Path] = []
    previous_handlers: dict[int, Any] = {}
    signals_installed = False
    interrupt_lock = threading.Lock()
    win_ctrl_unhook: Any = None

    def finish_trial_console(started: float, hard_pass: bool, reasons: list[str] | None = None) -> None:
        nonlocal pending_line, n_ok, n_fail
        pending_line = False
        if hard_pass:
            n_ok += 1
            line_ok(started)
        else:
            n_fail += 1
            line_fail(started, reasons)

    def close_all_live() -> None:
        with live_lock:
            clients = list(live_clients)
        for item in clients:
            item.close_live()

    def on_live_armed() -> None:
        live_changed.set()
        if interrupt_event.is_set():
            close_all_live()

    def register_client(client: BenchClient) -> None:
        with live_lock:
            live_clients.append(client)

    def note_interrupt() -> None:
        with interrupt_lock:
            if interrupt_event.is_set():
                return
            sys.stderr.write("received interrupt, doing graceful shutdown\n")
            sys.stderr.flush()
            interrupt_event.set()
        live_changed.set()

    def disarm_second_ctrl_c() -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal.SIG_DFL)

    def on_interrupt(signum: int, frame: Any) -> None:
        note_interrupt()
        disarm_second_ctrl_c()
        raise KeyboardInterrupt

    def abort_http_when_interrupted() -> None:
        interrupt_event.wait()
        while True:
            close_all_live()
            live_changed.wait(timeout=0.2)
            live_changed.clear()

    def hook_win_ctrl() -> Any:
        if sys.platform != "win32":
            return None
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handler_routine = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)
        ctrl_c, ctrl_break = 0, 1

        def win_handler(ctrl_type: int) -> bool:
            if ctrl_type not in (ctrl_c, ctrl_break):
                return False
            if interrupt_event.is_set():
                os._exit(130)
            note_interrupt()
            return True

        callback = handler_routine(win_handler)
        if not kernel32.SetConsoleCtrlHandler(callback, True):
            return None

        unhooked = False

        def unhook() -> None:
            nonlocal unhooked
            if unhooked:
                return
            unhooked = True
            kernel32.SetConsoleCtrlHandler(callback, False)

        return unhook

    threading.Thread(
        target=abort_http_when_interrupted,
        name="bench-abort-http",
        daemon=True,
    ).start()

    if threading.current_thread() is threading.main_thread():
        previous_handlers[signal.SIGINT] = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, on_interrupt)
        if hasattr(signal, "siginterrupt"):
            signal.siginterrupt(signal.SIGINT, True)
        if hasattr(signal, "SIGBREAK"):
            previous_handlers[signal.SIGBREAK] = signal.getsignal(signal.SIGBREAK)
            signal.signal(signal.SIGBREAK, on_interrupt)
        signals_installed = True
        win_ctrl_unhook = hook_win_ctrl()

    try:
        for model in cfg["models"]:
            print_header(str(run_dir), str(model["name"]))
            mdir_name = model_dir_name(model)
            for profile_name in profiles:
                profile = dict(cfg["profiles"][profile_name])
                profile["name"] = profile_name
                dest_profile = run_dir / mdir_name / profile_name
                dest_profile.mkdir(parents=True, exist_ok=True)
                client = _make_client(cfg, model, interrupt_event, on_live_armed)
                register_client(client)
                tool_case_ids = [c for c in case_ids if c != "C01"]
                tools_timeout = _suite_timeout(cfg, "tools")
                tools_max_tokens = _suite_max_tokens(cfg, "tools")
                log_flush_bytes = int(cfg.get("log_flush_bytes") or 256)
                if tool_case_ids:
                    pending_line = True
                    t0 = line_start(f"preflight  {profile_name}")
                    client.raise_if_interrupted()
                    pre = run_preflight(
                        client,
                        profile,
                        request_timeout_s=tools_timeout,
                        max_tokens=tools_max_tokens,
                    )
                    pending_line = False
                    if pre.get("ok"):
                        line_ok(t0)
                    else:
                        line_fail(t0, list(pre.get("reasons") or ["preflight failed"]))
                else:
                    pre = {"ok": True, "skipped": True, "reasons": ["no tools/agent cases selected"]}
                write_json(dest_profile / "preflight.json", pre)
                entry = {
                    "model": model["name"],
                    "profile": profile_name,
                    "sampler": {**profile, "resolved_temperature": resolve_temperature(profile, model)},
                    "preflight_ok": pre.get("ok"),
                    "chat_format": (pre.get("server") or {}).get("chat_format"),
                    "build": (pre.get("server") or {}).get("build"),
                    "thinking": profile.get("thinking"),
                    "prompt_variants": [name for name, _ in variants] if tool_case_ids else [],
                }
                manifest_models.append(entry)
                if tool_case_ids and not pre.get("ok"):
                    write_text(dest_profile / "ABORTED.txt", "preflight failed\n" + "\n".join(pre.get("reasons") or []))
                    tool_case_ids = []
                else:
                    skip_set: set[str] = set()
                    if tool_case_ids and not pre.get("parallel_tool_calls_confirmed"):
                        skip_set.update(
                            cid
                            for cid in tool_case_ids
                            if (c := catalog.get(cid)) is not None and c.invalidate_if_no_parallel
                        )
                    if tool_case_ids:
                        dialect = pre.get("template_dialect") or {}
                        skip_set.update(
                            cid
                            for cid in tool_case_ids
                            if (c := catalog.get(cid)) is not None and dialect_skip_reason(c, dialect)
                        )
                    if skip_set:
                        print_skipping(sorted(skip_set))
                        tool_case_ids = [cid for cid in tool_case_ids if cid not in skip_set]

                for variant_name, system_text in variants:
                    if not tool_case_ids:
                        break
                    dest = dest_profile / variant_name
                    dest.mkdir(parents=True, exist_ok=True)
                    write_text(dest / "SYSTEM.txt", system_text + "\n")

                    sweep_cfg = cfg.get("optional_temp_sweep") or {}
                    if sweep_cfg.get("enabled"):
                        sweep_rows = []
                        sweep_suite = str(sweep_cfg.get("suite") or "tools")
                        sweep_ids = ((cfg.get("suites") or {}).get(sweep_suite) or {}).get("cases") or []
                        sweep_timeout = _suite_timeout(cfg, sweep_suite)
                        sweep_max_tokens = _suite_max_tokens(cfg, sweep_suite)
                        for temp in sweep_cfg.get("temperatures") or []:
                            for cid in sweep_ids:
                                if cid not in catalog or cid == "C01" or cid not in tool_case_ids:
                                    continue
                                seed = choose_seed(profile, f"SWEEP-{cid}", 0, variant_name)
                                sampler = sampler_kwargs(profile, model, seed)
                                sampler["temperature"] = float(temp)
                                bound = apply_prompt_variant(catalog[cid], system_text)
                                pending_line = True
                                t0 = line_start(
                                    trial_label(
                                        case_id=cid,
                                        suite=catalog[cid].suite,
                                        profile=profile_name,
                                        variant=f"{variant_name}@t{temp}",
                                        repeat=0,
                                        nrep=1,
                                        seed=seed,
                                    )
                                )
                                sweep_raw = dest / f"sweep_{cid}_t{temp}.raw.txt"
                                sweep_raw.write_bytes(b"")
                                sweep_wire = RawWireLog(sweep_raw, flush_bytes=log_flush_bytes, verbose=verbose)
                                try:
                                    res = run_tool_or_agent_case(
                                        client=client,
                                        case=bound,
                                        sampler=sampler,
                                        preflight=pre,
                                        prompt_variant=variant_name,
                                        repeat=0,
                                        max_tokens=sweep_max_tokens,
                                        request_timeout_s=sweep_timeout,
                                        stream=True,
                                        wire=sweep_wire,
                                        progress_every_bytes=log_flush_bytes,
                                    )
                                finally:
                                    sweep_wire.close()
                                client.raise_if_interrupted()
                                hp = bool((res.get("score") or {}).get("hard_pass"))
                                finish_trial_console(t0, hp, score_reasons(res.get("score")))
                                sweep_rows.append({
                                    "temperature": float(temp),
                                    "case_id": cid,
                                    "prompt_variant": variant_name,
                                    "hard_pass": hp,
                                    "violations": (res.get("score") or {}).get("violations"),
                                })
                        write_json(dest / "temp_sweep.json", sweep_rows)

                    for cid in tool_case_ids:
                        case = catalog[cid]
                        bound = apply_prompt_variant(case, system_text)
                        nrep = repeats_for(cfg, case.suite, profile_name)
                        suite_name = case.suite
                        case_timeout = _suite_timeout(cfg, suite_name)
                        case_max_tokens = _suite_max_tokens(cfg, suite_name)
                        tdir = dest / "cases" / cid
                        tdir.mkdir(parents=True, exist_ok=True)
                        if case.purpose and case.expected_result:
                            write_case_md(
                                tdir / "CASE.md",
                                case_id=cid,
                                purpose=str(case.purpose),
                                expected_result=str(case.expected_result),
                            )
                        for rep in range(nrep):
                            seed = choose_seed(profile, cid, rep, variant_name)
                            sampler = sampler_kwargs(profile, model, seed)
                            pending_line = True
                            json_path = tdir / f"trial_{rep + 1:03d}.json"
                            txt_path = tdir / f"trial_{rep + 1:03d}.txt"
                            raw_path = tdir / f"trial_{rep + 1:03d}.raw.txt"
                            pending_trial_jsons = [json_path]
                            trial_id = opaque_id(str(model["name"]), profile_name, variant_name, cid, str(rep))
                            trial: dict[str, Any] = {
                                "trial_id": trial_id,
                                "case_id": cid,
                                "suite": suite_name,
                                "repeat": rep,
                                "seed": seed,
                                "prompt_variant": variant_name,
                                "in_progress": True,
                            }
                            snapshot_trial_files(json_path, txt_path, trial)
                            raw_path.write_bytes(b"")
                            wire = RawWireLog(raw_path, flush_bytes=log_flush_bytes, verbose=verbose)

                            def on_snapshot(
                                partial: dict[str, Any],
                                trial=trial,
                                json_path=json_path,
                                txt_path=txt_path,
                            ) -> None:
                                trial.update(partial)
                                snapshot_trial_files(json_path, txt_path, trial)

                            t0 = line_start(
                                trial_label(
                                    case_id=cid,
                                    suite=suite_name,
                                    profile=profile_name,
                                    variant=variant_name,
                                    repeat=rep,
                                    nrep=nrep,
                                    seed=seed,
                                )
                            )
                            try:
                                res = run_tool_or_agent_case(
                                    client=client,
                                    case=bound,
                                    sampler=sampler,
                                    preflight=pre,
                                    prompt_variant=variant_name,
                                    repeat=rep,
                                    max_tokens=case_max_tokens,
                                    request_timeout_s=case_timeout,
                                    stream=True,
                                    wire=wire,
                                    on_snapshot=on_snapshot,
                                    progress_every_bytes=log_flush_bytes,
                                )
                            finally:
                                wire.close()
                            trial.update(res)
                            trial["in_progress"] = False
                            snapshot_trial_files(json_path, txt_path, trial)
                            pending_trial_jsons = []
                            all_trials.append(trial)
                            hp = bool((trial.get("score") or {}).get("hard_pass"))
                            finish_trial_console(t0, hp, score_reasons(trial.get("score")))
                    write_summary_tree(run_dir)

                if "C01" in case_ids:
                    nrep = repeats_for(cfg, "coding", profile_name)
                    coding_client = _make_client(cfg, model, interrupt_event, on_live_armed)
                    register_client(coding_client)
                    coding_timeout = _suite_timeout(cfg, "coding")
                    coding_max_tokens = int(cfg["ctx_size"])
                    for rep in range(nrep):
                        seed = choose_seed(profile, "C01", rep)
                        sampler = sampler_kwargs(profile, model, seed)
                        trial_id = opaque_id(str(model["name"]), profile_name, "C01", str(rep))
                        cdir = dest_profile / "coding" / f"trial_{rep + 1:03d}"
                        pending_line = True
                        pending_trial_jsons = [
                            cdir / "trial.json",
                            cdir / "conversation.json",
                            cdir / "meta.json",
                        ]
                        t0 = line_start(
                            trial_label(
                                case_id="C01",
                                suite="coding",
                                profile=profile_name,
                                variant="—",
                                repeat=rep,
                                nrep=nrep,
                                seed=seed,
                            )
                        )
                        row = run_coding_trial(
                            client=coding_client,
                            sampler=sampler,
                            trial_dir=cdir,
                            max_rounds=int(cfg.get("max_coding_rounds") or 7),
                            max_tokens=coding_max_tokens,
                            request_timeout_s=coding_timeout,
                            stream=True,
                            log_flush_bytes=log_flush_bytes,
                            verbose=verbose,
                            on_progress_bytes=log_flush_bytes,
                        )
                        row.update({
                            "trial_id": trial_id,
                            "case_id": "C01",
                            "suite": "coding",
                            "repeat": rep,
                            "seed": seed,
                            "prompt_variant": None,
                            "in_progress": False,
                            "transcript_text": (cdir / "conversation.txt").read_text(encoding="utf-8"),
                        })
                        write_json(
                            cdir / "trial.json",
                            {k: v for k, v in row.items() if k != "transcript_text"},
                        )
                        pending_trial_jsons = []
                        coding_rows.append(row)
                        all_trials.append(row)
                        coding_ok = bool(row.get("python_checks_ok")) and not row.get("infra")
                        reasons = []
                        if row.get("infra"):
                            reasons.append(str(row.get("infra")))
                        if not row.get("python_checks_ok"):
                            reasons.append("python_checks")
                        finish_trial_console(t0, coding_ok, reasons)
                    write_summary_tree(run_dir)
    except KeyboardInterrupt:
        interrupted = True
        disarm_second_ctrl_c()
        if pending_line:
            line_interrupted()
        interrupt_at = datetime.now(timezone.utc).isoformat()
        mark_interrupted_if_present(pending_trial_jsons)
        pending_trial_jsons = []
    finally:
        if interrupt_event.is_set():
            disarm_second_ctrl_c()
        elif signals_installed:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            if win_ctrl_unhook is not None:
                win_ctrl_unhook()

    if interrupted:
        write_text(
            run_dir / "INTERRUPTED.txt",
            f"interrupted by user\n{interrupt_at}\n",
        )
    write_summary_tree(run_dir)
    write_json(
        run_dir / "MANIFEST.json",
        {
            "created_at": _now_stamp(),
            "base_url": cfg.get("base_url"),
            "models": manifest_models,
            "profiles": list(profiles),
            "prompt_variants": [
                {"name": name, "system": text}
                for name, text in variants
            ],
            "suites": list(suites),
            "interrupted": interrupted,
            "launcher": {
                "path": cfg.get("_launcher"),
                "kind": cfg.get("_launcher_kind"),
                "ctx_size": cfg.get("ctx_size"),
                "api_key": "***" if cfg.get("api_key") else None,
            },
        },
    )
    write_json(run_dir / "trial_id_map.json", {
        t["trial_id"]: {
            "case_id": t["case_id"],
            "repeat": t.get("repeat"),
            "prompt_variant": t.get("prompt_variant"),
        }
        for t in all_trials
    })
    print_totals(n_ok=n_ok, n_fail=n_fail, interrupted=interrupted)
    return run_dir
