from __future__ import annotations

import os
import hashlib
import json
import signal
import sys
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench.client import BenchClient
from bench.coding_loop import run_coding_trial
from bench.config import model_dir_name, resolve_temperature, snapshot_config
from bench.hard_score import (
    DIMENSIONS,
    SUITE_WEIGHTS,
    aggregate_trials,
    mode_key,
    normalize_tool_calls,
    score_agent_trial,
    score_tools_turn,
)
from bench.mocks import execute_mock, weather_claim_tokens, weather_required_substrings
from bench.preflight import run_preflight
from bench.results_io import trial_txt, write_json, write_text, write_yaml
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
from bench.suites.cases import all_cases, case_stem
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
            if not case or case["suite"] != name:
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


def bind_weather_expect(case: dict[str, Any], repeat: int) -> dict[str, Any]:
    expect = dict(case.get("expect") or {})
    if expect.pop("final_must_contain_weather", False):
        expect["final_must_contain"] = weather_required_substrings(repeat)
    if expect.pop("must_not_claim_weather", False):
        expect["must_not_claim_success_tokens"] = weather_claim_tokens()
    out = dict(case)
    out["expect"] = expect
    return out


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
    client: BenchClient,
    case: dict[str, Any],
    sampler: dict[str, Any],
    preflight: dict[str, Any],
    repeat: int = 0,
) -> dict[str, Any]:
    case = bind_weather_expect(case, repeat)
    dialect = (preflight or {}).get("template_dialect")
    id_map: dict[str, str] = {}
    messages = coalesce_history(
        adapt_messages(json.loads(json.dumps(case["messages"])), dialect, id_map),
        dialect,
    )
    extra_mock = {"error_cities": case.get("error_cities") or [], "repeat": repeat}
    steps = []
    total_latency = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    ttf = None
    max_steps = int(case.get("max_steps") or 1)
    followup = case.get("followup_user")
    followup_sent = False

    hit_max = False
    for step_i in range(max_steps):
        client.raise_if_interrupted()
        result = client.chat(
            messages,
            tools=case.get("tools"),
            tool_choice=case.get("tool_choice", "auto"),
            parallel_tool_calls=case.get("parallel_tool_calls"),
            **sampler,
        )
        total_latency += result.latency_s
        prompt_tokens += result.prompt_tokens or 0
        completion_tokens += result.completion_tokens or 0
        normalized = normalize_tool_calls(result.tool_calls) if result.ok else []
        if normalized and ttf is None:
            ttf = result.latency_s
        turn_score = {}
        if case["suite"] == "tools" and max_steps <= 1:
            turn_score = score_tools_turn(case=case, result=result, normalized=normalized)
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
        if not result.ok:
            break
        if not normalized:
            messages.extend(adapt_messages([result.to_message()], dialect, id_map))
            messages = coalesce_history(messages, dialect)
            if followup and not followup_sent:
                messages.append({"role": "user", "content": followup})
                messages = coalesce_history(messages, dialect)
                followup_sent = True
                continue
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

    hit_max = bool(steps and steps[-1].get("normalized")) and len(steps) >= max_steps

    final_content = ""
    for st in reversed(steps):
        if not st.get("normalized"):
            final_content = st.get("content") or ""
            break
    if case["suite"] == "agent" or max_steps > 1:
        score = score_agent_trial(case, steps, final_content, hit_max)
    else:
        score = steps[0]["score"] if steps else {
            "hard_pass": False,
            "violations": ["INFRA_ERROR"],
            "excluded_from_rate": False,
            "invalidated": False,
        }

    last = steps[-1] if steps else {}
    ncalls = last.get("normalized") or []
    mk = mode_key(ncalls if case["suite"] == "tools" else [
        c for st in steps for c in (st.get("normalized") or [])
    ], final_content if case["suite"] == "agent" else (steps[0].get("content") if steps else ""), last.get("result", {}).get("finish_reason"))

    return {
        "ok_run": True,
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


def _rates_block(trials: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list] = defaultdict(list)
    by_suite: dict[str, list] = defaultdict(list)
    by_dim: dict[str, list] = defaultdict(list)
    for t in trials:
        by_case[t["case_id"]].append(t)
        by_suite[t["suite"]].append(t)
        for dim, ids in DIMENSIONS.items():
            if case_stem(t["case_id"]) in ids:
                by_dim[dim].append(t)
    case_agg = {cid: aggregate_trials(rows) for cid, rows in by_case.items()}
    suite_agg = {s: aggregate_trials(rows) for s, rows in by_suite.items()}
    dim_agg = {d: aggregate_trials(rows) for d, rows in by_dim.items()}
    weighted = []
    for suite, agg in suite_agg.items():
        w = SUITE_WEIGHTS.get(suite, 1.0)
        if agg.get("hard_pass_rate") is None or w == 0:
            continue
        weighted.append((w, agg["hard_pass_rate"]))
    overall = None
    if weighted:
        overall = sum(w * r for w, r in weighted) / sum(w for w, _ in weighted)
    return {
        "by_case": case_agg,
        "by_suite": suite_agg,
        "by_dimension": dim_agg,
        "weighted_suite_hard_pass_rate": overall,
        "n_trials": len(trials),
    }


def _rate_delta(instructed: Any, neutral: Any) -> Any:
    if instructed is None or neutral is None:
        return None
    try:
        return instructed - neutral
    except TypeError:
        return None


def _agg_delta(instructed: dict[str, Any] | None, neutral: dict[str, Any] | None) -> dict[str, Any] | None:
    if not instructed or not neutral:
        return None
    keys = set(instructed) | set(neutral)
    out: dict[str, Any] = {}
    for key in sorted(keys):
        a = instructed.get(key) or {}
        b = neutral.get(key) or {}
        out[key] = {
            "hard_pass_rate": _rate_delta(a.get("hard_pass_rate"), b.get("hard_pass_rate")),
            "instructed": a.get("hard_pass_rate"),
            "neutral": b.get("hard_pass_rate"),
            "n_counted_instructed": a.get("n_counted"),
            "n_counted_neutral": b.get("n_counted"),
        }
    return out


def build_summary(all_trials: list[dict[str, Any]], coding_rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_trials = [t for t in all_trials if not t.get("interrupted") and not t.get("in_progress")]
    coding_rows = [r for r in coding_rows if not r.get("interrupted") and not r.get("in_progress")]
    by_variant: dict[str, list] = defaultdict(list)
    for t in all_trials:
        if t.get("suite") == "coding":
            continue
        by_variant[str(t.get("prompt_variant") or "unknown")].append(t)
    variant_blocks = {name: _rates_block(rows) for name, rows in by_variant.items()}
    primary_name = "neutral" if "neutral" in variant_blocks else (next(iter(variant_blocks), None))
    primary = variant_blocks.get(primary_name) or _rates_block([])
    inst = variant_blocks.get("instructed")
    neu = variant_blocks.get("neutral")
    delta = None
    if inst and neu:
        delta = {
            "note": "instructed minus neutral. Positive = the scoring rubric prompt fixed a failure.",
            "weighted_suite_hard_pass_rate": _rate_delta(
                inst.get("weighted_suite_hard_pass_rate"),
                neu.get("weighted_suite_hard_pass_rate"),
            ),
            "by_dimension": _agg_delta(inst.get("by_dimension"), neu.get("by_dimension")),
            "by_case": _agg_delta(inst.get("by_case"), neu.get("by_case")),
        }
    return {
        "ground_truth": True,
        "do_not_rejudge": True,
        "primary_prompt_variant": primary_name,
        "by_prompt_variant": variant_blocks,
        "delta_instructed_minus_neutral": delta,
        "by_case": primary.get("by_case"),
        "by_suite": primary.get("by_suite"),
        "by_dimension": primary.get("by_dimension"),
        "weighted_suite_hard_pass_rate": primary.get("weighted_suite_hard_pass_rate"),
        "suite_weights": SUITE_WEIGHTS,
        "coding": [
            {
                "trial_id": r.get("trial_id"),
                "final_review": r.get("final_review"),
                "truncated": r.get("truncated"),
                "n_attempts": r.get("n_attempts"),
                "attempt_files": r.get("attempt_files"),
                "conversation": r.get("conversation"),
                "python_checks_ok": r.get("python_checks_ok"),
            }
            for r in coding_rows
        ],
        "n_trials": len(all_trials),
    }


def summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "GROUND TRUTH — do not rejudge mechanical fields.",
        f"primary_prompt_variant: {summary.get('primary_prompt_variant')}",
        f"weighted_suite_hard_pass_rate (primary): {summary.get('weighted_suite_hard_pass_rate')}",
        "",
        "== by prompt variant ==",
    ]
    for name, block in (summary.get("by_prompt_variant") or {}).items():
        lines.append(f"{name}: weighted={block.get('weighted_suite_hard_pass_rate')} n={block.get('n_trials')}")
    delta = summary.get("delta_instructed_minus_neutral")
    if delta:
        lines.append("")
        lines.append("== delta instructed − neutral (by dimension) ==")
        for dim, row in (delta.get("by_dimension") or {}).items():
            lines.append(
                f"{dim}: delta={row.get('hard_pass_rate')} instructed={row.get('instructed')} neutral={row.get('neutral')}"
            )
        lines.append(f"weighted delta: {delta.get('weighted_suite_hard_pass_rate')}")
    lines.append("")
    lines.append("== by dimension (primary) ==")
    for dim, agg in (summary.get("by_dimension") or {}).items():
        lines.append(f"{dim}: hard_pass_rate={agg.get('hard_pass_rate')} mode_agreement={agg.get('mode_agreement')} n={agg.get('n_counted')}")
    lines.append("")
    lines.append("== by case (primary) ==")
    for cid, agg in sorted((summary.get("by_case") or {}).items()):
        lines.append(f"{cid}: hard_pass_rate={agg.get('hard_pass_rate')} mode_agreement={agg.get('mode_agreement')} infra={agg.get('n_infra')}")
    lines.append("")
    lines.append("== coding ==")
    for row in summary.get("coding") or []:
        lines.append(
            f"{row.get('trial_id')} verdict={row.get('final_review')} truncated={row.get('truncated')} "
            f"attempts={row.get('n_attempts')} python_checks_ok={row.get('python_checks_ok')} "
            f"conversation={row.get('conversation')}"
        )
    return "\n".join(lines) + "\n"


def run_benchmark(
    cfg: dict[str, Any],
    *,
    profiles: list[str],
    suites: list[str],
    out_root: Path,
    prompt_variants: list[str] | None = None,
) -> Path:
    run_dir = out_root / _now_stamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(run_dir / "config.snapshot.yaml", snapshot_config(cfg))
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
                client = BenchClient(
                    base_url=cfg["base_url"],
                    api_key=str(cfg["api_key"]),
                    model=str(model["name"]),
                    interrupt_event=interrupt_event,
                    on_live_armed=on_live_armed,
                )
                register_client(client)
                tool_case_ids = [c for c in case_ids if c != "C01"]
                if tool_case_ids:
                    pending_line = True
                    t0 = line_start(f"preflight  {profile_name}")
                    client.raise_if_interrupted()
                    pre = run_preflight(client, profile)
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
                            if (catalog.get(cid) or {}).get("invalidate_if_no_parallel")
                        )
                    if tool_case_ids:
                        dialect = pre.get("template_dialect") or {}
                        skip_set.update(
                            cid
                            for cid in tool_case_ids
                            if dialect_skip_reason(catalog.get(cid) or {}, dialect)
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
                                        suite=catalog[cid]["suite"],
                                        profile=profile_name,
                                        variant=f"{variant_name}@t{temp}",
                                        repeat=0,
                                        nrep=1,
                                        seed=seed,
                                    )
                                )
                                res = run_tool_or_agent_case(
                                    client=client, case=bound, sampler=sampler, preflight=pre, repeat=0
                                )
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
                        nrep = repeats_for(cfg, case["suite"], profile_name)
                        for rep in range(nrep):
                            seed = choose_seed(profile, cid, rep, variant_name)
                            sampler = sampler_kwargs(profile, model, seed)
                            pending_line = True
                            tdir = dest / "cases" / cid
                            pending_trial_jsons = [tdir / f"trial_{rep + 1:03d}.json"]
                            t0 = line_start(
                                trial_label(
                                    case_id=cid,
                                    suite=case["suite"],
                                    profile=profile_name,
                                    variant=variant_name,
                                    repeat=rep,
                                    nrep=nrep,
                                    seed=seed,
                                )
                            )
                            res = run_tool_or_agent_case(
                                client=client, case=bound, sampler=sampler, preflight=pre, repeat=rep
                            )
                            trial_id = opaque_id(str(model["name"]), profile_name, variant_name, cid, str(rep))
                            trial = {
                                "trial_id": trial_id,
                                "case_id": cid,
                                "suite": case["suite"],
                                "repeat": rep,
                                "seed": seed,
                                "prompt_variant": variant_name,
                                **res,
                            }
                            payload = {k: trial[k] for k in trial if k != "messages"}
                            payload["messages"] = trial.get("messages")
                            write_json(tdir / f"trial_{rep + 1:03d}.json", payload)
                            write_text(tdir / f"trial_{rep + 1:03d}.txt", trial_txt(trial))
                            pending_trial_jsons = []
                            all_trials.append(trial)
                            hp = bool((trial.get("score") or {}).get("hard_pass"))
                            finish_trial_console(t0, hp, score_reasons(trial.get("score")))

                if "C01" in case_ids:
                    nrep = repeats_for(cfg, "coding", profile_name)
                    coding_client = BenchClient(
                        base_url=cfg["base_url"],
                        api_key=str(cfg["api_key"]),
                        model=str(model["name"]),
                        interrupt_event=interrupt_event,
                        on_live_armed=on_live_armed,
                    )
                    register_client(coding_client)
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
                        )
                        row.update({
                            "trial_id": trial_id,
                            "case_id": "C01",
                            "suite": "coding",
                            "repeat": rep,
                            "seed": seed,
                            "prompt_variant": None,
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

    summary = build_summary(all_trials, coding_rows)
    if interrupted:
        summary["interrupted"] = True
        write_text(
            run_dir / "INTERRUPTED.txt",
            f"interrupted by user\n{interrupt_at}\n",
        )
    write_json(run_dir / "summary.json", summary)
    write_text(run_dir / "summary.txt", summary_text(summary))
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
