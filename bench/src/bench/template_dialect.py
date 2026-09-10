from __future__ import annotations

import hashlib
import json
import re
import string
from typing import Any, Callable


CALL_SENTINEL = "PREFLIGHT_TOPIC_ZXQ9"
RESULT_SENTINEL = "PREFLIGHT_TOOL_RESULT_ZXQ9"
PROBE_TOOL_NAME = "get_news"
ALPHANUM = string.ascii_letters + string.digits
NATIVE_MARKERS = (
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<tools>",
    "</tools>",
    "[TOOL_CALLS]",
    "[TOOL_RESULTS]",
    "<|tool_call|>",
    "<|tool_response|>",
    "<|python_tag|>",
)
_LENGTH_RE = re.compile(r"length (\d+)", re.I)
_ALTERNATE_RE = re.compile(r"alternat", re.I)
_JINJA_EXCEPTION_RE = re.compile(
    r"(?:Error:\s*)?Jinja Exception:\s*(.+?)(?:\n|$)",
    re.I,
)
_RAISE_EXCEPTION_RE = re.compile(
    r"""raise_exception\(\s*["']([^"']{12,})["']\s*\)""",
    re.I,
)

# Keep native OpenAI tool_calls when possible; XML dialects are a fallback.
OPENAI_STRUCTURES = (
    "openai",
    "openai_empty",
    "openai_named",
    "openai_args_obj",
)
OTHER_STRUCTURES = (
    "ipython",
    "hermes",
    "hermes_tool_role",
    "qwen_tools",
)
STRUCTURES = OPENAI_STRUCTURES + OTHER_STRUCTURES
PROBE_SHAPES = ("simple", "padded", "parallel")


def case_needs_history_dialect(case: dict[str, Any]) -> bool:
    if case.get("messages_after_system"):
        return True
    if int(case.get("max_steps") or 1) > 1:
        return True
    return bool(case.get("followup_user"))


def case_needs_padded_history(case: dict[str, Any]) -> bool:
    """T16 (history already has assistant-after-tool) and A06 (follow-up after a tool round)."""
    return bool(case.get("messages_after_system") or case.get("followup_user"))


def dialect_skip_reason(case: dict[str, Any], dialect: dict[str, Any] | None) -> str | None:
    if not case_needs_history_dialect(case):
        return None
    dialect = dialect or {}
    if not dialect.get("ok"):
        return dialect.get("note") or "template dialect probe failed"
    if case_needs_padded_history(case) and not dialect.get("padded_history_ok", True):
        return dialect.get("padded_note") or "template rejects padded tool history"
    return None


def spec_from_example(example: str) -> dict[str, Any]:
    text = example or ""
    if text.isalnum():
        return {"kind": "alnum", "length": len(text), "prefix": "", "alphabet": ALPHANUM, "example": text}
    match = re.fullmatch(r"(call_)([A-Za-z0-9]+)", text)
    if match:
        return {
            "kind": "prefix",
            "length": len(text),
            "prefix": "call_",
            "alphabet": ALPHANUM,
            "example": text,
        }
    match = re.fullmatch(r"(call_pad_)(\d+)", text)
    if match:
        return {
            "kind": "prefix",
            "length": len(text),
            "prefix": "call_pad_",
            "alphabet": string.digits,
            "example": text,
        }
    return {"kind": "alnum", "length": 9, "prefix": "", "alphabet": ALPHANUM, "example": text}


def _alphabet(spec: dict[str, Any]) -> str:
    return str(spec.get("alphabet") or ALPHANUM)


def id_conforms(value: str, spec: dict[str, Any] | None) -> bool:
    if not spec or not value:
        return False
    prefix = str(spec.get("prefix") or "")
    length = int(spec.get("length") or 0)
    if length and len(value) != length:
        return False
    if prefix and not value.startswith(prefix):
        return False
    rest = value[len(prefix) :]
    allowed = set(_alphabet(spec))
    return bool(rest) and all(ch in allowed for ch in rest)


def make_id(source: str, spec: dict[str, Any], *, used: set[str] | None = None) -> str:
    prefix = str(spec.get("prefix") or "")
    length = int(spec.get("length") or 9)
    alphabet = _alphabet(spec)
    need = max(1, length - len(prefix))
    taken = used or set()
    seed = source or "id"
    for n in range(0, 4096):
        raw = hashlib.sha256(f"{seed}:{n}".encode("utf-8")).digest()
        chars: list[str] = []
        for byte in raw:
            chars.append(alphabet[byte % len(alphabet)])
            if len(chars) >= need:
                break
        while len(chars) < need:
            chars.append(alphabet[len(chars) % len(alphabet)])
        candidate = prefix + "".join(chars[:need])
        if length:
            candidate = (prefix + candidate[len(prefix) :])[:length]
            if len(candidate) < length:
                candidate = (candidate + alphabet[0] * length)[:length]
        if candidate not in taken:
            return candidate
    return (prefix + alphabet[0] * need)[:length]


def mapped_id(original: str, spec: dict[str, Any] | None, id_map: dict[str, str]) -> str:
    key = original or ""
    if key in id_map:
        return id_map[key]
    if not spec:
        id_map[key] = key
        return key
    if id_conforms(key, spec):
        id_map[key] = key
        return key
    value = make_id(key or "empty", spec, used=set(id_map.values()))
    id_map[key] = value
    return value


def _args_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _args_string(raw: Any) -> str:
    if isinstance(raw, str):
        return raw if raw.strip() else "{}"
    return json.dumps(raw if raw is not None else {}, ensure_ascii=False)


def _tool_call_payloads(message: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        out.append(
            {
                "id": call.get("id") or "",
                "name": fn.get("name") or call.get("name") or "",
                "arguments": fn.get("arguments") if "function" in call else call.get("arguments"),
            }
        )
    return out


def _openai_tool_call(call_id: str, name: str, arguments: Any, *, as_string: bool) -> dict[str, Any]:
    args = _args_string(arguments) if as_string else _args_object(arguments)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    }


def _xml_call_block(name: str, arguments: Any, *, tag: str, call_id: str = "") -> str:
    payload: dict[str, Any] = {"name": name, "arguments": _args_object(arguments)}
    if call_id:
        payload["id"] = call_id
    return f"<{tag}>\n{json.dumps(payload, ensure_ascii=False)}\n</{tag}>"


def _xml_result_block(content: str, *, tag: str = "tool_response") -> str:
    return f"<{tag}>\n{content}\n</{tag}>"


def _hermes_assistant(message: dict[str, Any], payloads: list[dict[str, Any]], *, tag: str) -> dict[str, Any]:
    blocks = [_xml_call_block(p["name"], p["arguments"], tag=tag, call_id=str(p.get("id") or "")) for p in payloads]
    text = message.get("content")
    prefix = text if isinstance(text, str) and text.strip() else ""
    body = "\n".join(blocks)
    content = f"{prefix}\n{body}".strip() if prefix else body
    return {"role": "assistant", "content": content}


def _join_content(left: Any, right: Any) -> Any:
    parts: list[str] = []
    for value in (left, right):
        if value is None or value == "":
            continue
        parts.append(str(value))
    if not parts:
        return left if left is not None else right
    return "\n".join(parts)


def merge_consecutive_roles(
    messages: list[dict[str, Any]],
    *,
    roles: frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    merge_roles = roles if roles is not None else frozenset({"user"})
    out: list[dict[str, Any]] = []
    for raw in messages:
        message = json.loads(json.dumps(raw))
        if out and message.get("role") == out[-1].get("role") and message.get("role") in merge_roles:
            prev = out[-1]
            prev["content"] = _join_content(prev.get("content"), message.get("content"))
            continue
        out.append(message)
    return out


def coalesce_history(messages: list[dict[str, Any]], dialect: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not dialect or not dialect.get("ok"):
        return messages
    if dialect.get("merge_consecutive"):
        return merge_consecutive_roles(messages)
    return messages


def _already_wrapped_tool_body(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "[TOOL_RESULTS]",
            "<tool_response>",
            "</tool_response>",
            "<tool_call>",
        )
    )


def _tool_as_user_content(message: dict[str, Any]) -> str:
    body = message.get("content")
    text = "" if body is None else str(body)
    if _already_wrapped_tool_body(text):
        return text
    call_id = message.get("tool_call_id")
    if call_id:
        return f"[TOOL_RESULTS]\n{call_id}\n{text}\n[/TOOL_RESULTS]"
    return f"[TOOL_RESULTS]\n{text}\n[/TOOL_RESULTS]"


def _apply_tool_role(messages: list[dict[str, Any]], tool_role: str) -> list[dict[str, Any]]:
    if tool_role != "user":
        return messages
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role in {"tool", "ipython"}:
            converted.append({"role": "user", "content": _tool_as_user_content(message)})
        else:
            converted.append(message)
    return converted


def adapt_messages(
    messages: list[dict[str, Any]],
    dialect: dict[str, Any] | None,
    id_map: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if not dialect or not dialect.get("ok"):
        return [json.loads(json.dumps(m)) for m in messages]
    spec = dialect.get("id_spec")
    structure = str(dialect.get("structure") or "openai")
    tool_role = str(dialect.get("tool_role") or "tool")
    mapping = id_map if id_map is not None else {}
    names_by_id: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            payloads = []
            for raw in _tool_call_payloads(message):
                new_id = mapped_id(str(raw["id"]), spec, mapping)
                if raw["name"]:
                    names_by_id[new_id] = str(raw["name"])
                    names_by_id[str(raw["id"])] = str(raw["name"])
                payloads.append(
                    {
                        "id": new_id,
                        "name": raw["name"],
                        "arguments": raw["arguments"],
                    }
                )
            if structure in {"hermes", "hermes_tool_role"}:
                out.append(_hermes_assistant(message, payloads, tag="tool_call"))
            elif structure == "qwen_tools":
                out.append(_hermes_assistant(message, payloads, tag="tools"))
            else:
                as_string = structure != "openai_args_obj"
                content: Any
                if structure == "openai_empty":
                    content = message.get("content") if message.get("content") not in (None, "") else ""
                    if content is None:
                        content = ""
                else:
                    content = message.get("content")
                adapted = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        _openai_tool_call(p["id"], p["name"], p["arguments"], as_string=as_string)
                        for p in payloads
                    ],
                }
                out.append(adapted)
            continue
        if role == "tool":
            original = str(message.get("tool_call_id") or "")
            new_id = mapped_id(original, spec, mapping)
            content = message.get("content")
            if structure == "hermes":
                out.append({"role": "user", "content": _xml_result_block(str(content or ""))})
            elif structure == "hermes_tool_role":
                out.append({"role": "tool", "tool_call_id": new_id, "content": _xml_result_block(str(content or ""))})
            elif structure == "qwen_tools":
                out.append({"role": "user", "content": _xml_result_block(str(content or ""))})
            elif structure == "ipython":
                out.append({"role": "ipython", "content": content})
            else:
                tool_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": new_id,
                    "content": content,
                }
                if structure == "openai_named":
                    tool_msg["name"] = message.get("name") or names_by_id.get(new_id) or names_by_id.get(original) or ""
                out.append(tool_msg)
            continue
        out.append(json.loads(json.dumps(message)))
    out = _apply_tool_role(out, tool_role)
    if dialect.get("merge_consecutive", tool_role == "user"):
        out = merge_consecutive_roles(out)
    return out


def _id_candidates(issued_ids: list[str]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    def add(name: str, example: str) -> None:
        if not example or example in seen:
            return
        seen.add(example)
        out.append({"name": name, "example": example, "id_spec": spec_from_example(example)})

    for i, issued in enumerate(issued_ids):
        add(f"issued_{i}", str(issued))
    add("alnum9", make_id("probe-alnum9", spec_from_example("a1b2c3d4e")))
    add("alnum8", make_id("probe-alnum8", spec_from_example("a1b2c3d4")))
    add("alnum10", make_id("probe-alnum10", spec_from_example("a1b2c3d4e5")))
    add("alnum32", make_id("probe-alnum32", spec_from_example("a" * 32)))
    add("call8", make_id("probe-call8", spec_from_example("call_" + "a" * 8)))
    add("call_pad_0", "call_pad_0")
    return out


def _id_matches_hint(spec: dict[str, Any], hint: dict[str, Any] | None) -> bool:
    if not hint:
        return True
    if hint.get("kind") == "alnum":
        if spec.get("prefix"):
            return False
        if "length" in hint and int(spec.get("length") or 0) != int(hint["length"]):
            return False
        return str(spec.get("alphabet") or "") == ALPHANUM or not spec.get("alphabet")
    return True


def unwrap_template_error(error: str | None) -> str | None:
    """Pull the Jinja/runtime message out of llama.cpp parser-wrapper text."""
    if not error:
        return None
    text = str(error)
    match = _JINJA_EXCEPTION_RE.search(text)
    if match:
        return " ".join(match.group(1).split())
    match = _RAISE_EXCEPTION_RE.search(text)
    if match:
        inner = " ".join(match.group(1).split())
        if not inner.endswith("..."):
            return inner
    return " ".join(text.split())


def _hint_blob(error: str | None) -> str:
    if not error:
        return ""
    inner = unwrap_template_error(error) or ""
    return f"{error}\n{inner}"


def id_hints_from_error(error: str | None) -> dict[str, Any] | None:
    blob = _hint_blob(error)
    if not blob:
        return None
    low = blob.lower()
    if "alphanumeric" not in low and "isalnum" not in low:
        return None
    match = _LENGTH_RE.search(blob)
    if not match:
        return None
    return {"kind": "alnum", "length": int(match.group(1))}


def error_implies_role_alternate(error: str | None) -> bool:
    return bool(_ALTERNATE_RE.search(_hint_blob(error)))


def _structure_uses_non_ua_result_role(structure: str) -> bool:
    return structure not in {"hermes", "qwen_tools"}


def _assistant_probe(example_id: str, structure: str, *, extra_ids: list[str] | None = None) -> dict[str, Any]:
    arguments = json.dumps({"topic": CALL_SENTINEL}, ensure_ascii=False)
    ids = [example_id, *(extra_ids or [])]
    as_string = structure != "openai_args_obj"
    content: Any = "" if structure == "openai_empty" else None
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            _openai_tool_call(call_id, PROBE_TOOL_NAME, arguments, as_string=as_string) for call_id in ids
        ],
    }


def _tool_probe(example_id: str, structure: str) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": example_id,
        "content": RESULT_SENTINEL,
    }
    if structure == "openai_named":
        message["name"] = PROBE_TOOL_NAME
    return message


def build_probe_messages(example_id: str, dialect: dict[str, Any], *, shape: str = "simple") -> list[dict[str, Any]]:
    structure = str(dialect.get("structure") or "openai")
    spec = dialect.get("id_spec") or spec_from_example(example_id)
    second_id = make_id(f"{example_id}:b", spec, used={example_id})
    user = {"role": "user", "content": "Get the latest news."}
    if shape == "parallel":
        canonical = [
            user,
            _assistant_probe(example_id, structure, extra_ids=[second_id]),
            _tool_probe(example_id, structure),
            _tool_probe(second_id, structure),
        ]
    elif shape == "padded":
        canonical = [
            user,
            _assistant_probe(example_id, structure),
            _tool_probe(example_id, structure),
            {"role": "assistant", "content": "Looking that up."},
            {"role": "user", "content": "What is the weather?"},
        ]
    else:
        canonical = [
            user,
            _assistant_probe(example_id, structure),
            _tool_probe(example_id, structure),
        ]
    return adapt_messages(canonical, dialect, {})


def score_prompt(prompt: str | None) -> dict[str, Any]:
    text = prompt or ""
    call = CALL_SENTINEL in text
    result = RESULT_SENTINEL in text
    native = any(marker in text for marker in NATIVE_MARKERS)
    dump = '"tool_calls"' in text or '"tool_call_id"' in text
    points = 0
    if call:
        points += 4
    if result:
        points += 4
    if native:
        points += 2
    if dump and not native:
        points -= 1
    snippets: list[str] = []
    for needle in (CALL_SENTINEL, RESULT_SENTINEL):
        idx = text.find(needle)
        if idx >= 0:
            start = max(0, idx - 40)
            snippets.append(text[start : idx + len(needle) + 40])
    return {
        "points": points,
        "ok": call and result,
        "renders_call": call,
        "renders_result": result,
        "native_markers": native,
        "openai_dump": dump,
        "snippets": snippets,
    }


def _truncate_error(error: str | None, limit: int = 240) -> str | None:
    if error is None:
        return None
    text = unwrap_template_error(error) or " ".join(str(error).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _error_text(payload: Any) -> str:
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        if isinstance(err, str):
            return err
        if payload.get("message"):
            return str(payload["message"])
    return str(payload)


class RenderOutcome:
    def __init__(
        self,
        *,
        ok: bool,
        prompt: str | None = None,
        error: str | None = None,
        prompt_tokens: int | None = None,
        via: str = "",
        missing: bool = False,
    ) -> None:
        self.ok = ok
        self.prompt = prompt
        self.error = error
        self.prompt_tokens = prompt_tokens
        self.via = via
        self.missing = missing


class TemplateRenderer:
    def __init__(
        self,
        *,
        apply_template: Callable[[list[dict[str, Any]], list[dict[str, Any]]], RenderOutcome],
        chat: Callable[..., RenderOutcome],
    ) -> None:
        self._apply_template = apply_template
        self._chat = chat
        self.apply_available: bool | None = None

    def render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> RenderOutcome:
        if self.apply_available is not False:
            outcome = self._apply_template(messages, tools)
            if outcome.missing:
                self.apply_available = False
            else:
                self.apply_available = True
                return outcome
        return self._chat(messages, tools, max_tokens=1)

    def confirm(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> RenderOutcome:
        return self._chat(messages, tools, max_tokens=1)


def _dialect_for_probe(*, structure: str, spec: dict[str, Any], tool_role: str) -> dict[str, Any]:
    return {
        "ok": True,
        "structure": structure,
        "id_spec": spec,
        "tool_role": tool_role,
        "merge_consecutive": tool_role == "user" or structure in {"hermes", "qwen_tools"},
    }


def _evaluate_shapes(
    renderer: TemplateRenderer,
    dialect: dict[str, Any],
    example_id: str,
    tools: list[dict[str, Any]],
    *,
    shapes: tuple[str, ...],
    id_hint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shape_ok: dict[str, bool] = {}
    errors: dict[str, str | None] = {}
    scored: dict[str, Any] | None = None
    via = ""
    chat_only = False
    role_hint: str | None = None
    spec = dialect.get("id_spec")
    for shape in shapes:
        if id_hint and spec and not _id_matches_hint(spec, id_hint):
            break
        messages = build_probe_messages(example_id, dialect, shape=shape)
        outcome = renderer.render(messages, tools)
        via = outcome.via or via
        chat_only = outcome.prompt is None
        if not outcome.ok:
            shape_ok[shape] = False
            errors[shape] = _truncate_error(outcome.error)
            id_hint = id_hints_from_error(outcome.error) or id_hint
            if error_implies_role_alternate(outcome.error):
                role_hint = "user"
            if id_hint and spec and not _id_matches_hint(spec, id_hint):
                break
            continue
        shape_ok[shape] = True
        errors[shape] = None
        if outcome.prompt is not None:
            got = score_prompt(outcome.prompt)
            if scored is None or shape == "padded":
                scored = got
            elif not scored.get("ok") and got.get("ok"):
                scored = got
        elif scored is None:
            scored = {
                "points": 6,
                "ok": True,
                "renders_call": None,
                "renders_result": None,
                "native_markers": None,
                "snippets": [],
            }
    return {
        "shape_ok": shape_ok,
        "errors": errors,
        "scored": scored or {},
        "via": via,
        "chat_only": chat_only,
        "id_hint": id_hint,
        "role_hint": role_hint,
        "messages_padded": build_probe_messages(example_id, dialect, shape="padded"),
        "messages_simple": build_probe_messages(example_id, dialect, shape="simple"),
    }


def _winner_payload(
    *,
    cand: dict[str, Any],
    structure: str,
    tool_role: str,
    eval_row: dict[str, Any],
    via: str,
    tried: list[dict[str, Any]],
    id_hint: dict[str, Any] | None,
    role_hint: str | None,
    confirmed: bool,
) -> dict[str, Any]:
    scored = eval_row["scored"]
    shape_ok = eval_row["shape_ok"]
    simple_ok = bool(shape_ok.get("simple"))
    padded_ok = bool(shape_ok.get("padded"))
    return {
        "ok": simple_ok,
        "via": via,
        "id_spec": cand["id_spec"],
        "structure": structure,
        "tool_role": tool_role,
        "merge_consecutive": tool_role == "user" or structure in {"hermes", "qwen_tools"},
        "simple_roundtrip_ok": simple_ok,
        "padded_history_ok": padded_ok,
        "parallel_results_ok": bool(shape_ok.get("parallel")),
        "renders_call": scored.get("renders_call"),
        "renders_result": scored.get("renders_result"),
        "native_markers": scored.get("native_markers"),
        "chat_confirmed": confirmed,
        "score": scored.get("points"),
        "prompt_snippets": scored.get("snippets") or [],
        "tried": tried,
        "hint": id_hint,
        "role_hint": role_hint,
    }


def probe_template_dialect(
    renderer: TemplateRenderer,
    *,
    tools: list[dict[str, Any]],
    issued_ids: list[str],
) -> dict[str, Any]:
    tried: list[dict[str, Any]] = []
    ranked: list[tuple[Any, ...]] = []
    id_hint: dict[str, Any] | None = None
    role_hint: str | None = None
    via = ""
    id_cands = _id_candidates(issued_ids)

    def consider_error(error: str | None) -> None:
        nonlocal id_hint, role_hint
        id_hint = id_hints_from_error(error) or id_hint
        if error_implies_role_alternate(error):
            role_hint = "user"

    allowed = STRUCTURES
    if renderer.apply_available is False:
        allowed = ("openai", "openai_empty", "hermes")
    # Native OpenAI tool_calls first; rewrite results as user before falling back to XML.
    passes = (
        ("tool", OPENAI_STRUCTURES),
        ("user", OPENAI_STRUCTURES),
        ("tool", OTHER_STRUCTURES),
        ("user", OTHER_STRUCTURES),
    )

    for tool_role, group in passes:
        for structure in group:
            if structure not in allowed:
                continue
            if role_hint == "user" and tool_role == "tool" and _structure_uses_non_ua_result_role(structure):
                continue
            for cand in id_cands:
                if role_hint == "user" and tool_role == "tool" and _structure_uses_non_ua_result_role(structure):
                    break
                spec = cand["id_spec"]
                if not _id_matches_hint(spec, id_hint):
                    continue
                dialect = _dialect_for_probe(structure=structure, spec=spec, tool_role=tool_role)
                eval_row = _evaluate_shapes(
                    renderer,
                    dialect,
                    cand["example"],
                    tools,
                    shapes=PROBE_SHAPES,
                    id_hint=id_hint,
                )
                via = eval_row["via"] or via
                id_hint = eval_row.get("id_hint") or id_hint
                role_hint = eval_row.get("role_hint") or role_hint
                shape_ok = eval_row["shape_ok"]
                scored = eval_row["scored"]
                simple_ok = bool(shape_ok.get("simple"))
                padded_ok = bool(shape_ok.get("padded"))
                parallel_ok = bool(shape_ok.get("parallel"))
                first_error = next((e for e in eval_row["errors"].values() if e), None)
                row = {
                    "id": cand["name"],
                    "structure": structure,
                    "tool_role": tool_role,
                    "simple": simple_ok,
                    "padded": padded_ok,
                    "parallel": parallel_ok,
                    "via": eval_row["via"],
                    "points": scored.get("points"),
                    "error": first_error,
                }
                tried.append(row)
                if not simple_ok:
                    continue
                if eval_row["chat_only"]:
                    prompt_ok = True
                else:
                    prompt_ok = bool(scored.get("ok"))
                if not prompt_ok:
                    continue
                # Prefer native OpenAI tool_calls and keeping role=tool when the template allows it.
                rank = (
                    1 if padded_ok else 0,
                    1 if parallel_ok else 0,
                    int(scored.get("points") or 0),
                    1 if structure in OPENAI_STRUCTURES else 0,
                    1 if tool_role == "tool" else 0,
                    -len(ranked),
                )
                ranked.append((rank, structure, tool_role, cand, eval_row, eval_row["chat_only"]))
                if padded_ok and parallel_ok and structure in OPENAI_STRUCTURES:
                    # Good enough: confirm and stop searching weaker dialects.
                    confirm_msgs = eval_row["messages_padded"]
                    if eval_row["chat_only"]:
                        confirmed = True
                    else:
                        confirm = renderer.confirm(confirm_msgs, tools)
                        via = confirm.via or via
                        confirmed = bool(confirm.ok)
                        if not confirmed:
                            consider_error(confirm.error)
                            tried.append(
                                {
                                    "id": cand["name"],
                                    "structure": structure,
                                    "tool_role": tool_role,
                                    "accepted": False,
                                    "via": confirm.via,
                                    "error": _truncate_error(confirm.error),
                                    "note": "chat confirm failed",
                                }
                            )
                            continue
                    return _winner_payload(
                        cand=cand,
                        structure=structure,
                        tool_role=tool_role,
                        eval_row=eval_row,
                        via=via,
                        tried=tried,
                        id_hint=id_hint,
                        role_hint=role_hint,
                        confirmed=confirmed,
                    )

    ranked.sort(key=lambda item: item[0], reverse=True)
    for _rank, structure, tool_role, cand, eval_row, already_chatted in ranked:
        shape_ok = eval_row["shape_ok"]
        confirm_msgs = eval_row["messages_padded"] if shape_ok.get("padded") else eval_row["messages_simple"]
        if already_chatted:
            confirmed = True
        else:
            confirm = renderer.confirm(confirm_msgs, tools)
            via = confirm.via or via
            confirmed = bool(confirm.ok)
            if not confirmed:
                consider_error(confirm.error)
                tried.append(
                    {
                        "id": cand["name"],
                        "structure": structure,
                        "tool_role": tool_role,
                        "accepted": False,
                        "via": confirm.via,
                        "error": _truncate_error(confirm.error),
                        "note": "chat confirm failed",
                    }
                )
                continue
        return _winner_payload(
            cand=cand,
            structure=structure,
            tool_role=tool_role,
            eval_row=eval_row,
            via=via,
            tried=tried,
            id_hint=id_hint,
            role_hint=role_hint,
            confirmed=confirmed,
        )

    return {
        "ok": False,
        "via": via,
        "id_spec": None,
        "structure": None,
        "tool_role": None,
        "merge_consecutive": False,
        "simple_roundtrip_ok": False,
        "padded_history_ok": False,
        "parallel_results_ok": False,
        "renders_call": False,
        "renders_result": False,
        "native_markers": False,
        "chat_confirmed": False,
        "score": None,
        "prompt_snippets": [],
        "tried": tried,
        "hint": id_hint,
        "role_hint": role_hint,
        "note": "no id/structure/role candidate rendered tool history and passed a chat confirm",
    }
