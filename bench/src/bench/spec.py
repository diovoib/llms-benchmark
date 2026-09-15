from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PromptVariant = Literal["neutral", "helpful", "instructed", "harness"]
ToolChoice = Literal["auto", "none"]


@dataclass
class Invocation:
    name: str
    """Catalog function name the model must call (or that this spec describes)."""

    arguments: dict[str, Any] = field(default_factory=dict)
    """Argument matchers keyed by JSON field. Empty means name-only: any arguments are left to other Expect rules."""


@dataclass
class LaterStep:
    after: str
    """Tool name that must appear in an earlier agent step than `tool`."""

    tool: str
    """Tool name that must appear in a later step than `after`. Same turn as `after` fails WRONG_TOOL."""


@dataclass
class Expect:
    """Mechanical scorer contract for one case. Every field maps to a hard_score check."""

    required_calls: list[Invocation] = field(default_factory=list)
    """Calls that must occur, with no extras. Empty list means zero tools. Extra, missing, or wrong-shape calls → WRONG_TOOL."""

    calls_in_order: bool = False
    """If True, required_calls must appear in list order. If False, any order is accepted."""

    forbidden_optional_keys: list[str] = field(default_factory=list)
    """Optional parameter names from the advertised tool schema that the user did not supply. If the model sends any of these keys → INVENTED_ARG."""

    final_answer_must_include: list[str] = field(default_factory=list)
    """Substrings that must appear in the last assistant text. Miss → IGNORED_OBSERVATION."""

    must_not_claim_success_tokens: list[str] = field(default_factory=list)
    """Substrings forbidden in the last assistant text unless a tool payload already contained them. Else IGNORED_OBSERVATION."""

    require_later_step: LaterStep | None = None
    """If set, `tool` must be invoked in a later step than `after`. Same-turn or missing → WRONG_TOOL."""


@dataclass
class Case:
    id: str
    """Stable id (T01_pl, A06_en, …). Config, results paths, and dimension stems key off this."""

    suite: Literal["tools", "agent", "coding", "verification"]
    """Which runner loop and suite weight apply. tools is typically one model turn; agent is the multi-step loop."""

    purpose: str
    """CASE.md: why this case exists. Not scored."""

    expected_result: str
    """CASE.md: what mechanical pass/fail means. Not scored."""

    expect: Expect
    """Scorer contract. hard_score reads these fields; it does not parse purpose text."""

    prompt: str = ""
    """User utterance when messages_after_system is unset. Becomes the first user message after the variant system prompt."""

    tools: list[dict[str, Any]] = field(default_factory=list)
    """OpenAI tool schemas advertised this trial. Hallucinated names → TOOL_HALLUCINATION; extra catalog picks → WRONG_TOOL."""

    max_steps: int = 1
    """Cap on model turns. The runner never exceeds this. If max_steps > 1 and the last turn is still a tool call, score NO_STOP."""

    tool_choice: ToolChoice = "auto"
    """Chat Completions tool_choice. This bench uses auto or none (T08)."""

    parallel_tool_calls: bool | None = None
    """If True, request parallel tool calls. None leaves the API default."""

    invalidate_if_no_parallel: bool = False
    """If True, skip this case when preflight did not confirm parallel tool calls."""

    messages_after_system: list[dict[str, Any]] | None = None
    """Padded history (user/assistant/tool dicts) used instead of a single prompt. Dialect must support history."""

    followup_user: str | None = None
    """If set, injected as a new user turn after the first tool round (A06)."""

    error_cities: list[str] = field(default_factory=list)
    """Cities for which get_current_weather mock returns UNKNOWN_CITY instead of a temperature."""

    policy_suffix: str = ""
    """Appended to the active system prompt (A04/A07 POLICY_CONFIRM)."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    """Runtime chat transcript after apply_prompt_variant. Wire-format dicts."""

    system: str = ""
    """Runtime system text actually sent (variant + policy_suffix)."""
