# Violation codes (closed list)

Use only these strings. Do not invent codes.

| code | mechanical? | meaning |
| --- | --- | --- |
| WRONG_TOOL | yes | Called a catalog tool other than the required one, or missed a required tool. |
| TOOL_HALLUCINATION | yes | Called a name not in the provided catalog. |
| MISSING_REQUIRED_ARG | yes | Required schema key absent. |
| INVENTED_ARG | yes | Extra key, or optional key filled without user mention. |
| BAD_JSON_TYPE | yes | JSON type mismatch (string `"21"` vs integer `21`, `"true"` vs boolean). |
| ENUM_OUT_OF_RANGE | yes | Value not in schema enum. |
| DUPLICATE_CALL | yes | Same tool name twice in one assistant turn. |
| LEAKED_TOOL_FORMAT | yes | Assistant `content` contains a native tool-template marker (`<tool_call>`, `<|tool_call|>`, `<tools>`, `[TOOL_CALLS]`, `<|python_tag|>`, `<invoke`, `<function`, `tool call: {`, …). |
| IGNORED_OBSERVATION | yes | Final text missing required observation token, or success claimed after error-only tools. |
| NO_STOP | yes | Agent loop still emitting tool_calls at max_steps; or C01 ended without `FINAL_REVIEW: OK` / `FINAL_REVIEW: NOK` (including `finish_reason=length` / context overflow). |
| INVENTED_API | hidden tests | Request path or JSON keys outside the given contract. |
| SPEC_ITEM_MISSING | hidden tests | Spec id failed, or `tool_client.py` missing/unparseable. |
| REGRESSION | unused | Leftover from the old multi-round coding harness. |
| CONTEXT_OVERFLOW | infra | Server error text reports context / n_ctx overflow. Exclude from hard_pass_rate. |
| INFRA_ERROR | infra | Connection failure or 5xx/HTTP failure that is not context overflow. Exclude from hard_pass_rate. |
| TIMEOUT | infra | Wall-clock deadline for that HTTP call expired. May be a model that never finished generating, or a server that stopped responding; the bench does not distinguish. Counted in n_infra. |
| JUDGE_UNSTABLE | validation | Two judge runs disagree on scores for a trial_id. |
| RUBBER_STAMP_REVIEW | judge only | C01: OK (or empty critique) with no real check of the code in the conversation. |
| ITERATION_PROTOCOL_BREAK | n/a for C01 | Unused for current C01 (no file tools). |
| MISSING_SUBMIT_REVIEW | n/a for C01 | Unused for current C01 (no `submit_review` tool). |

Mechanical codes are already in `summary.json` in this directory. The judge must not recompute them. Judge-only interpretation: `RUBBER_STAMP_REVIEW`, plus whether a no-tool clarification was appropriate.
