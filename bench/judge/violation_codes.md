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
| INVENTED_API | unused | Not used. C01 does not run hidden HTTP tests against `tool_client.py`. |
| SPEC_ITEM_MISSING | unused | Not used. C01 does not run hidden spec-item tests. Mechanical C01 checks are `python_checks.json` (syntax) and `FINAL_REVIEW` / truncation. |
| REGRESSION | unused | Leftover from the old multi-round coding harness. |
| CONTEXT_OVERFLOW | infra | Server error text reports context / n_ctx overflow. Exclude from hard_pass_rate. |
| INFRA_ERROR | infra | Connection failure or 5xx/HTTP failure that is not context overflow. A stream that stayed silent longer than min(request_timeout_s / 2, 5s) before the HTTP deadline, or a non-stream deadline with no token timeline, is this code. Exclude from hard_pass_rate. |
| BENCH_INTERNAL_ERROR | infra | The harness was about to send a chat request whose history contains tool-call arguments that are not a JSON object. That is a bench bug (missed a bad model turn, mis-parsed it, or assembled the next request wrongly). Do not rewrite history to hide it. Exclude from hard_pass_rate. |
| CASE_GENERATION_TIMEOUT | yes | Stream still produced content (text or tool-call deltas) within min(request_timeout_s / 2, 5s) of the suite HTTP deadline. Same class of fail as max_tokens / length: the model did not finish. Counted in hard_pass_rate. |
| JUDGE_UNSTABLE | validation | Two judge runs disagree on scores for a trial_id. |
| RUBBER_STAMP_REVIEW | judge only | C01: OK (or empty critique) with no real check of the code in the conversation. |
| ITERATION_PROTOCOL_BREAK | n/a for C01 | Unused for current C01 (no file tools). |
| MISSING_SUBMIT_REVIEW | n/a for C01 | Unused for current C01 (no `submit_review` tool). |

Mechanical codes are already in `summary.json` in this directory. The judge must not recompute them. Judge-only interpretation: `RUBBER_STAMP_REVIEW`, plus whether a no-tool clarification was appropriate.
