# Coding spec

Implement `tool_client.py` in the sandbox with `run_tool_chat(base_url, model, messages, tools, registry) -> str` as defined in `API_CONTRACT.md`.

Each item is a requirement.

- **S1** `run_tool_chat` exists and is callable with parameters `base_url`, `model`, `messages`, `tools`, `registry`.
- **S2** `foo = run_tool_chat(...)` sends HTTP POST whose URL path ends with `/chat/completions`.
- **S3** The first JSON body contains both `messages` and `tools`.
- **S4** When the first response contains `tool_calls`, the matching `registry[name]` is invoked.
- **S5** The second request JSON includes a message with `"role": "tool"`.
- **S6** That tool message `tool_call_id` equals the `id` from the first response tool call.
- **S7** Exactly two HTTP POSTs occur when the first response contains tool calls.
- **S8** The returned string contains the unique token from the stub's final assistant message. That token appears only in the live HTTP response; it is not written in this spec.
- **S9** When the first response has no `tool_calls`, the function returns that content and performs one POST.
- **S10** If `registry[name]` raises `ValueError`, `run_tool_chat` does not raise.
- **S11** First request JSON top-level keys are only `model`, `messages`, and `tools` (no invented fields).
- **S12** Importing `tool_client` performs no TCP connect.

Keep the complete program in the chat.
