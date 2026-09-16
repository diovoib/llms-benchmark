# HTTP API contract (Chat Completions with tools)

This document describes the HTTP Chat Completions API with tools.

## Endpoint

- `base_url` is a string supplied at runtime. Example: `http://127.0.0.1:8080/v1`. When the prefix `/v1` is used, it is already included in `base_url`.
- Request URL: `base_url` with trailing slashes removed, then the suffix `/chat/completions`. Example: `http://127.0.0.1:8080/v1/chat/completions`.
- Method: `POST`.
- The only header the program sets is `Content-Type: application/json`. Do not set `Authorization`, `Accept`, or any other header. Headers that the Python standard library adds by itself (for example `User-Agent` from `urllib`) are allowed.
- Do not put `Authorization`, `temperature`, `stream`, `max_tokens`, or any other field not listed below into the JSON body.
- HTTP timeout: 30 seconds.
- Request body bytes: `json.dumps(body, ensure_ascii=False).encode("utf-8")`.
- Response body bytes: decode as UTF-8, then `json.loads`.

## Transport errors (these raise)

`run_tool_chat` raises to its caller, and does not return `""`, when any of these happen:

- the TCP connection fails;
- the wait exceeds 30 seconds;
- the HTTP status is not `2xx`;
- the response body is not valid JSON.

Tool-dispatch failures below do not raise.

## First request JSON

Top-level keys are exactly these three, and no others:

- `model`: the `model` argument (string).
- `messages`: a new list that contains a **deep** copy of the `messages` argument (each element dict is a new object). Do not mutate the caller's list. Do not mutate any dict that was an element of the caller's list.
- `tools`: the `tools` argument if it is a list; otherwise `[]`.

## Second request JSON (only when the first response asks for tools)

Top-level keys are exactly these two, and no others:

- `model`: the same `model` argument.
- `messages`: that same new list, then the assistant message object from the first response, then one tool-result message per tool call, in the same order as `tool_calls`.

Do not include `tools` on the second request. Do not send a third request, even if the second response contains `tool_calls`.

## Message objects

User, system, and plain assistant messages:

`{ "role": "system"|"user"|"assistant"|"tool", "content": string|null }`

Assistant messages that call tools also contain `tool_calls`.

Each tool-result message:

`{ "role": "tool", "tool_call_id": <id or JSON null>, "content": "<string>" }`

If the tool call has no `id` or `id` is JSON `null`, set `tool_call_id` to JSON `null`.

## Tool-definition objects (`tools` array)

Each element has this shape.

```
{
  "type": "function",
  "function": {
    "name": "<string>",
    "description": "<string>",
    "parameters": {
      "type": "object",
      "properties": { "<arg name>": { "type": "...", ... } },
      "required": ["<arg name>", ...]
    }
  }
}
```

For a tool with no parameters, `properties` is `{}` and `required` is `[]`.

## Response JSON

```
{
  "choices": [
    {
      "finish_reason": "stop" | "tool_calls" | "length",
      "message": {
        "role": "assistant",
        "content": string | null,
        "tool_calls": [
          {
            "id": "call_...",
            "type": "function",
            "function": {
              "name": "string",
              "arguments": "<JSON object as a string>"
            }
          }
        ]
      }
    }
  ]
}
```

Treat the first response as **not** asking for tools when `choices` is missing or empty, or `choices[0]` has no `message` object, or `choices[0].message.tool_calls` is missing, `null`, or `[]`.

## Tool dispatch (these do not raise)

When the first response **asks for tools**, for each element of `tool_calls` in order:

1. Read `id` and `function.name`.
2. Parse `function.arguments`. If it is a string, `json.loads` it. If it is already a JSON object (a dict / `{}`), use it. If it is missing, use `{}`. If it is a JSON array, number, boolean, or `null`, or if `json.loads` fails, the tool `content` is the exception text or the string `arguments is not a JSON object`, and the callable is not invoked.
3. If `name` is not in `registry`, the tool `content` is `unknown tool ` plus the name.
4. If `name` is in `registry`, call `registry[name](**arguments)`. Extra keys in that dict are passed through. The tool `content` is `str` of the return value. If the callable raises any exception, the tool `content` is `str` of that exception.
5. Append `{ "role": "tool", "tool_call_id": <id or JSON null>, "content": <content string> }`.

Then send the second POST.

`run_tool_chat` does not raise because of steps 1–5.

## Return value

Read `choices[0].message.content` from the last HTTP response that returned JSON.

- If `choices` is missing or empty, or `choices[0]` has no `message` object, or `content` is missing or JSON `null`, return `""`.
- If `content` is a string, return that string.
- If `content` is any other JSON type, return `str(content)` (so JSON `0` becomes `"0"`, not `""`).

When the first response does not ask for tools: do not send a second POST. Return from the first response as above.
