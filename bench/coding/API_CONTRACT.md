# API contract (implement this exactly)

Base URL is provided at runtime as `base_url` (example: `http://127.0.0.1:9PORT/v1`). It already includes the `/v1` prefix when present.

## Request

- Method: `POST`
- Path: `{base_url}/chat/completions`
- Header: `Content-Type: application/json`
- JSON body allowed keys only:
  - `model` (string, required)
  - `messages` (array, required)
  - `tools` (array, optional; include when the caller passed tools)
- Do not send any other top-level JSON keys.

## Message objects

- `{ "role": "system"|"user"|"assistant"|"tool", "content": string|null }`
- Assistant messages that call tools also have `tool_calls`.
- Tool-result messages: `{ "role": "tool", "tool_call_id": "<id>", "content": "<string>" }`

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
              "arguments": "<JSON object as string>"
            }
          }
        ]
      }
    }
  ]
}
```

`tool_calls` may be omitted when the model is not calling a tool.

## Required client behaviour

Function to implement in `tool_client.py`:

`run_tool_chat(base_url, model, messages, tools, registry) -> str`

1. POST the first request with `messages` and `tools`.
2. If `choices[0].message.tool_calls` is present and non-empty:
   - For each call, parse `function.arguments` as JSON object.
   - Call `registry[name](**arguments)` to get a string result. If the callable raises, the returned string of the overall function must still be produced without raising to the caller; treat the exception text as the tool content.
   - Append the assistant message, then one `role=tool` message per call with the matching `tool_call_id`.
   - POST a second request with the updated `messages`. Do not send `tools` on the second request.
3. Return `choices[0].message.content` from the last response as a string (`""` if null).
4. If there are no tool calls, return the first response content and do not POST again.

Do not use the network at import time. Only `run_tool_chat` may perform HTTP.
