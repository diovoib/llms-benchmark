# Task specification

This document lists the requirements for the program `tool_client.py`. Each item is a requirement.

`API_CONTRACT.md` is not a second numbered requirements list. It describes the HTTP API. The program must speak that API.

## Requirements

**R1** The module defines a function with this exact signature and return annotation:

`run_tool_chat(base_url, model, messages, tools, registry) -> str`

The five parameters are positional, in that order. The function implements the client described in `API_CONTRACT.md`. Failures while talking to the HTTP server raise to the caller of `run_tool_chat` (see `API_CONTRACT.md`, transport errors). Failures while dispatching a tool do not raise (see `API_CONTRACT.md`, tool dispatch). `main()` exists to catch transport errors; that is not a contradiction.

**R2** The module imports and uses only the Python standard library.

**R3** Importing the module does not open a network connection. The only function that may perform HTTP is `run_tool_chat`. `main()` may call `run_tool_chat`; it must not open HTTP itself.

**R4** The module defines a list named `TOOLS` and a dict named `REGISTRY`. Each element of `TOOLS` has the tool-definition shape in `API_CONTRACT.md`. Each key of `REGISTRY` is a tool name and each value is a Python callable. `run_tool_chat` uses the `tools` and `registry` arguments passed by the caller. It does not substitute `TOOLS` or `REGISTRY` for those arguments.

**R5** `TOOLS` and `REGISTRY` contain exactly three tools, named `get_current_time`, `calculate`, and one third name that you choose.

**R6** `get_current_time` has no parameters. It returns a `str`: current UTC time in ISO-8601 with a trailing `Z` and second precision, for example `2026-09-16T18:50:00Z`. Its `TOOLS` entry uses the no-parameter form from `API_CONTRACT.md`: `properties` is `{}` and `required` is `[]`.

**R7** `calculate` has three parameters: `a` (integer), `b` (integer), `operation` (string, one of `+`, `-`, `*`, `/`). It returns a `str` and does not raise. For `+`, `-`, and `*` the string is the integer result in decimal. For `/` with `b` equal to `0` the string is exactly `division by zero`. For `/` with `b` not `0` the string is `str(a / b)` (true division). If `operation` is not one of `+`, `-`, `*`, `/`, the string is `unknown operation ` followed by that value. Its `TOOLS` entry has `parameters.properties` with `a` `{ "type": "integer" }`, `b` `{ "type": "integer" }`, `operation` `{ "type": "string", "enum": ["+", "-", "*", "/"] }`, and `required` exactly `["a", "b", "operation"]`. If the client passes JSON types that do not match those Python types, `calculate` may raise; `run_tool_chat` then turns that exception into tool `content` as in `API_CONTRACT.md`. Extra keys in `arguments` are not stripped by the client.

**R8** The third tool's callable returns a `str`, does not perform HTTP, and performs one simple useful local action. You invent its name, parameters, and behaviour. Its `TOOLS` entry uses the same tool-definition shape as in `API_CONTRACT.md`, with `name`, `description`, and `parameters` that match that callable.

**R9** The module defines `main()` and ends with `if __name__ == "__main__": main()`. Importing the module does not call `main()`.

**R10** `main()` parses command-line arguments with `argparse`. Flags: `--base-url` (string, default `http://127.0.0.1:8080/v1`), `--model` (string, required), `--message` (string, default `What is the current UTC time? What is 7 * 6?`).

**R11** `main()` calls `run_tool_chat(base_url, model, messages, TOOLS, REGISTRY)` where `messages` is `[{"role": "user", "content": <message>}]` using the parsed flags. It writes the returned string to standard output, then a newline. If that call raises, it writes the exception text to standard error and exits with status `1`.
