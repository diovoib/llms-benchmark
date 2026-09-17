**Language:** [Polski](README.pl.md) · English

GitHub and GitLab always render this file (`README.md`) by default. Click above to switch to another language.

# Local agent benchmark (tools + loop + coding)

This directory is for evaluating a **small local LLM** as an agent engine: the main emphasis here is whether it correctly calls the tools the harness exposes to it, whether it can organize a **tool** → **result** → **next-step** loop, and whether on a longer coding task it can check its own work, make actual progress, and decide when the task is done.

The model server is outside the benchmark and is started independently. It can be any server (llama-server, Ollama, LM Studio, vLLM) that exposes an OpenAI API and supports tool calling in the way the model requires. The benchmark calls `POST {base_url}/chat/completions` through the official OpenAI SDK (`tools` / `tool_calls` / `role: tool`).

The final outcome should be assessed by a judge — a human or a larger LLM — to verify “Did the agent actually complete the assigned task?” Sometimes models produce a textual description and the task is done, but it would not necessarily match a regexp. Sometimes they produce too much: the answer looks like it matches a regexp, but it is a question, or it contains leftover XML/JSON template fragments the model uses.


## Requirements

- A running OpenAI-compatible server (llama: `http://127.0.0.1:8080/v1`, ollama: `http://127.0.0.1:11434/v1`, etc)
- Python 3.10+
- Python requirements:
  - openai>=1.40.0
  - pyyaml>=6.0
  - pytest>=8.0
  - httpx>=0.27.0
- The model chat template **must support tools**. (For example Phi-4 does not support them correctly, while phi-4-mini does)



## Setup

1. Download a server that hosts models (GGUF or other) and exposes OpenAI-compatible `POST /v1/chat/completions`. llama.cpp and ollama are currently supported. LM Studio and vLLM will also work when `base_url` and the API key match between the running model-hosting server and the config file.

2.1. For llama.cpp copy `llama.bat.example` to `llama.bat` (or `llama.sh.example` to `llama.sh`) and edit that copy: fill in the path to `llama-server`, the models directory `--models-dir`, `--ctx-size`, `--api-key`. In the yaml config set llama.bat. Download models, e.g. from https://huggingface.co/models, look for a gguf version.

```text
copy llama.bat.example llama.bat
```

For example:

```text
"<llama-server.exe>" --models-dir "<weights_dir>" --models-max 1 --ctx-size 16384 --parallel 1 --threads 8 -lv 3 --jinja --api-key "<api_key>"
```

Or for ollama

2.2. For ollama copy `ollama.bat.example` to `ollama.bat` (or `ollama.sh.example` to `ollama.sh`) and edit that copy: fill in the path to the models directory. In yaml set `launcher: ../ollama.bat` (or `../ollama.sh`). Models are pulled separately (`ollama pull …`) and you put their names into the yaml config.

The yaml has a `launcher` field for which kind of server will be started and where to read the API key and context size from.

3. Python 3.10+ and install the Python dependencies. From the `bench/` directory:

```text
python -m pip install -r requirements.txt
```

4. While in the `bench/` directory, start the server (`llama.bat`/`ollama.bat`) and leave it running. Glance at the early server logs. From there you can take the model names the server found and put the ones you want to use into the config file (step 5). Check whether it found them at all, and fix errors if any show up.

llama-server starts a local chat Web UI — you can check that it works by opening `http://127.0.0.1:8080/` in a browser.

Regardless of which server you use, something like this should work (up to the port — the default port differs between llama, Ollama, and others):

```text
curl -H "Authorization: Bearer <api_key>" http://127.0.0.1:8080/v1/models
```

5. Copy [`bench/config.yaml.example`](bench/config.yaml.example) to `config.yaml` and edit it:
- `base_url` — e.g. `http://127.0.0.1:8080/v1`
- `models[0].name` — **exact router model name**
- `models[0].recommended_temperature` — used only when a sampler profile has `temperature: null`. Omit the key or set `null` to leave `temperature` out of the chat request (server/model default). A number is sent as `temperature`. The bundled `greedy` / `agentic` / `creative` profiles set temperature themselves, so this key is optional.
- `prompt_variants` — which variants to use when they are not given on the bench command line; see below.

6. From a command line in `bench/`, run:

```text
python run.py run
```

7. After it starts, the bench creates `bench/results/20260906T100000Z` with logs and results from the run.
While it runs it also prints basic progress, so you can see whether it is working at all, or whether it is already worth stopping and fixing the settings.


## Results

Here is the link to the report presenting the results of the sample models set: [Models Results](https://diovoib.github.io/llms-benchmark/models-result-report.en.html).


## Functionality details

### How does it all work?

After start-up there is a basic check that the server and the model fit together — preflight. The results of that check show whether it is even worth running the tests, because some model files can be damaged, have broken harness tool-calling templates, or the server may not support some OpenAI API option combinations — and then it is worth knowing that, so you do not judge the model through the limitations of the server itself.

This check is whether the server injects `tools` into the prompt (`prompt_tokens` with tools vs without). No difference — those cases are skipped (template/server defect). The result goes into `preflight.json`.

A single test is one task from the catalog, run once in an environment: one model, one sampler profile (temperature, seed…), one prompt variant (neutral, helpful…) and one repeat index. The bench builds the conversation — an optional `system` message from the chosen variant, then the task as `user` — and attaches the catalog of tools the model may call. Or sometimes an empty list. That payload goes out as one `POST /v1/chat/completions` to the model-hosting server, with the parameters above.

If the model returns `tool_calls`, the harness does not call real weather or calendar APIs. It runs a local mock and sends the result back as a `role: tool` message. In the `tools` suite most cases stop after that one turn: the model either calls a tool, refuses, or writes ordinary text. Some tool cases and the whole `agent` suite repeat the loop — model, mock, model again — until the assistant stops calling tools or hits the step limit. The `coding` suite does not use tools: there is one long user message and one (also long) assistant reply.

When the conversation ends, the bench assigns a hard 0/1 (plus any errors it hit). It checks mechanically what can be checked: whether arguments are valid JSON, whether the tool name is in the catalog, whether types and `enum` values match, whether a tool-call template leaked into the user-facing text, whether the finale contains the unique mock token when the case requires it. The transcript and score land in `trial_NNN.json` / `trial_NNN.txt`, and the console prints ok or fail.

Then the bench moves on to the next task.

When the run finishes, the bench writes the results.

After the bench has finished and saved the results, there is a review step — the judge’s job — evaluating what the model actually did and, on that basis, correcting the numbers that came from the hard test assertions. The point is a logical and linguistic check of the behavior and the outputs returned by the model under test, on a given task, with the given working parameters. Several repeats also let you judge how repeatable the model’s behavior is.


### Default run parameters

The default parameters are good for checking whether the setup works at all. The starting command is below and uses the following defaults:

```text
python run.py run
```

config: `--config config.yaml`

suite: `--suites tools`

profile: `--profiles greedy`

Full tests are worth running only after every model listed in the config file has been checked with these settings. `--verbose` prints each scored HTTP call as the chat the model sees (roles, text, tools as YAML, without sampler fields), then streams generated text and tool-call tokens as they arrive, then prints tool calls as YAML after that HTTP turn. Not used on preflight or `summarize`.

A command to copy and then delete list values or flags you do not need, once the basic run works. Details of the options are in the sections below.

```text
python run.py run --verbose --config my.config.yaml --profiles greedy,agentic,creative --suites tools,agent,coding --prompt-variants neutral,helpful,instructed,harness --out results
```

- `harness` — use only when the yaml config has the system prompt filled in, the one sent by the agent you actually use (e.g. Hermes).


### Sampler profiles

| Profile | Meaning |
| --- | --- |
| `greedy` | `temperature: 0`, fixed seed. Sanity check and backend non-determinism detector. |
| `agentic` | `temperature: 0.3`, `min_p: 0.10`, `top_k: 0`, `top_p: 1.0`. Daily local-agent sampling for small models. |
| `creative` | `temperature: 0.5`, `min_p: 0.15`, `top_k: 0`, `top_p: 1.0`. Prose / documentation; still Min-P, not the old stacked Top-K stack. |

For each profile the config defines how many times a given test is repeated. Repeats matter because models are non-deterministic in principle, even with `temperature: 0`.

All sampler fields (`top_p`, `top_k`, `min_p`, `repeat_penalty`, thinking / `chat_template_kwargs`) are required and are copied into the results manifest. Do not change the defaults at the start.


### Prompt variants

Do not mix them in one headline number. Path: `<model>/<profile>/<prompt_variant>/`.


| Variant | System prompt | What it measures |
| --- | --- | --- |
| `neutral` | *(none — no `system` message)* | Natural model behavior. |
| `helpful` | `SYS_NEUTRAL` | The simplest system prompt — `You are a helpful assistant` :) |
| `instructed` | `SYS_INSTRUCTED` | How far model behavior can be steered with a carefully prepared prompt |
| `harness` | `harness_system` in config | Here you can paste your own real agent prompt. |

For the `instructed` variant, the wording of the system prompt matters a great deal. Small changes and ambiguities strongly affect model behavior. Causes of errors and of mismatch with the behavior users expect most often sit right here. A system prompt does not, unfortunately, fix every failure.

`delta_instructed_minus_neutral` (per dimension and per case): positive means the rubric repaired a failure. Tool-call leakage into the user-facing text on `neutral`, clean on `instructed`, means the model is usable with a good system prompt. Leakage on both means that particular behavior will be hard to get out of the model.

Later tests and turns within tests send the model **the task and the data**, never the scoring mechanic.

Default config: `neutral`, `helpful`, and `instructed`. `harness` can be used only after `harness_system` is filled in in the config.

```text
python run.py run --prompt-variants neutral
python run.py run --prompt-variants helpful
python run.py run --prompt-variants neutral,helpful,instructed
```


### Suites

The `--suites` flag selects which kinds of tests the model gets. Each kind has a repeat count per suite and profile in the config.

| Suite | Contents |
| --- | --- |
| `tools` | Single tests, a question, a behavior check: T01, T02… |
| `agent` | Tasks, a behavior check, several turns: A01, A02… |
| `coding` | A programming task; the final result is rather for _manual_ judgment: C01 |


#### Tools (T01–T18)

This is the most convenient suite for a first run: short tests, usually one exchange with the model. It gets a user question and a list of tools it may use. The tools are fake — the bench answers the calls itself — so nothing goes out to real weather, or mail, or similar. From the reply you can then see whether the model picked the right function and whether in the finale it repeated the result, or made it up.

We look for readable behavior. The model should call a tool only when it is needed, take arguments from what the user actually said, not guess missing facts, and not paste a raw tool call into the chat. Some tasks are in Polish (when a city is named, it is Wrocław), some in English (London). Tests with no Polish counterpart have no `_pl` / `_en` suffix.

You do not need to know the numbers at the start — they show up in the results. Briefly, what they are about:

| ID | What happens |
| --- | --- |
| T01 | The user names a city and asks for the weather. The model should call the tool with that same name. |
| T02 | There are eight tools on the table. Pick the one that matches the question. |
| T03 | A general-knowledge question. Tools must not be called. |
| T04 | A weather request, but no city appears in the text. Filling in any city is guessing. |
| T05 | Arguments must have the right types: a number as a number, yes/no as a bool, a value from a list. |
| T06 | More nested arguments (object, list, hour). Nine in the morning is hour 9, not 21. |
| T07 | Two independent requests in one utterance — two calls, order does not matter. |
| T08 | This time tools must not be used. An ordinary text reply is expected. |
| T09 | The model must not invent a tool name it was not given. |
| T10 | Copy an identifier with Polish characters faithfully, without “fixing” it. |
| T11 | A tool with no arguments — an empty call, do not add anything extra. |
| T12 | The user gives a value outside the allowed list. The model should not push it through as valid. |
| T13 | District is optional and the user did not give one. Do not add it on your own. |
| T14 | Two similar tools: weather now versus the forecast. Pick the one that was asked for. |
| T15 | Do not make the same call twice in a row without a reason. |
| T16 | Same as T01, except the conversation is already long. The model still has to hit the city. |
| T17 | First create an event, then send mail. Mail first is a failure. |
| T18 | Two turns: first search for a place, then ask for details using the identifier from the result. |


#### Agent (A01–A07)

Here the task usually does not end on a single call. The model may call a tool, get a result, and continue until it answers the user or hits the step limit. This is closer to what an agent loop actually looks like.

We check whether after reading a result it can stop, whether it weaves that result into the answer, whether after a tool error it does not pretend that “it worked”, and whether it does not take irreversible actions (deleting a file, wiping a disk) without a clear confirmation. There are Polish and English versions; A05 and A07 are English-only.

| ID | What happens |
| --- | --- |
| A01 | The final answer must contain the concrete value the tool just returned. |
| A02 | After reading the temperature, stop. An unused tool from the catalog must not spin the loop. |
| A03 | The tool returns an error. The model should fix the call or explain the problem, not fake success. |
| A04 | A request to delete a file with no confirmation — do not call the tool. |
| A05 | A large catalog, a dozen or so tools. Do not get lost in the list. |
| A06 | The user’s second utterance uses what the model already read earlier. |
| A07 | Conflicting instructions (backup versus wiping the disk). The model should stop and ask, not resolve the conflict on its own. |


#### Coding

A programming task. The model gets no tools: in a single reply it should write a small program — a client for a tool-using chat, following the spec in [`bench/coding/`](bench/coding/) — and in that same reply review whether the code meets the requirements. At the end it should leave a clear verdict: it worked, or it did not.

The bench does not run that program. It saves the whole conversation and the extracted code snippets, and only checks whether it even looks like valid Python. Whether “it actually works and the review was not a self-congratulation” you leave to yourself or to a larger model. This test takes a long time; run it once the shorter suites already work.


### Optional temperature sweep

Config also has `optional_temp_sweep.enabled: true`. It is meant only for the suite named in `optional_temp_sweep.suite`, by default `tools` and `agents`, with a few extra temperatures. Results go into `temp_sweep.json`. Results from this option are not mixed with the main-option results.


### Results

After a run the bench creates a dated folder, for example `bench/results/20260906T100000Z`. That is where everything needed for scoring lands: conversation transcripts, unsparsed HTTP in `*.raw.txt`, layered `summary.json` files, and a copy of the config the run was started with. Required suite timeouts are `tools: 60s`, `agent: 90s`, `coding: 2400s` wall-clock per HTTP call. If that deadline expires while the stream still produced content within `min(request_timeout_s / 2, 5s)` of the cutoff, the trial is `CASE_GENERATION_TIMEOUT` (counted like max_tokens). A silent stream, a non-stream deadline, or a failed TCP connect within `connect_timeout_s` (2s) is `INFRA_ERROR`. Tools/agent `max_tokens` is 1024; C01 uses launcher `ctx_size` (default 16384).

To rebuild summaries for a packed set of model trees (without re-running models):

```text
mkdir results\zestaw-do-sedziego
xcopy /E /I results\20260908T045257Z\gemma-4-12b-it-Q4_K_M results\zestaw-do-sedziego\gemma-4-12b-it-Q4_K_M
xcopy /E /I results\20260909T065153Z\mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M results\zestaw-do-sedziego\mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M
python run.py summarize results\zestaw-do-sedziego
```

As a starting point there are two places: `summary.txt` says what passed and what did not. The `cases/` directories hold the actual conversations — the `.txt` file reads like a chat. When something fails, the transcript shows whether the model called a tool, guessed a city, or pasted template junk into the answer.

The rest of the files matter later, when comparing models or checking whether the server was at fault rather than the model. The `coding` suite saves the whole reply separately, together with the extracted code snippets.

Full results structure:

```text
results/<timestamp>/
  MANIFEST.json
  config.snapshot.yaml
  summary.json
  summary.txt                # short recap: what passed, what did not
  trial_id_map.json
  INTERRUPTED.txt            # only if the run was interrupted
  <model>/<profile>/
    preflight.json
    ABORTED.txt              # when preflight did not pass
    coding/trial_001/        # only if coding ran
      conversation.json
      conversation.txt
      conversation.raw.txt
      attempts/
      tool_client.py
      python_checks.json
      meta.json
      trial.json
    <prompt_variant>/
      SYSTEM.txt
      temp_sweep.json        # only when the sweep is enabled
      cases/<id>/CASE.md
      cases/<id>/trial_001.{txt,json,raw.txt}
```


## Judge

Automatic 0/1 does not finish the evaluation. Give the judge the results folder (it has its own `README.md`, `judge/`, `CASE.md`, transcripts). Do not expect the judge to open this git repo. Mechanical ground truth is the trial files plus the prompt-variant `summary.json`; greedy/agentic/creative, model, and root summaries are means of those child headlines.


### Interpretation

It is easy to confuse a server defect with a model defect. If the model both fails to call a tool when it should, and calls one when it must not, check the chat template and the `--jinja` flag first. A bad template ruins the whole run — then there is no point rejecting the model.

Compare two GGUFs as two separate `models` entries in the config, on the same server and the same suites.

The `greedy` profile (temperature 0) is there to see whether things work at all, and whether the backend still rolls dice with randomness “turned off”. For a decision on whether the model is fit to be an agent, look at `agentic` and at *which* kinds of errors are common, not at one average.


## Project contents

| Path | Role |
| --- | --- |
| [`llama.bat.example`](llama.bat.example) / [`llama.sh.example`](llama.sh.example) | Template for the local llama-server launcher (`--models-dir`, `--ctx-size`, `--jinja` — required for `tools`). Copy to `llama.bat` or `llama.sh` and edit; those files are gitignored. |
| [`ollama.bat.example`](ollama.bat.example) / [`ollama.sh.example`](ollama.sh.example) | Template for the local `ollama serve` launcher (`OLLAMA_HOST`, `OLLAMA_CONTEXT_LENGTH`, `OLLAMA_API_KEY`). Copy to `ollama.bat` or `ollama.sh` and edit; those files are gitignored. |
| [`bench/`](bench/) | The whole benchmark: CLI, cases, mocks, judge (files, no API call). |
| [`bench/config.yaml.example`](bench/config.yaml.example) | Endpoint, models, sampler profiles, suites and repeat counts. To be copied to `config.yaml` |
| [`bench/src/bench/`](bench/src/bench/) | Code: client, preflight, runner, hard 0/1, coding loop. |
| [`bench/src/bench/suites/cases.py`](bench/src/bench/suites/cases.py) | T01–T18, A01–A07 definitions. |
| [`bench/coding/`](bench/coding/) | C01 spec and API contract. |
| [`bench/judge/`](bench/judge/) | Judge prompt, criteria, violation codes, `schema.json`. |
| [`bench/results/`](bench/results/) | Run outputs (gitignored). |


### Out of the project

RAG, GUI, voice, web research, safety red-team.
