# Native-tools spike — Qwen3-VL + llama-cpp `tools=`

Date: 2026-11-11. Base: master at `b2117f4` (PR #89, hearing merged).
Question from the King's Reach council verdict: can Nomous adopt native
llama-cpp `tools=` against the shipped Qwen3-VL build, or must the
`TOOL_CALL:` text parser stay behind one adapter?

## Verdict: FAIL the structured bar — but the model is not the problem

| Schema (10 prompts each) | Structured `tool_calls` rate | Raw well-formed rate | Mean s/call |
| --- | --- | --- | --- |
| `search_memory` (1 required + 2 optional, incl. enum) | **0/10 (0%)** | **10/10** | 29.3 (first call 202.7 s, steady ~10 s) |
| `get_people_present` (no parameters) | **0/10 (0%)** | **10/10** | 7.5 (first call 18.4 s, steady ~6 s) |
| `record_observation` (2 required + 2 optional, enum + array) | **0/10 (0%)** | **10/10** | 19.8 (first call 40.0 s, steady ~16 s) |
| **Overall** | **0/30 (0%)** | **30/30 (100%)** | ~19 s |

The bar was ≥90% structured valid-call rate on all three schemas.
Structured rate is 0%, so **do not adopt bare
`create_chat_completion(tools=)` as the call path**.

But the failure is 100% on the binding's response-parsing side, not the
model. On every trial the model emitted a well-formed native Qwen tool
call — correct function name, JSON-parseable arguments, required args
present, types and enum values valid. Example (verbatim, trial 1):

```
<tool_call>
{"name": "search_memory", "arguments": {"query": "dog named Biscuit", "modality": "all"}}
</tool_call>
```

## Root cause: the fork never parses `<tool_call>` blocks on the `tools=` path

Fork: JamePeng `llama-cpp-python` 0.4.6, commit `ec56af26`
(`v0.4.6-cu128-linux-20261107` tag). File `llama_cpp/llama_multimodal.py`,
`MTMDChatHandler.__call__` (base class of `Qwen3VLChatHandler`):

- Line 2518: `tool = None`.
- Lines 2504–2525: `tool` is only set when the caller passes the legacy
  single-function `function_call=` parameter (it looks up the matching
  tool by name).
- Lines 2582–2587:
  ```python
  if tool is not None:
      tool_name = tool["function"]["name"]
      return _convert_completion_to_chat_function(tool_name, completion_or_chunks, stream)
  return _convert_completion_to_chat(completion_or_chunks, stream=stream)
  ```
- With `tools=` + `tool_choice="auto"` (the only sane way Nomous would
  call it), `function_call` is `None`, so `tool` stays `None` and the
  response goes through plain `_convert_completion_to_chat`
  (`llama_chat_format.py:593`), which copies the raw text into
  `message.content` and never looks for tool calls. `message.tool_calls`
  is therefore always empty on this path.

Meanwhile the handler's own `Qwen3VLChatHandler.CHAT_FORMAT` template
(`llama_multimodal.py:4680`) instructs the model to reply exactly in the
format it used:

```
For each function call, return a json object with function name and
arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

There is no code anywhere in the fork that parses a
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` block out of a
completion into structured `tool_calls`. The template's history rendering
(`:4722`) does handle `message.tool_calls` when re-rendering a
conversation — but since the response never populates them, multi-turn
tool use cannot work either. `_convert_completion_to_chat_function`
(`llama_chat_format.py:610`) is not a parser: it wraps the *entire*
completion text as the arguments of one pre-named tool, and only runs on
the legacy single-function path.

## Recommendation

Keep a parser behind one adapter — but parse the model's **native**
`<tool_call>` JSON, not the custom `TOOL_CALL:` text format. The model
already emits the native format 30/30, so the adapter's job is purely
mechanical: regex out `<tool_call>...</tool_call>` blocks from
`message.content`, `json.loads` the arguments, validate against the
registry schema (the same definitions `ToolExecutor.get_tools_schema`
already produces), and execute via `ToolExecutor.execute_tool`. That is
the same shape as the existing `parse_tool_calls` (`tools.py:854`) with a
different wire format, and it leaves the door open to true native calls
later if the fork gains a parser (or Nomous patches the fork's
`MTMDChatHandler.__call__` to parse the blocks before converting).

Concretely: pass `tools=` with each registry definition wrapped as
`{"type": "function", "function": definition}` (the template and the
model handle the rest), parse in the adapter, execute, feed results back for the follow-up turn
exactly as `_generate` does today. Do not wait for an upstream fork fix
to do this — the parsing is ~20 lines and fully in Nomous's control.

## Method and environment

- Runner: 3 schemas taken verbatim from `ToolExecutor.get_tools_schema`
  (`tools.py:539`) — `search_memory`, `get_people_present`,
  `record_observation` — in OpenAI tool format, 10 distinct prompts per
  schema, each only answerable via the tool. `tool_choice="auto"`,
  `temperature=0.2`, `max_tokens=384`, thinking ON (faithful to the
  shipped vision path, where `vision.show_thinking: true` turns thinking
  on). Validity = structured `tool_calls` with the expected name and
  schema-conformant JSON arguments.
- Model: `Qwen/Qwen3-VL-8B-Instruct-GGUF`,
  `Qwen3VL-8B-Instruct-Q4_K_M.gguf` (5,027,784,800 bytes). Handler:
  `Qwen3VLChatHandler` with `extra_template_arguments={"enable_thinking":
  True}`. `n_threads=2`, CPU-only, no GPU.
- Deviations from the shipped Nexus configuration, forced by this VM's
  8 GB RAM (see "Side findings"): mmproj is the **Q8_0** build
  (`mmproj-Qwen3VL-8B-Instruct-Q8_0.gguf`, 752,289,728 bytes) instead of
  the shipped F16 (1,159,029,824 bytes), and `n_ctx` is 2048 instead of
  16384. Projector precision cannot affect text-only tool calls (no
  image tokens are involved), and the smaller context only caps the KV
  cache. On Nexus (16 GB VRAM) neither deviation is needed; the rates
  should be re-confirmed there with the F16 mmproj before the adapter
  ships, since the template/model are identical but the environment
  differs.
- A thinking-off control phase was planned if phase 1 scored <90%, and
  was **skipped deliberately**: the failure mode is identical on all 30
  calls and is fully explained by the response-parsing code path above.
  Thinking cannot change which converter the handler calls, so a
  control could not change the outcome. The raw-emission rate (30/30)
  already answers the model-capability question the control was meant to
  separate.
- First call per schema is slow (202.7 s / 18.4 s / 40.0 s; the very
  first call overall is the worst) — a cold-start cost, likely the tool
  definitions being tokenized/cached. Steady state on this 1.5-CPU VM is
  ~6–19 s/call. Budget for one slow first call per process start.
- Raw per-trial data (prompt, raw content, classification, seconds):
  `spike_results.jsonl` / `spike_summary.json` in the spike workspace
  (`~/workspace/nomous-spike/`); runner at
  `~/workspace/nomous-spike/spike_native_tools.py`.

## `register_tool` consumption in `llm.py` — call sites

Verified against master `b2117f4` (26 tools registered in
`tools.py:_register_builtin_tools`, lines 171–518):

| Site | What happens |
| --- | --- |
| `llm.py:260–261` | `self.tools = ToolExecutor(self)`; `tools_enabled` from `config.yaml` (`llm.tools_enabled`, default `True`) |
| `llm.py:1149–1182` | `_build_prompt()` embeds `self.tools.get_tools_prompt()` (`tools.py:555`, `TOOL_CALL:` text format) when tools are enabled |
| `llm.py:1274` (`_generate`) | Text path calls raw `self.model(...)` completion — **not** `create_chat_completion`. Adopting native tools means moving this path to chat completions; it is not a drop-in swap |
| `llm.py:1366–1376` | `_generate` detects `TOOL_CALL:` in the raw response, `self.tools.parse_tool_calls()` (`tools.py:854`), `self.tools.execute_tool()` (`tools.py:600`), then re-generates a follow-up with results (tool calls disabled on the follow-up) |
| `llm.py:1583` (`_run_vision_completion`) | Vision path already uses `create_chat_completion` — but without `tools=` |
| `llm.py:1712, 1790, 2139, 2298, 2381` | `_build_prompt` callers with tools: `chat`, `process_audio`, vision, `_idle_act`/autonomous |
| `tools.py:530 / 539 / 555` | `get_mcp_tools` / `get_tools_schema` / `get_tools_prompt` — all three derive from the same MCP-style definitions, so they cannot disagree about parameters. The per-tool content (`name`, `description`, `parameters`) that `get_tools_schema` returns is already OpenAI-shaped, but each flat definition must be wrapped as `{"type": "function", "function": definition}` before passing it to `tools=` — the spike did this wrapping; it is a one-line envelope, not a schema translation |
| `scripts/run_bridge.py:599, 612` | Dashboard `tools/list` serves `get_mcp_tools()`; `tools/call` executes via `execute_tool` |

## MCP client adapter — placement and estimate

**Yes, it can live in a new file outside `llm.py`.** The registry already
exposes everything the adapter needs (`get_tools_schema`,
`get_mcp_tools`, `execute_tool`), and `llm.py` touches tools at only two
seams: prompt construction (`_build_prompt`, :1182) and the parse/execute
step (`_generate`, :1366–1376). Proposed shape:

- New file, e.g. `src/backend/native_tools.py`: converts the registry
  schema to OpenAI tools (thin wrapper over `get_tools_schema`), parses
  `<tool_call>` JSON blocks out of `message.content`, validates against
  the schema, and returns structured calls for `execute_tool`.
- `llm.py` changes, confined to the two seams: a native-call path in
  `_generate` (build messages, `create_chat_completion(tools=...)`,
  hand the response to the adapter, feed results back for the follow-up
  turn), plus a config flag to select native vs `TOOL_CALL:` text mode
  so the old path stays as a fallback.
- Tests: unit tests for the parser (well-formed, malformed JSON, wrong
  name, missing required arg, bad enum, multiple blocks, no block) plus
  an integration test with a stubbed completion.

**Estimate: ~2 days** for the adapter + native path + tests. The full
MCP *client* (external servers, transport, allowlist) is additional work
on top of the adapter — roughly +2–3 days — and this spike does not block
it: the wire format the client would carry is now proven 30/30, and the
allowlist can be enforced in the adapter before execution.

## Side findings (recorded, not in scope for this PR)

- `scripts/start.py` (~line 503) builds the fork from
  `git+https://github.com/JamePeng/llama-cpp-python.git@v0.4.6`, but **no
  `v0.4.6` ref exists** in the fork (only versioned tags such as
  `v0.4.6-cu128-linux-20261107` and branches such as `0.4.5`). The
  launcher's source-build fallback would fail to resolve. Needs a
  separate fix (pin the real tag or a commit).
- No Linux CPU wheel exists for the fork (releases publish CUDA wheels
  for Linux/Windows and Metal for macOS only), so a source build is
  required. The build fails unless the vendored llama.cpp submodule is
  checked out first (`git submodule update --init --depth 1
  vendor/llama.cpp`) — CMake errors on the missing
  `vendor/llama.cpp/CMakeLists.txt` otherwise.
- F16 mmproj + 8B model together exceed an 8 GB RAM box: the first run
  thrashed (63 GB disk reads in 18 min, first call still unfinished) and
  was killed. Q8_0 mmproj + `n_ctx` 2048 fixed it (model load 29 s,
  steady ~6–19 s/call). On Nexus this is a non-issue, but it is the
  reason for the mmproj/`n_ctx` deviations above.
