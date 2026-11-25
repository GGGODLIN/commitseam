# Building Blocks Tested (`openminion==0.1.24`)

All observations below were verified first-hand against pinned **`openminion==0.1.24`** on Python `3.12.16` in both offline deterministic loopback runs (`pytest -v`) and live Groq runs (`openai/gpt-oss-20b`, recorded in [`docs/live-sdk-receipts.json`](/docs/live-sdk-receipts.json)).

> **Fault Attribution Boundary**: CommitSeam deliberately injects two application-level crashes (`pre-commit` and `post-commit` inside `QueueLedger.enqueue`) to demonstrate recovery invariants. Every finding below distinguishes **genuine `openminion==0.1.24` SDK behavior** from **harness-injected faults**.

## Component Matrix

| OpenMinion 0.1.24 Building Block | How CommitSeam Uses It | Observed Status | Key Finding (Verified First-Hand) |
| --- | --- | --- | --- |
| `APIRuntime.from_config(...)` | Programmatic runtime bootstrap with isolated `om_home` / `om_data` / `openminion.sqlite` | Works with caveats | Requires keyword-only `config`, `home_root: Path`, `data_root: Path` (no defaults for `home_root`/`data_root`). Auto-registers 40+ built-in tool families on startup. |
| `AgentProfileConfig` | Declares `coord` and `triage` agent profiles | Rough edge | `default_channel` defaults to `""`, causing `RuntimeError: inbound authenticity denied (reason=invalid_channel, mode=warn)` unless `default_channel="console"` is set (matches upstream `#864`). |
| `@tool(min_scope=..., dangerous=..., idempotent=...)` | Registers `inspect_failure` (`READ_ONLY`) and `enqueue_rerun` (`WRITE_SAFE`, `dangerous=True`, `idempotent=False`) | Rough edge | Only accepts `{"READ_ONLY", "WRITE_SAFE", "POWER_USER", "UI_AUTOMATION"}` (`registry_toolspec.py:462`). Invalid scope strings (e.g. `"WRITE"`) are NOT validated at decoration time and fail late at runtime with `KeyError`. |
| `Agent.run(approval_callback=decide)` | Binds human approval to `(request_id, item_id, queue_path)` under `ActionPolicyConfig(mode="confirm")` | Works (caveat with `output_type`) | Cleanly blocks tool execution when `decide(...)` returns `False` (`0` rows written, `tool_calls=0`). However, when `Agent(..., output_type=BaseModel)` is set, denial returns plain text `'coord: Confirmation denied. The action was not run.'` and crashes in `_coerce_output` with `AgentOutputValidationError`. |
| `Agent(..., output_type=BaseModel)` | Tested in Gate 01 for typed structured turn outputs | Rough edge | `_build_payload` (`api/agent.py:158-194`) never injects `output_type.model_json_schema()` into system prompts or `response_format`. Replies are also prefixed with `f"{agent_id}: "`, forcing `_extract_json_object` fallback on every turn. |
| `Handoff` | Exposes read-only `triage_failure` peer to coordinator `Agent` | Security caveat | `api/handoff.py:162` calls `target.run(message)` without forwarding the parent `approval_callback` (upstream `#774`/`#776`). CommitSeam enforces strictly `READ_ONLY` tools on `Handoff` peers and keeps all write tools on the coordinator. |
| `session_id` + turn leases | Retry attempts open a fresh `session_id` (`-s1` → `-s2`) under the same business `request_id` | Design boundary | Gateway dedupe (`services/gateway/service.py:428-501`) and `RuntimeEngine` tool cache (`services/runtime/engine.py:250-268`) record completion only AFTER tool return; immediate retry on a crashed `session_id` risks `SESSION_TURN_BUSY` (`turn_leases.py:89-161`). |
| `cancel_event` | Aborts the OpenMinion turn immediately on injected seam faults | Works cleanly | Passing `threading.Event` as `cancel_event` and setting it inside the fault seam terminates the turn cleanly (`RuntimeError: Turn cancelled.`) without wasting a post-crash LLM synthesis call. |
| `GroqProviderConfig` | Live real-LLM execution (`openai/gpt-oss-20b`) | Rough edge | Default model `llama-3.3-70b-versatile` returns HTTP 404 on Groq. `tool_call_strategy="hybrid"` can complete with `tool_calls=0` even when `forced_tools` is set; `tool_call_strategy="native"` invokes tools and fires `approval_callback` reliably. |
| `ProviderError` (`RATE_LIMITED`) | Organic post-tool failure observed live on Groq (`8K` TPM free tier) | Validated thesis live | Because `openminion==0.1.24` issues multiple internal LLM calls per turn (`~2.4K–7.7K` tokens), Groq organically raised `ProviderError: RATE_LIMITED` *after* `enqueue_rerun` committed to SQLite and *before* `Agent.run` returned — proving CommitSeam's SQLite ledger is essential even without injected faults! |
| `MemoryBundle.to_json()` | One-way read-only redacted audit artifact per scenario | Works cleanly | Round-trips cleanly to JSON; CommitSeam wraps items/metadata with recursive key+value secret and host-path redaction and never uses `MemoryBundle` as recovery authority. |
| `APIRuntime.close()` | Called in `finally` block after every run | Caveat | `with Agent(runtime=...)` does NOT close caller-owned `APIRuntime` (`api/agent.py:425`). Callers must always close `runtime` explicitly in `finally`. |
| `context_traces` | Read-only inspection in `commitseam diagnose` | Observed gap | `APIRuntime` exposes no public `context_traces` query method (`list_context_traces` lives under internal modules). In our live 5-session run, all 5 sessions existed in `openminion.sqlite` (`sessions_found`), yet `0` `context_manifest_created` events were recorded under default `APIRuntime.from_config`. |

## Reproducible Verification Commands

Every finding above can be reproduced directly in this repository:

```bash
# 1. Run full offline suite (16 tests + 500 Hypothesis P1..P5 cases + real OpenMinion 0.1.24 loopback)
.venv/bin/pytest -v

# 2. Reproduce @tool(min_scope="WRITE") late KeyError vs valid "WRITE_SAFE"
.venv/bin/python -c '
from openminion import tool
@tool(min_scope="WRITE")
def bad_scope(x: str) -> str:
    return x
print("Decoration succeeded silently despite invalid min_scope=WRITE")
'

# 3. Reproduce APIRuntime.from_config missing default_channel="console" denial
.venv/bin/python -c '
from openminion.base.config.core import AgentProfileConfig
cfg = AgentProfileConfig(name="coord", provider="openai")
print("AgentProfileConfig.default_channel default =", repr(cfg.default_channel))
'

# 4. Run full offline CLI demo + diagnose reconciliation (0 credentials required)
.venv/bin/commitseam demo --yes --provider offline --run-dir /tmp/cs-offline --request-id demo-1
.venv/bin/commitseam diagnose --run-dir /tmp/cs-offline --request-id demo-1-post --json

# 5. Run live Groq demo + diagnose (requires GROQ_API_KEY)
.venv/bin/commitseam demo --yes --provider groq --model openai/gpt-oss-20b --run-dir /tmp/cs-live --request-id live-1
.venv/bin/commitseam diagnose --run-dir /tmp/cs-live --json
```
