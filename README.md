# 🧔 DadBot Sovereign — Your Local AI Dad

A warm, private AI companion with long-term memory that remembers you, grows with you, and feels like having a supportive dad in your corner.

**Local processing is available.** Ollama chat, local memory, local STT, and Piper TTS can run on your machine. Auto voice may use edge-tts or ElevenLabs and send reply text to external services; choose Piper or pyttsx3 for local speech.

Ollama chat clients use a 120-second transport timeout (including waiting for
response headers or the next stream chunk). Separately, submitted turns default
to a 120-second cooperative execution deadline; an explicit per-turn timeout
overrides it. The synchronous turn lock and async thread bridge stop waiting
after 180 seconds. Cancellation cannot forcibly stop a synchronous worker;
the transport timeout still bounds network inactivity. A five-second health
probe may time out during a normal cold model load.

Factual turns such as `2+2` skip model-based mood detection when neither the
message nor a relevant emotional follow-up contains emotional signals. They also skip
tool planning without tool intent. Emotional messages, attachments, tool-name
keywords, and contextual tool follow-ups remain eligible. The same planning
gate applies to a configured Home Assistant MCP service; it never bypasses
approval for an actual action. Answer generation remains on the canonical graph.

Emotional chat turns classify mood and generate the supportive answer in one
validated JSON request (Ollama thinking disabled); only the reply reaches the UI. Mood is committed once
at the normal save boundary. An optional family reference shares that request.
Direct-answer and empathy-first strategies retain critique diagnostics and
safety checks but do not generate a second answer for critique revisions.

Enable INFO logging for `dadbot.managers.runtime_client` to see per-purpose
Ollama durations measured with `time.perf_counter()`, prompt/generation token
counts, and completion reasons without logging prompts or reasoning text.

Tests construct isolated real managers and mock only inference/embedding
boundaries. Retired command-dispatcher and natural-language reminder-splitter
contracts have narrow, strict xfails with explicit reasons; the canonical
`generate_candidate` contract remains tested, not xfailed. A supplied
`DadBotConfig` is preserved through boot, including its tenant and storage paths.

---

## Why Dad-Bot?

- **Actually remembers you**: Learns from your conversations over weeks/months. Tracks your goals, relationships, mood patterns.
- **Feels real**: Adaptive personality, emotional awareness, proactive check-ins. More than a chatbot.
- **Privacy controls**: Ollama chat and local voice backends are available; Auto voice may select a cloud-backed TTS service when installed/configured.
- **Voice support path**: Local STT/Piper options and optional edge-tts/ElevenLabs backends are included; setup maturity depends on your machine tooling.
- **No subscriptions**: Install once, use forever.

---


## ✨ Key Features

- 🧠 **Deep Long-Term Memory** — Remembers conversations, goals, relationships, and emotional context
- 💭 **Relationship Modeling** — Understands trust, emotional momentum, and personal history
- 🎤 **Voice Pipeline (Experimental)** — Local STT/TTS integration hooks are present; full turnkey setup is still maturing
- 📜 **Oral History Capture** — Record or upload audio, review and correct the local transcript, then explicitly save it as an `oral_history` memory. The original audio is not retained.
- 📅 **Calendar & Proactive Flows (Partial)** — Core reminder/check-in plumbing exists; full calendar-sync automation remains in progress
- 📬 **Proactive Flows** — Daily check-in is fully supported; calendar sync and advanced scheduling are in development.
- 🎨 **Avatar Support (Partial)** — Avatar rendering exists with fallback paths; generation workflows vary by environment
- 📱 **Mobile-Friendly** — PWA works on phone browsers
- 🔒 **Local modes available** — Auto voice may use cloud services; select a local backend for on-device speech

The 200-question memory regression test uses an isolated, seeded 50-fact vault
and requires at least 98% top-3 recall. The default test uses mocked embeddings;
to also test real local `nomic-embed-text` embeddings, start Ollama, pull that
model, set `DADBOT_RUN_REAL_EMBEDDING_BENCHMARK=1`, and run
`python -m pytest tests\eval\test_memory_benchmark.py -s`. The questions are
templated and repeat the fact topics; results do not measure production-vault
or source-mode citation quality. The same opt-in run also includes a separate
25-question engineering-authored paraphrase diagnostic against the synthetic
facts. That diagnostic reports misses without pretending it meets the
200-question regression gate.

Semantic candidate search uses the current user query without synthetic
personality/history labels; recent turns remain available to contextual
scoring. This prevents metadata tokens from displacing query terms in the
bounded candidate filter.

General-memory retrieval now uses SQLite FTS5 BM25 for its lexical arm.
FTS5 is required in the Python SQLite build; initialization/query errors
propagate rather than silently reverting to word overlap. A private in-memory
index backfills the current normalized tenant catalog on first use, rebuilds
when its keys/text change, and closes at shutdown. No embedding job or saved
lexical database is needed, including when semantic storage uses Postgres.
There is no source-citation approval filtering on this conversational path;
source mode remains a separate, approved-evidence-only retrieval contract.

BM25's negative ranks are converted to positive scores and divided by the
query's maximum match score. The convex blend retains a 0.75 weight on the
existing contextual dense score (semantic relevance, graph signal, importance,
and recency) and 0.25 on BM25. The old token-overlap fallback is removed.
Set `DADBOT_MEASURE_FUSION=1` alongside the real-embedding benchmark flag
to sweep weights and report 160 training / 40 held-out questions, split by
whole fact topic. Real Nomic measured 200/200 top-1 and top-3 for all sampled
dense weights from 0.0 through 0.9, including the existing 0.75; dense-only
measured 191/200 top-1 and 192/200 top-3. BM25-only also passed every question.
This plateau does not identify a unique optimal weight: the templated
questions favor lexical matching.

The 25 natural paraphrases initially measured 19/25 top-1 and top-3.
Diagnostic logging identified these pre-cosine exclusions in the SQLite path:

| Target | Query example | Exclusion |
| --- | --- | --- |
| Favorite color | Which shade would I pick? | No selected token substring matched |
| Preferred season | Which time of year do I enjoy most? | No selected token substring matched |
| Wake-up time | When do I get out of bed on workdays? | Neither tokens nor inferred `work` category matched the routines entry |
| Vacation destination | Where am I planning to go on holiday? | Neither tokens nor inferred `goals` category matched the travel entry |

The query's neutral mood and general category were not active filters.
The 108-row candidate limit did not cause these four exclusions in the
50-fact vault, but could exclude older entries in larger catalogs.

Conversational semantic search now fetches every indexed key in the supplied
current catalog before exact cosine scoring, in both SQLite and pgvector.
Only catalog membership gates this fetch: token, inferred category, mood,
vector distance, and row-limit predicates do not. Missing indexed keys emit
a coverage warning, rather than silently masquerading as relevance misses.
Alignment/freshness and relevance remain scoring decisions; the output limit
applies after scoring. The hybrid caller also retains the entire semantic
result set until BM25 fusion, avoiding a second, lossy intermediate top-k.

After both eligibility fixes, real Nomic measured 200/200 top-1 and top-3 on
templates and 20/25 top-1, 22/25 top-3 on paraphrases at unchanged weight 0.75.
All four previously excluded targets reach cosine scoring. Three remaining
top-3 misses (season, bedtime routine, commute) have positive cosine and
semantic scores but zero BM25, so these are now final-ranking cases, not
pre-cosine exclusions. The small paraphrase split remains 20 training / 5
held-out questions. At 0.75 it measures 17/20 training top-1, 19/20 top-3 and
3/5 on both held-out metrics. Training alone selects dense-only in the latest
sweep; that does not establish a safe replacement for the existing blend,
which continues to pass the separate 200-question gate. The sweep verifies
that replay at the current weight reproduces actual retrieval counts.

The catalog-key fetch does not use vector ordering and cannot use HNSW for
nearest-neighbor selection, even if an ANN index already exists on Postgres.
The local configuration requests pgvector with no ANN index or fixed
dimensions; live existing server indexes remain unverified because the
inventory connection timed out. No separate ANN replacement is needed for
this conversational path. Other backend candidate APIs retain their existing
filter semantics. Source mode remains separate and already scores every
currently eligible approved source without query prefilters. Tests verify
1,100 catalog keys across SQLite batches, catalog exclusion of unrelated rows,
pgvector key-only SQL under all ANN configurations, facade scoring coverage,
and scoring eligibility of 120 approved sources while excluding unapproved
evidence. These are synthetic scoping checks, not a production-vault audit.

Source mode keeps a bounded, in-memory LRU cache of 64 exact normalized queries
per service, scoped by tenant. Hits become most recently used; capacity
evicts the least recently used answer. Each lookup reads current approved
evidence and the full vault projection. Any canonical vault mutation or
save attempt (even unchanged content), projection-content change, or approval
revision invalidates all cached replies. This deliberately coarse policy
also covers unrelated/unapproved memories, edits, deletion, and provenance
changes, trading hit rate for conservative invalidation. Final citation
validation bypasses the cache and recomputes the answer from current sources.
The cache does not survive restart, persist transcripts separately, reuse
semantically similar queries, or skip safety/approval validation. It avoids
repeat ranking/formatting, not evidence reads; no end-to-end speedup is claimed.

Reviewed oral histories save transcript evidence with a unique ID, UTC review
time, and SHA-256 content hash alongside the memory; the transcript is stored
both as memory text and as its evidence. Source-mode citation approval is separate.
Missing or mismatched evidence disables citation, including for older oral-history
memories saved before evidence was recorded; review and save those again to cite
them. Deleting the memory deletes its attached evidence. A hash checks consistency,
not the identity of the speaker, the truth of the account, or protection against tampering.
Existing backups and other historical snapshots may still contain deleted evidence;
this is not a secure-erasure guarantee.

### Spatial Agent Live Avatar Integration

Dad-Bot can embed a live avatar frontend (for example a fork of josepeon/spatial-agent) directly in the chat presence stage.

Set in .env:

```bash
DADBOT_SPATIAL_AVATAR_URL=http://localhost:8080/index.html?dadbot_dev_no_auth=1
DADBOT_SPATIAL_RELAY_WS_URL=ws://127.0.0.1:8787/v1/spatial/ws
# Local dev convenience: disable API auth for direct browser websocket testing
DADBOT_API_AUTH_REQUIRED=0
```

Recommended local wiring on Windows:

```powershell
# Terminal 1: run Dad-Bot API (includes Spatial relay websocket at /v1/spatial/ws)
cd C:\Users\josep\OneDrive\Desktop\Dad-Bot
.\.venv\Scripts\python.exe Dad.py --api --api-host 127.0.0.1 --api-port 8787

# Terminal 2: run your Spatial Agent backend
cd C:\path\to\spatial-agent
.\.venv\Scripts\python.exe -m uvicorn backend:app --reload --host 0.0.0.0 --port 8001

# Terminal 3: serve the Spatial Agent frontend over HTTP
cd C:\path\to\spatial-agent\frontend
python -m http.server 8080

# Terminal 4: run Dad-Bot UI with embed + relay URLs
cd C:\Users\josep\OneDrive\Desktop\Dad-Bot
$env:DADBOT_SPATIAL_AVATAR_URL = "http://localhost:8080/index.html?dadbot_dev_no_auth=1"
$env:DADBOT_SPATIAL_RELAY_WS_URL = "ws://127.0.0.1:8787/v1/spatial/ws"
.\.venv\Scripts\python.exe -m streamlit run dad_streamlit.py --server.port 8501
```

VS Code tasks are also available:
- `Run Dad Bot API (Spatial Relay)`
- `Run Spatial Live Mock Frontend`
- `Run Dad Bot UI (Spatial Embed)`
- `Run Dad Bot Spatial Stack` (launches mock frontend + UI + API together)

Local live mock avatar client:
- A websocket-enabled demo avatar is included at `static/spatial_live_mock/index.html`.
- It reads the relay URL from `dadbot_ws`/`dadbotWs` query parameters, sends text turns, and plays returned base64 WAV audio.
- For authenticated use, omit `dadbot_dev_no_auth=1`, configure `DADBOT_REDIS_URL`, add the frontend origin to `DADBOT_API_CORS_ORIGINS` (for example `http://localhost:8080`), and enter credentials in the avatar's **API authentication** fields. Credentials remain in the page's memory, not URLs or browser storage.

### WebSocket handshake authentication

Browser clients must exchange HTTP credentials for an opaque ticket before each WebSocket connection or reconnect:

1. `POST /v1/websocket/ticket` with normal `Authorization: Bearer ...` (or `X-Dadbot-Token`) and, if enabled, `X-DADBOT-KEY` headers.
2. Send JSON `{"channel":"pulse","session_id":"your-session","tenant_id":"your-tenant"}`. Channels are `events`, `pulse` and `spatial`; `spatial` does not need a session ID and requires write scope. Session streams require read scope.
3. Use the returned ticket as `?ticket=...` on the matching WebSocket endpoint. A supplied `tenant_id` must match the authenticated tenant.

Tickets last **at most 60 seconds**, never outlive their original principal, and work exactly once. Redis 6.2+ provides atomic `GETDEL` consumption across API processes; configure `DADBOT_REDIS_URL` for ticket issuance. Missing/unavailable Redis fails closed (HTTP 503 or WebSocket close 1013); it never falls back to process-local ticket storage. The production Compose Redis 7 service supports this.

Long-lived `access_token`, `dadbot_key` and `x_dadbot_key` query parameters are rejected on all three WebSocket endpoints. Non-browser clients may continue authenticating with headers. The Python service client's `authenticated_session_event_stream_url(...)` performs the HTTP exchange and returns a fresh, versioned ticket URL; request another URL after reconnecting. Ordinary stream URL helpers remain side-effect-free descriptors so offline dashboard rendering does not contact the API.

Use HTTPS/WSS outside loopback and redact WebSocket query strings in proxy/access logs, even though tickets are short-lived and single-use. `dadbot_dev_no_auth=1` is an explicit demo-only bypass of the exchange, usable only when the server itself has authentication disabled and no secret-key requirement; it does not bypass server authentication.

Notes:
- The Dad-Bot chat stack stays in Dad-Bot; the avatar panel embeds the external frontend.
- If the URL is empty or unavailable, Dad-Bot falls back to local avatar media files.
- The iframe URL gets `dadbot_ws` and `dadbotWs` query params automatically so avatar clients can discover the relay endpoint.
- Ensure the embedded frontend allows being framed (no restrictive X-Frame-Options/CSP frame-ancestors).

---

## 🏠 Smart Home Capabilities

Home actions use an external Home Assistant MCP server. The old in-memory device/scene classes remain only as dormant compatibility scaffolding; chat no longer routes actions through them. Dad-Bot does not discover devices itself or maintain a parallel device registry. The MCP connection is disabled until an endpoint and explicit tool allowlist are configured.

Install the MCP extra:

```bash
pip install -e .[mcp]
```

Configure the MCP server endpoint and allow only the tool names you intend Dad-Bot to call. The optional token is sent only as an HTTP bearer header and is never added to model prompts:

```powershell
$env:DADBOT_HA_MCP_URL = "https://your-home-assistant-mcp-endpoint"
$env:DADBOT_HA_MCP_TOKEN = "your-mcp-bearer-token"
$env:DADBOT_HA_MCP_ALLOWED_TOOLS = "tool_name_one,tool_name_two"
```

The endpoint must support MCP Streamable HTTP. Use the tool names returned by that server in `DADBOT_HA_MCP_ALLOWED_TOOLS`. The Smart Home page can verify the connection and report which allowlisted tools are available. No tool is callable from chat unless it is both offered by the MCP server and included in this allowlist.

The client uses the MCP 2 SDK stream pair and snake-case schema/result fields.
Planning is constrained by each live tool's argument schema, including required
fields. An explicit named-tool request cannot be reported as completed without
a tool call and confirmed result. Ollama planning uses non-thinking structured
generation; validation failures remain explicit refusals, never device success.

Every proposed MCP call pauses for explicit approval in the Chat page. The UI shows the exact tool name and validated arguments; each call in a multi-call plan requires a separate approval. Before an approved call is sent, Dad-Bot rechecks the live server's allowlist and input schema. Rejecting a call sends no request, and new chat input is held while an approval is pending in that conversation.

The MCP loop returns the model's validated final answer, prompted in the Dad persona, rather than raw tool payloads or unparsed plans. If model finalization fails after tool execution, it reports that failure and warns that device actions may already have occurred; it does not claim success or fall back to a "no action performed" reply. This protection does not depend on Streamlit reply normalization.

---

## ⚠️ Experimental Features & Warnings

- **Replay and deterministic recovery infrastructure exists but is experimental and disabled in the default product path; it is not currently a certified user-facing capability.**

---

## 🚀 Quick Start

### Install and Run

```bash
git clone https://github.com/josephgiardello-cloud/Dad-Bot.git
cd Dad-Bot
python install.py
```

If you prefer manual setup, copy `.env.template` to `.env` and adjust values before launch.

This will:
- Install dependencies and set up Python environment
- Prepare profile and memory files
- Check for Ollama installation
- Pull recommended AI models
- Launch Streamlit UI

### Quick Launch (Next Time)

```bash
python launch.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📸 Screenshots & Demo

No sanitized product screenshots are currently included. Capture demo views using a clean test profile and synthetic conversation data; do not commit personal avatars, user conversations, credentials, or device details.

---

## 🎯 First-Run Setup

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.com) installed and running
- Recommended: `ollama pull qwen3.5:9b` and `ollama pull nomic-embed-text`

### Local chat model and streaming

The default chat model is `qwen3.5:9b`. Explicit model selections and generation options are preserved; the runtime does not disable thinking or override the context window. Actual GPU usage and latency depend on the model, prompt, attachments, and other running applications.

Existing profile model choices are not overwritten. If a saved profile selects another model, update its `llm.model` field to `qwen3.5:9b` and restart the app. `DADBOT_LLM_MODEL` can set the default; CLI `--model` and service `DADBOT_DEFAULT_MODEL` overrides remain available.

Chat, Status probes, Workshop probes, and outbox retries call the canonical `process_user_message` entrypoint and validate replies through shared reply finalization. Chat displays, saves, and speaks only the finalized reply; it does not run a separate UI model executor or expose planning JSON as drafts. New chat and Duplicate controls are in the main chat column, and the conversation selector follows the new thread.

The Streamlit runtime boots orchestration services into the bot's existing manager registry, then registers its UI adapters there. The service container, orchestrator, graphs, and persistence aliases share that registry. `ServiceRegistry.boot()` creates a new registry by default; passing `registry=existing` populates and returns that exact instance.

### Environment Setup

1. Copy `.env.template` to `.env`
2. Set at minimum:
	- `OLLAMA_HOST`
	- `DADBOT_LLM_MODEL`
	- `POSTGRES_PASSWORD` (if using postgres-backed flows)
3. Keep `.env` local and uncommitted

The install script will guide you through the rest.

### Local Model Context and Timing

Ollama requests allocate `num_ctx` in power-of-two buckets starting at 512
(512, 1024, 2048, 4096, 8192, and larger when requested), capped by the
model's context capacity. The prompt guard still uses DadBot's configured
`CONTEXT_TOKEN_BUDGET` (default 6000) or the request's explicit budget, capped
by the model's capacity; bucketing does not expand the allowed prompt.
Thus a 6000-token budget allocates an 8192-token context when supported,
using more KV-cache memory than an unbucketed allocation.
A Modelfile's default `num_ctx` is not treated as the architectural capacity.
Before transport, the prompt guard drops older history and trims oversized
messages, preserving the latest task where possible. Thinking-capable models
reserve at least 3072 tokens for generation, capped at half the request window;
other models use `RESERVED_RESPONSE_TOKENS` (minimum 64).

Reasoning-only responses that end with `done_reason=length` raise an explicit
budget-exhaustion error rather than succeeding with an empty answer. This
does not guarantee unlimited reasoning fits; increase the configured context
within available memory when necessary. INFO-level Ollama timing logs include
request purpose, token counts, load/generation durations, and completion reason,
but not prompt, answer, or hidden-reasoning text. Mood detection, tool planning,
and chat can each incur a separate generation.

Chat requests use `keep_alive=-1` by default. Set
`DADBOT_OLLAMA_KEEP_ALIVE_SECONDS=300` before startup to restore a five-minute
idle timeout, or `0` to unload after each request. Residency is subject to
Ollama's memory availability; multiple models can compete for RAM/VRAM.
This setting does not change the embedding model's idle timeout.
Automatic `num_keep`, Flash Attention, and KV quantization are not enabled
by DadBot. Existing explicit `num_keep` options are preserved. No latency
reduction is claimed without measuring actual request timing.

`num_keep` is not an automatic prefix-cache accelerator: Ollama's published
runner implementation uses it when shifting/truncating context; prefix-cache
reuse matches existing input tokens. DadBot's system prompt also includes
turn-specific context. An opt-in synthetic probe compares warm prefill with
`num_keep=0` and `128` without allowing prompt trimming:
set `DADBOT_RUN_REAL_OLLAMA_PREFIX_CACHE=1` and run
`python -m pytest tests\unit\test_ollama_prefix_cache.py -s`.
On the tested local `llama3.2` model both settings had a 10.8 ms median
warm prefill time. This small probe is not a TTFT measurement or a guarantee
for other models, overflow scenarios, or the full companion prompt.

Canonical graph turns already batch emotional classification and the answer
in one structured model response when emotional markers are present. The
result supplies the final mood and natural-language reply; malformed
structured results fail validation. No separate mood-model call is made on
that path. This does not guarantee that tools, maintenance, or other turn
paths make only one model call.

Optional Ollama server tuning is global, separate from DadBot request options:
`OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, and
`OLLAMA_KEEP_ALIVE=-1`. On Windows, set these user environment variables and
restart the Ollama app. Verify `/api/version`, real inference, and server logs
for the effective cache/attention settings. Official Ollama guidance describes
`q8_0` as approximately half the KV memory of `f16`, with a small precision
loss that depends on model/task; it is not a risk-free performance setting.
Restore `OLLAMA_KV_CACHE_TYPE=f16` and restart to disable KV quantization.
Global keep-alive remains overridden by an explicit request `keep_alive`.

### Story Mode (Password Protected)

Story Mode enables deeper personalization learning. Set a password before launch:

```bash
export DADBOT_STORY_MODE_PASSWORD="replace-with-local-secret"  # or set in .env
python launch.py
```

---

## 🧑‍💻 Development & Contribution

### Running Tests

Install the declared development dependencies first (`pip install -e ".[dev]"`).
Tests require `pytest-timeout`: each test has a 120-second default deadline and
stalled-thread diagnostics after 30 seconds. On Windows the thread timeout
terminates the test process; use `pytest -vv --timeout=30` to identify a stalled
case sooner. The mock-backed bot fixtures reject accidental live inference and
disable optional graph LLM extraction; tests must explicitly stub the model port
or transport when needed.

```bash
pytest tests/
```

For regular stability validation, keep to the core lanes:

```bash
pytest -m unit -q
pytest -m integration -q
pytest -m durability -q
```

### Test Lanes Quick Map

- `Test Lane: DEV` (`-m unit`) for fast local iteration
- `Test Lane: INTEGRATION` (`-m integration`) for service/storage boundaries
- `Test Lane: DURABILITY / PHASE4` (`-m durability`) for restart/commit integrity

Anything beyond those lanes should be treated as supplemental, not mandatory.

### Feature Maturity Matrix

| Capability | Maturity | Evidence |
|---|---|---|
| Core local chat + memory pipeline | Available, with UI limitations | `dadbot/core/dadbot.py`, `dadbot/core/turn_mixin.py`, `dadbot/core/graph.py`; the Streamlit sample currently uses no-op turn-event/checkpoint persistence services. Profile and memory storage are separate. |
| Voice stack hooks and optional deps | Experimental | `pyproject.toml` (`[project.optional-dependencies].voice`), `dad_streamlit.py`, `dadbot/agentic.py` |
| Local reminders + local calendar events | Shipped (local storage) | `dadbot/agentic.py` (`add_reminder`, `add_calendar_event`, `list_calendar_events`) |
| Calendar feed synchronization | Partial | `dadbot/core/boot_mixin.py` (`ical_feed_url`, `schedule_calendar_sync`) |
| Proactive heartbeat/nudges | Partial | `dadbot/core/boot_mixin.py` (`_run_proactive_heartbeat_loop`), `dadbot/assistant_runtime.py` |
| Avatar rendering support | Partial | `dad_streamlit.py`, `pyproject.toml` (`[project.optional-dependencies].heritage`) |

The Streamlit runtime is a development/demo composition, not the durable service profile: its turn-event and checkpoint persistence methods are no-ops, and its event bus is in-memory. Do not treat those records as restart-durable.

### Docker (Dev & Production)

```bash
docker compose up
```

See `docker-compose.yml` for profile options.

---

## 📚 Technical Documentation

### Production Spine (Authoritative)

- Runtime spine contract: `docs/PRODUCTION_SPINE.md`
- Archive policy and source-of-truth boundary: `docs/ARCHIVE_POLICY.md`

### Architecture & Internals

- **Execution Model** — `docs/explicit-execution-model.md`
- **Explainability** — `docs/system-level-explainability.md`
- **Failure Handling** — `docs/failure-model.md`
- **Confluence & Semantics** — `docs/confluence-law.md`
- **Historical migration notes** — `archive/docs/legacy-root-notes/` (reference only, not runtime authority)

### Deep Validation

Supplemental scale and stress harness:

```bash
python tools/system_validation_at_scale.py --turns 1000
```

This exercises selected long-horizon and adversarial scenarios. It does not establish production reliability or end-to-end replay equivalence. Replay and deterministic recovery remain experimental and are disabled in the default product path.

The checked-in [system audit report](artifacts/system_audit_matrix_report.json) records 15/15 checks passing with `deep_scan: false`. These are selected structural and invariant checks, not a full deep scan, live-provider evaluation, smart-home integration test, or certification of replay behavior.

### Runtime Health & Observability

- **Health Levels**: `green` (normal) → `yellow` (pressure) → `red` (critical)
- **Adaptive Guardrails**: Auto-throttling when memory/prompt pressure rises
- **Status Dashboard**: Real-time memory usage, trim counters, runtime health signals

#### Local OpenTelemetry span export

Install the `service` extra and set `DADBOT_OTEL_ENABLED=1` (or use API `--otel`). Streamlit boot, API startup and spawned service workers share the same opt-in tracing setup. Set `DADBOT_SERVICE_NAME` to a non-sensitive deployment label.

The DadBot-owned SDK provider uses a span processor and a local JSON-lines console exporter on **stderr**. It covers scheduler/service spans, synchronous/asynchronous Ollama requests (including full stream consumption), and executed MCP calls. Nested operations retain OpenTelemetry parent/child IDs without changing canonical kernel IDs. Export is immediate; API/worker shutdown explicitly flushes, and the SDK also registers process-exit cleanup.

Exported fields are allowlisted: fixed operation names, generated trace/span IDs, timestamps, status, service label, and supported numeric/enum metadata. Prompts, replies, tool names/arguments/results, session/request IDs, exception messages/stacks, span events and arbitrary attributes are excluded. Unknown operation names are replaced with `dadbot.operation`. This does **not** scrub the application's other logs or stored history, nor enable automatic third-party instrumentation. Raw kernel event export is not enabled by installing the trace bridge.

There is no remote trace exporter configured by default. Redirect stderr to an access-controlled local file if desired and include it in your retention policy. Explicitly enabling tracing without the SDK raises a configuration error; export failures are logged.

---

## 📦 Packaging & Entrypoints

### Entrypoints
- **Primary**: `launch.py` (routes CLI, UI, and API startup)
- **Compatibility**: `Dad.py` (legacy import/launch shim)
- **API wrapper**: `api_entrypoint.py` (forwards API startup through `launch.py`)
- **UI wrapper/surface**: `launch_streamlit.py` and `dad_streamlit.py` (direct Streamlit entrypoints)

### Feature Groups

```bash
uv sync --locked --extra service --extra voice --extra notifications --extra heritage --extra mcp
```

### Reproducible dependency environments

`uv.lock` is the authoritative, UTF-8 universal lock, generated from `pyproject.toml`; the old platform-specific `pip freeze` snapshot is retired. Use **uv 0.13.7**, enforced by project configuration and pinned in CI/Docker.

```powershell
python -m pip install uv==0.13.7
uv sync --locked --extra dev --extra service --extra mcp
uv run --no-sync python -m pytest -m unit -n 2 --timeout=30
```

`uv sync --locked` refuses manifest/lock drift and verifies pinned distribution hashes. CI installs the `dev`, `service` and `mcp` extras into its locked virtual environment; Docker installs `voice`, `service` and `mcp` from the same lock. Other optional extras can be selected explicitly with `--extra`. Do not run synchronization against a running application's virtual environment.

After an intentional dependency change, run `uv lock`, validate the selected environments, and commit both the manifest and lock. Prefer `uv lock --upgrade-package NAME` for targeted upgrades. Do not hand-edit the lock or regenerate it with `pip freeze`. Platform markers keep Windows-only dependencies out of Linux installs; a universal resolution is not a claim that every Python/platform combination has been runtime-tested. OS packages, Python/base images and external models remain separate deployment inputs, not bit-for-bit image reproducibility guarantees.

---

### Jor-El source mode

In Chat, enable **Jor-El source mode** for the selected conversation. New chats
default to ordinary chat; switching back restores that thread's selection during
the current UI session. Status/Workshop probes retain their normal behavior.
Queued retries preserve the original message's mode, even after changing the
thread's selection.
Python callers can pass `source_mode=True` to any `process_user_message*` delivery.

This mode returns **exact source excerpts, not generated answers**. It makes no
model, embedding, planner, or MCP calls, including when evidence exists. No
qualifying source produces the fixed refusal: "I don't have a relevant approved
source for that. Jor-El can only quote the profile and explicitly approved memories."
Safety checks remain active; crisis responses take precedence, and unsafe excerpts
are replaced with a fixed safety refusal. Persona rewrites, check-ins, signoffs,
automatic speech, and turn-triggered model maintenance are disabled in this mode.
Other independently running background jobs are unaffected.

Sources are the profile's `family`, `education`, `sports_preferences`, `facts`,
`biography`, `background`, `interests`, `hobbies`, `values`, `stories`, and `work`
sections, plus style name, listener name, tone, and signoff. Runtime/security
configuration and secret-named fields are excluded. Conversation history,
attachments, and unapproved memories are never evidence.

Use **Review memory citation approvals** to inspect and explicitly approve or
revoke individual ordinary/consolidated memory summaries. Existing records start
unapproved: pinning, model confidence, and model-written approval fields do not
grant consent. Approval is tenant-scoped and bound to the exact text and record
identity; edits invalidate it. Superseded records are excluded. Approvals persist
separately from model-writable memories; revocation and corpus edits invalidate
source-mode request reuse.

Each excerpt includes a content-bound citation ID and its profile path or memory
identity. Citation integrity is checked again after selection and at delivery;
modified quotations, invented citations, or extra claims fail closed. Markdown
inside quotes is displayed as inert code, not executable links/images.
**Retrieval match strength** is the percentage of distinct significant query
tokens found in the source text/label, using the existing English tokenizer.
At least 50% is required; only the strongest tied matches (up to three) are
returned. Matches below 100% are explicitly partial. This lexical score is
neither semantic understanding nor calibrated confidence/probability of truth.
An approved quotation proves provenance, not factual correctness; conservative
refusals and lexical false positives remain possible.

## 🔒 Privacy & Security

- **Local mode** — Ollama chat, local STT, and Piper TTS can be configured to stay on-device.
- **Cloud voice mode** — edge-tts and ElevenLabs send reply text to their respective services; the Auto backend can choose them when available. Do not use Auto when strict local-only processing is required.
- **Diagnostics** — Local logs, events, and snapshots may contain conversation text. Local inference does not mean that no data is written to disk.
- **Story Mode Lock** — Optional password protection for sensitive contexts
- **Data export** — Full access to your memory files (JSON, SQLite)

### Storage, retention, and deletion

| Data | Default local location / configuration |
|---|---|
| Profile and preferences | `dad_profile.json`; `DADBOT_PROFILE_PATH` |
| Long-term memory and JSON mirror | `dad_memory.json`; `DADBOT_MEMORY_PATH` |
| Human memory-citation consent (hashes and revision counters, not source text) | `<memory-file-stem>.citation-approvals.sqlite3` beside the configured memory file |
| Semantic memory and relationship graph | `dad_memory_semantic.sqlite3`, `dad_memory_graph.sqlite3`; `DADBOT_SEMANTIC_DB_PATH`, `DADBOT_GRAPH_DB_PATH` |
| Conversation histories, ledgers, checkpoints, and diagnostics | `session_logs`; `DADBOT_SESSION_LOG_DIR`; also inspect configured log, trace, and backup locations |
| Service session/task/event state and vectors | Configured Redis, PostgreSQL, or vector backends; tenant-scoped data may exist independently of the local files |

Memory-entry PII scrubbing is not a guarantee that conversation histories,
events, logs, attachments, or backups are scrubbed. The optional
`DADBOT_EVENT_ENCRYPTION_KEY` protects supported service event payloads, not every
JSON file, SQLite database, log, or backup. Use filesystem permissions and
appropriate disk/database encryption for the remaining stores.

There is no single automatic retention period or complete erasure command
covering all stores. New chat starts another conversation; it does **not** erase
prior conversations or long-term memory. Operators must set and enforce their
own retention policy, including logs, uploads, snapshots, backups, and replicas.

For complete deletion, stop DadBot and its background workers first. Identify
the affected tenant and the effective configured paths; remove that tenant's
profile, memory, conversation, graph/vector, ledger, checkpoint, and uploaded
data, including relevant SQLite WAL/SHM sidecars and backups. For service
backends, delete the tenant's session/task/event/document/vector records using
the backend's administration tools. Do not delete an entire shared database or
volume to erase one tenant. Remove matching logs, exports, replicas, and backup
copies as well; otherwise restoring them can restore deleted information.
Restart and verify that the deleted data is no longer retrievable.

Compose publishes the UI, API, and Ollama ports on `127.0.0.1` only. Remote access
requires an explicitly configured authenticated TLS reverse proxy or a deliberate
binding change; do not expose Ollama directly to an untrusted network.

---

## 🛠️ Troubleshooting

### Ollama Not Found

```bash
# On macOS
brew install ollama

# On Windows/Linux
# Download from https://ollama.com
```

### Virtual Environment Issues (Windows)

If working with multiple `Dad-Bot` folders, always activate the local venv:

```powershell
& ./.venv/Scripts/Activate.ps1
python -m pytest
```

### First Run Slow?

Initial setup pulls model weights (~4GB). Subsequent runs are instant.

---

## 🤝 Contributing

Found a bug? Have an idea? Open an issue or PR.

## Engineering Guardrails

- Prefer direct, concrete fixes over new framework layers
- Treat replay-mode invariants as strict runtime contracts
- Add trace hooks at manager boundaries instead of scattering instrumentation
- Keep docs aligned with shipped behavior; mark partial features explicitly

---

## 📋 Current Refactor Status

Phase 4 graph hardening is complete. Turn pipeline is frozen around these invariants:

- Graph execution is the default path
- Sync, async, stream entrypoints preserve correlation
- Graph failures are fail-closed
- Legacy fallback execution is guarded

Current work focuses on **system validation & hardening** rather than graph redesign.

## Streamlit Quick Actions

The Status tab includes one-click operational controls:

- Clear semantic index
- Force memory consolidation
- Export memory
- Optimize for hardware (auto-tunes stream and thinking budgets from CPU count)
- Enable or disable quiet mode

The status view also includes rolling health trends so you can watch pressure over time.
