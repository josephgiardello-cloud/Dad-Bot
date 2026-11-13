# 🧔 DadBot Sovereign — Your Local AI Dad

A warm, private AI companion with long-term memory that remembers you, grows with you, and feels like having a supportive dad in your corner.

**Local processing is available.** Ollama chat, local memory, local STT, and Piper TTS can run on your machine. Auto voice may use edge-tts or ElevenLabs and send reply text to external services; choose Piper or pyttsx3 for local speech.

---

## Why Dad-Bot?

- **Actually remembers you**: Learns from your conversations over weeks/months. Tracks your goals, relationships, mood patterns.
- **Feels real**: Adaptive personality, emotional awareness, proactive check-ins. More than a chatbot.
- **Privacy controls**: Ollama chat and local voice backends are available; Auto voice may select a cloud-backed TTS service when installed/configured.
- **Voice support path**: Browser voice capture can transcribe locally, submit a normal runtime turn, and synthesize the reply in local Piper sentence chunks. A Piper model must be configured; full-duplex streaming and acoustic barge-in are still experimental.
- **No subscriptions**: Install once, use forever.

---


## ✨ Key Features

- 🧠 **Deep Long-Term Memory** — Remembers conversations, goals, relationships, and emotional context
- 💭 **Relationship Modeling** — Understands trust, emotional momentum, and personal history
- 🎤 **Voice Pipeline (Partial)** — Browser capture → local Whisper → canonical conversation runtime → local Piper reply is wired. Streaming playback, VAD turn detection, and hands-free barge-in are not yet certified.
- 📜 **Oral History Capture** — Record or upload audio, review and correct the local transcript, then explicitly save it as an `oral_history` memory. The original audio is not retained.
- 📅 **Reminders & Calendar** — Local reminders and calendar events are available; feed sync and advanced proactive scheduling are still in development.
- 🎨 **Avatar Support (Partial)** — Avatar rendering exists with fallback paths; generation workflows vary by environment
- 📱 **Mobile-Friendly** — PWA works on phone browsers
- 🔒 **Local modes available** — Auto voice may use cloud services; select a local backend for on-device speech

## Implementation Notes

### Memory retrieval evaluation

Retrieval evaluations use an isolated synthetic vault; they are not accuracy
claims about production memories or source-mode citations. Real embeddings,
reranker, and PostgreSQL cases are opt-in. The test harness is
[`tests/eval/test_memory_benchmark.py`](tests/eval/test_memory_benchmark.py);
setup and runtime configuration are summarized under
[Memory retrieval and performance](#memory-retrieval-and-performance).

### Local service authority consistency

When API authentication is enabled, app construction requires an initialized
SQLite delegation authority and a token secret, in every environment. The
authority file and the actual SQLite service-state file must be separate.
Disabling API authentication selects trusted-local execution; do not expose
that mode as an authenticated service.

SQLite-backed authenticated result publication holds a write reservation on the
authority database while it verifies the delegation and persists the result.
Revocation and membership changes using that database are ordered by the same
SQLite write lock: a revocation or invalidating membership change committed
before verification rejects publication; one arriving after publication
acquires the lock waits until publication ends.
The authority and service-state databases must be separate files.

This fence does not make the two databases one atomic transaction, prevent
expiry during an already-authorized publication, or fence tool and other live
effects. An interruption can persist session state before the task result;
queued work can be retried, but this is not crash-atomic publication. An
identical result for a completed task is accepted without publishing events
again only after its current authority is verified; conflicting results are
rejected. Legacy completed records lacking the original delegated snapshot may
not support repeated publication. It does not provide a distributed or
PostgreSQL authority guarantee.

Task ownership remediation remains incomplete. The experimental
`SQLiteOwnedTasks` repository uses a separate composite-key table and leaves
legacy task rows untouched, but production state stores, task publication and
recovery do not use it. Existing production task operations can still cross
tenant boundaries; the experimental repository is not an authorization
boundary and must not be used as evidence of tenant isolation.

Authenticated worker-bound memory storage and tenant-ledger methods check the
current delegation and tenant before the guarded operation. This is partial:
newly opened local ledger instances, other cached data, and independent direct
filesystem access are not covered by that binding. File-WAL backend reads and
appends recheck authority and tenant-bearing events when a delegated context is
present, and appends hold the authority fence. Authenticated workers now pass
their configured mode into DadBot and require a durable file WAL. Its signed
tenant header is checked against a separately persisted sibling `.binding`
file and an authority-store identity; missing or inconsistent copies fail
closed. `SQLiteDelegationStore.initialize` assigns that identity once. A
consistent SQLite backup preserves it, while a newly initialized database
receives a different identity. Initialize existing authority databases explicitly
before using this experimental WAL format; a missing or unsupported identity
fails closed. Version 1 experimental headers require an explicit migration,
which is not implemented. Restoring an older authority backup can also restore
old membership and revocation state; the identity does not prevent rollback.

This is not yet a permanent security-mode boundary. Trusted-local files and
legacy files remain unbound. A caller that can remove both the header and its
local anchor can still reopen the remaining file through the default backend.
Protecting against arbitrary replacement of both local files requires a
separately protected trust anchor; independent filesystem access is outside
these Python checks. Legacy ownership transitions are not implemented and
legacy data is not automatically claimed. The API runtime's direct DadBot
bootstrap has no delegated startup identity, so authenticated startup now
fails rather than silently falling back to trusted-local execution; the
worker path must be used with a verified delegation. Do not rely on this
experimental implementation for tenant isolation.
These checks do not establish tenant isolation. Authenticated tenant-ledger
commits and local projection commits hold the SQLite authority write lock from
delegation verification through the guarded commit. A revocation committed
first is observed; one arriving afterward waits for the lock. Local tenant
establishment and direct local commit helpers also hold this lock when a
delegated execution context is present. If a process terminates after the WAL
append but before snapshot repair, a later authorized establishment can
replay the committed revision and repair the snapshot without appending it
again. Other memory paths and raw backend calls without a delegated context are
not covered by this guarantee. It is not a transaction across the authority and
WAL or snapshot files.

If an authenticated bot shuts down without execution authority, its final
memory flush raises an authorization error and leaves shutdown incomplete so
the caller can retry with current authority. The pending in-process projection
is not guaranteed to survive process termination.

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

Service execution requires tenant IDs already in the existing lowercase
canonical form: letters, digits, `.`, `_`, and `-`, beginning and ending
with a letter or digit. IDs that would require normalization are rejected at
the service boundary rather than silently mapped to another tenant. This does
not migrate existing tenant aliases or establish storage isolation. Custom
worker bot factories must accept `model_name` and `tenant_id` and return a bot
whose `config.tenant_id` matches the task.

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

The UI records proposals, decisions, dispatch starts, and outcomes in its canonical execution ledger, scoped by tenant, thread, trace, and approval ID. Approval audit events store input/output hashes, not raw arguments or tool results. A proposal or pre-dispatch audit failure blocks the action. A timeout, cancellation, tool error, or missing completion record after dispatch leaves an uncertain outcome; it is never automatically retried. Pending and approved-but-undispatched proposals expire when the runtime restarts; an interrupted dispatched call becomes uncertain rather than being sent again. An in-memory execution ledger does not make these records durable across process exit.

Approved MCP dispatch now uses the kernel's `execute_tool_async` entrypoint, sharing IO recording, scoped/versioned replay validation, live authority checks, and structured audit with synchronous tools. It awaits in the caller's task and does not use the synchronous sandbox's global cache: identical arguments under a new approval represent a new action. Successfully completed calls have separate committed `TOOL_IO_RECORDED` receipts containing the full inputs/results for exact local replay. These records can contain private external-service data; protect the execution ledger and its backups and never commit them to source control. The hash-only approval completion references its IO receipt. Both writes must complete before success is recorded.

`ToolApprovalJournal.replay` returns only a durably successful receipt through kernel replay, validating tenant/thread/trace/approval identity, payload hashes, tool contract version, and environment fingerprint. Graph-dispatched receipts additionally bind the actual continuation trace/request; replay requires a matching `turn_context` for that execution scope. It never connects to the external service, sends an action, or replays an uncertain outcome. Missing or incompatible receipts fail explicitly without live fallback. Pending approvals still expire after restart.

Approve/reject clicks now submit a new canonical continuation turn linked to the original proposal trace and exact approval ID. The original proposal job is already complete and is not reopened. The decision and result appear in conversation history; the continuation runs context, safety, reflection and SaveNode, producing its own job, policy trace and checkpoint. It bypasses fresh tool planning and ordinary chat inference. Changed configuration expires the proposal inside that graph path; stale buttons cannot authorize a replacement proposal. Source mode disables approval decisions. Successful dispatch and uncertain outcomes both retain their kernel IO in canonical checkpoints; normalization, thinning, loading and recovery preserve it, and corrupt IO fails explicitly rather than becoming an empty replay ledger. Checkpoint replay requires the matching recorded decision result and never dispatches the external action.

The Workshop's **Home Assistant action review** projects a tenant/thread-scoped durable action history and an unresolved-uncertainty queue from the canonical ledger. It shows approval identity, recorded outcome, user confirmation and retry policy without exposing full tool receipts. After checking the original proposal and the provider's action history, users can explicitly record that an uncertain action happened or did not happen. A current device state alone does not prove either outcome. Review is a new canonical no-dispatch continuation turn with conversation history, safety audit and checkpoint. Its separate committed `TOOL_ACTION_REVIEWED` event binds the exact uncertain-outcome event and original approval to the review trace/request.

Manual review is labeled `user_confirmation`, not provider verification or tool success. The original `uncertain` status never changes and cannot authorize receipt replay or resend. Conflicting confirmations and stale selections fail explicitly; identical repeated confirmations do not duplicate the audit event. Unreviewed uncertainty requires manual review; confirmed performance and durable success must not be repeated; a confirmed absence, rejection or expired undispatched proposal requires a new proposal and approval before any new action. Started actions must await completion without retry. Source mode disables review. Confirmations survive WAL recovery, while a failed/uncertain ledger append blocks review reads until explicit backend recovery and verification. Pending approvals still expire after restart.

This completes graph-routed UI approval continuations, local IO retention and durable manual reconciliation, not all tool orchestration: provider-verified reconciliation, automatic/durable retry workers, multi-process boot-to-resume certification, and live calendar/Home Assistant/notification acceptance remain separate work. The review queue never schedules or sends an external action. A failed approval continuation may have sent its approved action, so the UI reports the failure and does not enqueue an automatic resend. A failed review continuation reports that its confirmation may already be recorded and requires checking ledger history before retrying. Each explicitly submitted review gets a distinct conversation-attempt identity, so a failed no-dispatch attempt does not permanently block subsequent user review; journal confirmation remains idempotent by original approval/outcome identity. This does not change approval dispatch deduplication or permit replay of an ambiguous external effect. The standalone low-level MCP client helpers remain compatible; the UI approval service owns this journaled boundary.

The MCP loop returns the model's validated final answer, prompted in the Dad persona, rather than raw tool payloads or unparsed plans. If model finalization fails after tool execution, it reports that failure and warns that device actions may already have occurred; it does not claim success or fall back to a "no action performed" reply. This protection does not depend on Streamlit reply normalization.

### Authenticated local MCP server

The optional Dad-Bot-owned MCP server is a separate Streamable HTTP service at
`http://127.0.0.1:8765/mcp` by default. It requires the `mcp` extra,
`DADBOT_API_TOKEN_SECRET`, and `DADBOT_AUTHORITY_SQLITE_PATH`; use the same
secret, issuer, and explicitly initialized authority database as the
authenticated API. Initialize the authority database once with
`SQLiteDelegationStore.initialize(path)` and provision active tenant memberships
before starting the API; API startup validates the existing store but does not
create a missing one. Then start the server with
`python -m dadbot_system.local_mcp_server`.
The server binds to loopback by default; changing `DADBOT_MCP_HOST` to expose
it beyond the local machine requires an appropriately secured network boundary.

An authenticated API client can exchange its API bearer token for a narrowly
scoped MCP token. `$env:DADBOT_API_AUTH_TOKEN` must be a provisioned API token
for an active tenant member with the base scope required by each requested
tool:

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8010/v1/mcp/token" `
  -Headers @{ Authorization = "Bearer $env:DADBOT_API_AUTH_TOKEN" } `
  -ContentType "application/json" `
  -Body '{"tools":["memory_snapshot","list_reminders"],"ttl_seconds":120}'
```

The caller must hold active tenant membership and the base `read` or `write`
scope for every requested tool. Issued bearer tokens are audience-bound to
`DADBOT_MCP_RESOURCE_URL` (default `http://127.0.0.1:8765/mcp`), include
tool-specific grants, and expire within five minutes. The MCP server checks
tenant authority again when each tool or resource is accessed; revocation
therefore blocks subsequent effects. Tool listing and resource listing expose
only items covered by the token. Set `DADBOT_MCP_RESOURCE_URL` consistently in
the API and MCP server if the endpoint differs from the default.

Startup fails closed when the approved tool manifest is missing, invalid, or
does not match the registered tool schemas and descriptions. The manifest is a
reviewed schema-drift lock, not a cryptographically signed manifest. Tokens use
Dad-Bot's existing HMAC-signed service-token mechanism; this endpoint is not a
standard OAuth 2.1 authorization server.

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

The default chat model is `qwen3.5:9b`. Explicit model selections and generation options are preserved. Set `DADBOT_OLLAMA_CHAT_THINK=false` before startup to disable thinking for ordinary Ollama chat; unset or `auto` preserves the model default, and `true` enables thinking. Explicit per-call settings win, and this default does not change safety, planning, or reflection calls. Actual GPU usage and latency depend on the model, prompt, attachments, and other running applications.

Existing profile model choices are not overwritten. If a saved profile selects another model, update its `llm.model` field to `qwen3.5:9b` and restart the app. `DADBOT_LLM_MODEL` can set the default; CLI `--model` and service `DADBOT_DEFAULT_MODEL` overrides remain available.

Chat, Status probes, Workshop probes, and outbox retries call the canonical `process_user_message` entrypoint and validate replies through shared reply finalization. Chat displays, saves, and speaks only the finalized reply; it does not run a separate UI model executor or expose planning JSON as drafts. New chat and Duplicate controls are in the main chat column, and the conversation selector follows the new thread.

The Streamlit runtime boots orchestration services into the bot's existing manager registry, then registers its UI adapters there. The service container, orchestrator, graphs, and persistence aliases share that registry. `ServiceRegistry.boot()` creates a new registry by default; passing `registry=existing` populates and returns that exact instance.

Streamlit retains the canonical safety and maintenance services, including their `safety` and `health` aliases. Ordinary turns run scheduled proactive jobs; source mode skips them. Maintenance failures are logged and reported as degraded, not always-OK ticks. Durable synthesis remains post-commit, never part of the health hot path.

Its context adapter also retains the canonical memory evaluation, snapshot, provenance, and retrieval sections, then adds visible thread messages. Both `context_service` and `memory` refer to that adapter. Context retrieval reuses the memory manager's configured semantic backend, including tenant-qualified SQLite paths under local WAL authority, rather than opening a separate unqualified index.

Browser reconnects reuse one runtime per configured tenant within the Streamlit process rather than opening competing WAL writers. A dedicated reentrant UI admission lock serializes thread selection, turn context, approvals, and media model overrides; the internal session lock is not held across the canonical worker-thread bridge. Version replacement closes the old runtime before publishing its replacement. Media surfaces bind to the same bot and the tab's thread without disabling strict graph mode or using the legacy greeting fast path. Multiple server processes still require separate coordination and are not covered by this local runtime pool.

New turns bind the active tenant and selected thread into the canonical request, jobs, checkpoints, and audit events. Explicit retry/replay contexts retain their original identity. Policy/capability traces and Merkle anchors accept the canonical top-level session metadata as well as the older nested control-plane form; a non-default UI thread must not execute or audit as `default`.

Visible conversation history is stored separately in `session_logs/streamlit_threads.sqlite3` (override with `DADBOT_STREAMLIT_STATE_DB`), keyed by tenant and thread. New and duplicated threads are saved immediately, including empty threads. Each tab keeps its selected conversation in the `thread` URL parameter, so reloads and server restarts restore that selection without a shared last-selected preference. The URL is a selector, not an ownership credential; unavailable threads produce a warning instead of creating or exposing history.

Unqualified legacy `chat_threads` rows are not imported automatically. Back up and verify the database before explicitly assigning reviewed history to its configured tenant in `owned_chat_threads`; do not infer ownership from a thread name or URL. Corrupt stored history raises an explicit error rather than silently becoming an empty conversation. Generated audio is not stored. UI outbox retries and toggle preferences remain browser-session-local; concurrent multi-user operation is not certified by the reload tests.

Local-WAL graph refresh reads committed tenant sources even when runtime state has staged changes; it neither publishes those changes nor relaxes stale-revision checks. The local UI acceptance pass verifies canonical memory snapshots and a green runtime-health record after a real turn. This does not certify every deployment or external integration. An unconfigured optional EventTap still reports its separate checkpoint limitation; canonical graph checkpoint persistence does not depend on that tap.

### Environment Setup

1. Copy `.env.template` to `.env`
2. Set at minimum:
	- `OLLAMA_HOST`
	- `DADBOT_LLM_MODEL`
	- `POSTGRES_PASSWORD` (if using postgres-backed flows)
3. Optionally set `DADBOT_RECEIPT_KEY` to a cryptographically random value of at
   least 32 bytes, consistently across processes, to enable signed stage receipts
   in agent-loop completion metadata. If absent or too short, turns continue
   normally but durable receipts are reported unavailable. Keep the key secret;
   to rotate it while older durable loops may still need recovery, set
   `DADBOT_RECEIPT_VERIFY_KEYS` to a JSON array of previous key strings (each at
   least 32 UTF-8 bytes). The active key signs new receipts; previous keys are
   accepted only for verification. Remove a previous key only after no pending
   loop recovery depends on it. Invalid verification-key configuration fails
   at startup.
4. Keep `.env` local and uncommitted

The install script will guide you through the rest.

### Experimental agent-loop recovery

`AssistantRuntime.run_agent_loop(..., durable=True)` opts into a bounded,
ledger-backed cursor on an fsync-enabled File-WAL installation. It requires
the stable receipt key described above. After an interruption, call
`run_agent_loop(..., resume=True)` with the same initial input, session, and
`DriverLoopPolicy`. Restore the intended observation/reflection hooks explicitly;
callables are not persisted.

Recovery verifies the exact canonical completion, receipt signatures, trace,
stage order, and chain before continuing with the remaining turn/failure/no-op
budgets. If a pending request has no canonical submission, queue, start,
completion, failure, reconciliation, or effect evidence, recovery records a
failed turn and may continue within the configured failure and turn budgets;
it does not resend that request. Any such evidence without a valid completion
keeps the dispatch uncertain and blocks automatic resend. Completed runs return
their recorded counters without calling the model or executing another turn.
Returned records describe only this invocation; counters include recovered
committed turns. Cursor action/system-observation text is bounded to 16,384
characters.

An authorized operator can explicitly attest to a dispatched loop request via
`ExecutionControlPlane.attest_agent_loop_dispatch_not_applied(AgentLoopNoEffectAttestation(...))`.
The attestation must match the pending loop run and turn, confirm both that no
external effect occurred and that execution is quiesced, and run under
authenticated `admin` authority for the tenant. It is recorded with a
privacy-preserving operator reference. Recovery then accounts for the
interrupted turn as failed and replans within the saved budgets; it never
resends the original request. A `resume_eligible` reconciliation without this
attestation, including one applied automatically at boot, does not release an
agent-loop cursor.

Goal completion can be checked with a versioned deterministic verifier on a
durable loop. Set `DriverLoopPolicy(outcome_verifier_id="yourapp.goal.v1")`,
pass an `outcome_verifier` callback, and call `run_agent_loop(..., durable=True)`.
It receives tenant/session and
canonical request/job/trace identities, bounded goal proposals, hash-only tool
outcomes, and receipt evidence already verified with the configured stable key.
Return exactly `"satisfied"`, `"not_satisfied"`, or `"unknown"`. Only
`"satisfied"` reports `goal_status="satisfied"` and stops as `goal_satisfied`;
`"not_satisfied"` permits another bounded turn, while `"unknown"` stops without
claiming success and requires an explicit resume to recheck. Keep the verifier
ID stable and change it when its semantics change; durable resume requires the
same ID and callback. The goal blackboard remains progress input, not proof of
completion. The verifier should check authoritative post-state, not trust
reflection text.

The cursor is a `SESSION_STATE_UPDATED` projection, not a standalone loop
snapshot. Local snapshots remain diagnostics and are neither imported nor used
to authorize replay. Same-tenant/session File-WAL loop runs share a cross-process
claim; the WAL lock is not held across canonical execution. Signed stage receipts
prove execution-chain integrity, not goal postconditions; the optional verifier
combines verified receipts with caller-owned authoritative post-state checks.

PostgreSQL/Redis primitive integration tests do not enable non-file loop resume.
Authenticated PostgreSQL execution remains unsupported, and non-SQLite physical
memory projection recovery is not certified. Prior receipt keys can be supplied
through `DADBOT_RECEIPT_VERIFY_KEYS` during a rotation window. Uncertain
external-effect reconciliation remains separate work.

### Memory retrieval and performance

The default `AssistantRuntime` observation hook now uses
`retrieve_context(..., record_access=False)` with `semantic_top_k=5` (previously
3) and a 12,000-character rendered context budget. Interactive loops, opt-in
durable loops, canonical resume, queued proactive triggers, and heartbeat
dispatches have populated-catalog integration coverage. The initial goal remains
the recall query across generic follow-up replies.

File-WAL observation combines the existing semantic/BM25 ranking with relevant
graph, consolidated, and archive evidence from a revision-fenced committed
tenant catalog. It does not update access counters, call a model to select
consolidated memories, or publish memory changes. Snippets carry content-derived
source IDs, tenant/revision, available provenance, timestamps, conflict references,
and graph edge validity. Consolidated/archive schemas do not provide full raw-fact
lineage; absent provenance is not invented.

The pruner counts the entire rendered context, including identity, task, and
headings and executive startup task notes. It drops whole evidence entries and
their reflection signals rather than cutting provenance fragments, and reports
dropped snippets. If pinned
identity/task text alone exceeds the budget, it raises explicitly. Missing,
corrupt, revoked, or conflicting authority is an error, not empty recall.
Embedding unavailability uses visibly degraded lexical/graph retrieval; legacy
adapters without the native bundle also report degraded status.

`enable_semantic_memory=False` and a custom `observation_hook` opt out of the
default observation hook. Raw `AgentDriverLoop` construction remains
hook-controlled. These options do not bypass canonical kernel memory validation:
File-WAL agent-loop dispatches use read-only tenant-revision evidence instead of
the legacy sovereign-checksum handshake. Ordinary/non-file turns retain their
existing integrity path.

The versioned `memory-observation.v1` synthetic File-WAL/SQLite profile uses
55 records, 257 queries, deterministic offline embeddings, and a frozen
2026-11-12 query clock. It measured recall@5 **0.8863**, nDCG@5 **0.8742**, and
p95 **90.78 ms**, below both 1.20 times its paired **76.54 ms** baseline and
200 ms. Exact-fact recall/nDCG remained 1.0; paraphrase recall was only 0.42.
This is local-profile evidence, not live-model or non-SQLite adapter certification.
See [the hardening plan](docs/AGENT_MEMORY_PLAN.md#2-certify-memory-first-observation)
for subgroup results and remaining limits.

Memory retrieval records catalog and candidate counts, stage durations, index
pending/degraded status, and reranker timing in
`bot.memory_query.get_retrieval_diagnostics()`. These diagnostics do not contain
memory summaries or the user's query. Access counters are persisted with memory
state without rebuilding the semantic and graph indexes for access-only updates.

For explicitly scoped retrieval, `bot.memory_query.relevant_memories_for_input`
accepts inclusive `from_date` and `to_date` keyword arguments in `YYYY-MM-DD`
form. The range uses an explicit `event_time` date when supplied, otherwise
the recorded creation date; entries with neither are excluded. Date portions
of timestamps use their recorded local calendar date. Unscoped retrieval
retains its existing behavior.

When there is no positive semantic score, lexical match, or graph evidence
matching the eligible catalog, retrieval returns no memories rather than
selecting on importance alone. This
does not guarantee no-answer accuracy when an embedding produces a misleading
similarity score; evaluate with representative queries.

The default retrieval mode scores the complete catalog. For a PostgreSQL
pgvector deployment, `DADBOT_MEMORY_CANDIDATE_LIMIT` may be set to a positive
integer to opt into bounded semantic, lexical, and graph candidate selection; `0`
(the default) retains exhaustive ranking. SQLite remains exhaustive. Bounded
selection can change recall, so evaluate it on a representative isolated dataset
before enabling it. An HNSW or IVFFlat index and matching vector dimensions are
needed for approximate index acceleration; scoped filtering can affect recall.

The optional local reranker uses `DADBOT_MEMORY_RERANKER_ENABLED=1` and
`DADBOT_QWEN_RERANKER_PATH`. Its confidence gate can be adjusted with
`DADBOT_MEMORY_RERANKER_TOP_SCORE_THRESHOLD` (default `1.0`, range `(0, 1]`);
lower values skip reranking more often and can reduce recall. The isolated
`tests/eval/test_memory_benchmark.py` benchmark reports recall, retrieval p50/p95
and reranker firing rate. Set `DADBOT_RUN_REAL_EMBEDDING_BENCHMARK=1` to run
its real local Ollama embedding cases; those cases need `nomic-embed-text`.

### Current memory persistence authority

**Opt-in local file-WAL authority:** With
`DADBOT_DURABLE_EXECUTION_LEDGER=1` and
`DADBOT_EXECUTION_LEDGER_BACKEND=file`, an established tenant's committed
memory catalog is authoritative. Indexed additions, updates, staged saves,
and clears commit the tenant revision before repairing snapshots, graph,
and semantic indexes. Repairs use revision fences and run independently.
A post-commit repair failure is reported explicitly, preserves committed
memory, and blocks further writes until restart/recovery. Startup replays
the catalog and repairs derived stores; durable status replay identifies
dirty indexes even when a crash interrupted status recording.
Graph reads project the committed catalog rather than staged turn memory.
This preserves pending updates for SaveNode without leaking them into graph
context or weakening the tenant revision fence.

Repeated committed-memory reads reuse the verified in-process ledger only
when a SHA-256 digest of **all current WAL bytes**, read under the existing
process lock, matches the last successfully loaded snapshot. Both plain
and CRC WALs use this unchanged-file fast path; file size and modification
time alone are never trusted. Binding checks still run on every plain-WAL
check, and delegated/authenticated reads retain full per-event authority
validation even when the bytes match. Any byte change triggers the full parse, seal,
sequence, revision and replay-filter validation before publishing state.
Local writes and new replay filters invalidate the digest. Explicit
`load_from_backend()` recovery always performs a full reload, and failed
reads/writes remain blocked until that explicit recovery succeeds.
Stateful replay filters and custom backend subclasses retain full reloads.
Graph and semantic index fences share a byte-verified committed projection
snapshot while keeping the raw WAL writer lock held through publication.
Every byte change forces a fresh verifier before checking the exact catalog
and revision; staged runtime memory is never used as index authority.
This removes repeated historical verification from context building
without buffering commits, weakening fsync, or suppressing corruption.
Hot-path request/effect indexes copy all event headers but only the payloads
they actually consume; reconciliation reads copy only reconciliation events.
Checkpoint, turn-event and lifecycle lookups likewise select their event
types before copying; trace-to-session lookup needs only event headers.
SaveNode's built-in-ledger authority lookup selects complete `GRAPH_CHECKPOINT`
events before copying, then still compares the full checkpoint and session
state. Payload trace matching, latest-event order, missing-checkpoint fallback
and uncertain-ledger failures are unchanged; custom ledgers keep full reads.
All existing tenant/session filters, malformed-record rejection, continuity
hashes and explicit uncertain-ledger recovery requirements remain in place.
Complete-run trace and composition checks select the union of matching trace
headers and payload job IDs before copying, retaining each selected payload.
The disabled built-in compactor checks the reconciled event count without
copying payloads. Its trusted-local control-plane report hashes every complete
payload under the ledger lock and keeps only isolated scalar digest records.
It does not copy historical payloads or trust their canonical seals as a
substitute for the full payload digest. An unchanged locked ledger version
allows reuse of that exact digest. Changed heads still require a complete
post-operation snapshot and digest; custom ledger/compactor/digest/partition
implementations, replay filters and authenticated execution retain complete
pre-operation snapshots. No events are trimmed, no durable-checkpoint requirement
is relaxed, and uncertainty still requires explicit ledger recovery.
Default full/tail reads still return complete, isolated deep copies, and
trace roots, completion results and ambiguity checks remain ledger-derived.

Verified built-in trusted-local WAL session-state writes omit top-level fields
whose complete JSON value already matches the committed session projection.
This uses the existing `SESSION_STATE_UPDATED` patch contract: CoreState event
history and full runtime state are retained, and even an empty patch still
commits its event and fsync boundary. Comparison and append hold the ledger lock
and use ledger events, never the mutable registry; changed external heads fail
the existing append checks. Tenant/session isolation, session deletion and
`JOB_COMPLETED` result projection are preserved. Custom, filtered, unverified,
non-fsynced and authenticated/delegated ledgers keep complete patches.
Verified append fingerprints compare the complete appended JSON record,
including its seals, rather than Python container equality. Tuple results and
other existing JSON-normalized values therefore retain the verified fingerprint
without changing the original event values returned to runtime callers.

Trusted-local plain/CRC WAL appends also reuse a verified sealed prefix only
after hashing **every existing byte under the writer lock**. The cached
fingerprint binds those verified bytes plus the exact encoded append, not a
possibly altered post-write file. New seals and sequences are always checked;
tenant mutations still validate the entire tenant revision/operation history.
Byte changes, delegated execution and custom backend subclasses retain full
historical validation. Each required flush/fsync completes before caching the
new prefix; a failed append discards it and keeps explicit recovery mandatory.
Full recovery still verifies every historical event, but restores already
verified seals without hashing them a second time. Legacy and replay-filtered
records retain their existing normalization/resealing behavior. These changes
reduce durable write overhead without batching away commit boundaries.

Successful trusted-local writes retain that fingerprint in the runtime ledger
only when the verified prefix matches its previous all-byte fingerprint (or
every persisted record exactly equals its runtime counterpart), and the new
event matches exactly. This prevents routine job/session writes from forcing a
full WAL reload during the next context read without comparing historical
payloads again for an already verified prefix. The refresh still hashes every
byte under the writer lock; a matching canonical head alone is insufficient.
Concurrent successors, differing metadata, replay filters, authenticated
execution and custom ledger/backend classes retain the full refresh path.
Failed appends or runtime publication still require explicit recovery.

Job submission rebuilds the complete lifecycle projection from validated
`EXECUTION_LIFECYCLE` events without copying unrelated memory/checkpoint payloads.
Custom lifecycle projections retain their complete ledger input.
Approval status/restart projections read all approval updates; the action queue
reads approval updates and manual reviews together in one locked snapshot.
These selectors preserve order, tenant/thread validation, isolated results and
uncertain-ledger rejection. They do not narrow validation to only the selected
thread or cache action decisions. Receipt replay still checks the original
referenced event across the complete ledger before validating its IO type.

SaveNode's local memory commits use the same byte-verified append prefix when
it also exactly matches the reconciled shared runtime ledger. After fsync and
snapshot publication, only the new committed event advances that runtime's
head; the subsequent refresh still hashes all WAL bytes to detect another
writer. Changed bytes, stale runtime heads, replay filters and delegated
execution retain full validation/recovery. Tenant establishment evidence,
revision fences and operation uniqueness are checked on every commit.
Graph/semantic publication fences can reuse the backend's verified prefix
after another all-byte check, while still validating the entire tenant
catalog. This avoids repeatedly copying and rehydrating unrelated execution
history during mood updates and maintenance saves. Failed writes or snapshot
publication leave the shared ledger uncertain until explicit recovery; memory
mutation also retains its existing restart/repair requirement.

This local-first indexed-write path requires SQLite projections. Set
`DADBOT_MEMORY_INDEX_BACKEND=sqlite` to select them without changing PostgreSQL
snapshot persistence; unset or `auto` preserves the existing backend selection.
Invalid backend values fail configuration validation. Their
configured filenames gain `.tenant-<full-tenant-hash>` before the extension,
so different tenants do not share an index. Existing unqualified caches are
preserved and rebuilt from the ledger rather than trusted as migration input.
Other indexed backends remain rejected in this path. Do not enable authority
on existing unowned memory snapshots without a reviewed migration: startup
requires tenant ownership evidence and does not infer it from a shared file.
Once established, disabling or switching the ledger is rejected; recovery
must preserve WAL authority rather than overwrite it with a snapshot.
Physical publication generations and the full release/recovery matrix remain
separate acceptance work. See
[durable memory implementation status](./DURABLE_MEMORY_IMPLEMENTATION_STATUS.md).

**Without established file-WAL authority**, the following snapshot-first
behavior applies:

The configured tenant document store is the primary memory snapshot when it is
durable; with a composite Redis/Postgres store, Postgres is primary. In the
default local configuration, the document store is ephemeral and the JSON
memory snapshot is the durable copy. Graph and semantic indexes are separate
derived stores, and the JSON file can also be a mirror or backup. The
execution ledger records control-plane events, not all tenant memory writes.
It is optional and is not transactionally coupled to either snapshot.

Consequently, a successful snapshot write is not evidence that a corresponding
execution-ledger event exists, and a process crash can interrupt the
cross-store coordination. Snapshot-plus-ledger replay parity, automatic
repair of derived indexes after a crash, and safe memory-ledger compaction
are not currently guaranteed. Do not delete snapshots or ledger records on
the assumption that they are interchangeable.
The control-plane session projection does replay its own `memory_store` field
from session events. Subprocess crash tests cover interruption immediately
before and after that event is appended. This is session-scoped recovery only:
it does not rebuild or commit the tenant-wide memory store.

Current tenant memory write and recovery surfaces:

- Normal writes, including UI, maintenance, MCP, archive restore and delete,
  route through `MemoryManager.mutate_memory_store` and
  `MemoryStorageBackend._coordinated_commit`. The coordinator writes the
  tenant snapshot and JSON mirror, then updates semantic and graph stores;
  compensation is best effort, not crash-atomic.
- Semantic-index sync reports an incomplete embedding batch as a failure and
  retains old entries until replacement embeddings have been written. This
  reduces partial index damage but does not provide automatic crash repair.
- `save_memory_store` (including shutdown) and `clear_memory_projection`
  write the snapshot directly. Loading can also write migrated snapshots,
  restore a JSON backup, or initialize the tenant document store. These paths
  do not append tenant mutation events to the execution ledger.
- `MEMORY_STORE` replacement and session-snapshot restore update the in-memory
  projection; a later save can persist it. When a tenant event exists in the
  loaded ledger, session restore and transaction rollback instead preserve its
  committed tenant projection after refreshing the ledger; rollback rebuilds
  rather than restores the old graph. This does not make normal writes
  ledger-authoritative or serialize restore against concurrent writers.
  `save=False` mutations likewise change the projection without an immediate
  durable write.
- The turn-summary ledger, session-state snapshots, exports, forgetting
  backups, semantic index and graph store persist independently. They are not
  interchangeable with the tenant snapshot and can retain sensitive data
  after a memory is removed.

The intended atomicity model is **one authoritative tenant-scoped mutation
log**, with snapshots and indexes as recoverable projections. A durable
mutation becomes committed when its complete ledger record is durable, not
when a mirror or index write succeeds. A write with an uncertain outcome
must block further writes until recovery determines the committed head.
Concurrent writers must serialize and reject stale revisions.

For local mode this requires a durable execution-ledger backend with a
cross-process single-writer boundary and a crash-safe migration of the
existing JSON snapshot into an initial ledger checkpoint. For durable tenant
stores, the ledger backend and any transactionally maintained checkpoint must
use the same store and transaction (for example a Postgres transaction), not
an independent local WAL. A Redis-only deployment must explicitly provide
equivalent persistence and atomicity or reject durable authority mode.
Mirrors and derived indexes must be rebuilt from the authoritative state after
failure; they cannot cause a committed event to be silently rolled back.
Backends without those capabilities must fail closed rather than fall back to
another authority. This is the chosen design, **not an implemented mode**.
It requires migration, tenant isolation, coordinated direct-save/clear paths,
and replay verification before any snapshot or ledger history is retired.
The execution ledger now exposes a tested tenant-event primitive with revision
checks and operation IDs, but the memory manager does not use it yet. It is not
a supported migration or a replacement for the existing tenant snapshot.
The primitive also offers `reconcile_tenant_memory_operation` to reload and
check an operation ID and its expected state after an uncertain append; this
is not yet wired into application memory writes.
Ledger reload validates tenant-event revisions and operation IDs for every
tenant before accepting the stream, and fails closed on malformed events.
For sealed tenant-event appends, file backends also check tenant revisions and
operation IDs while holding their writer lock; PostgreSQL performs that check
inside the transaction holding the ledger head. This protects the ledger
primitive but does not connect application memory writes to it.
If the selected ledger already contains tenant-memory events for the current
tenant, startup refuses to hydrate a snapshot rather than silently overwrite
that committed state. This is a safety stop, not replay or recovery; it does
not detect a missing or truncated ledger that has lost all establishment
evidence.
Disabling the durable execution ledger also refuses startup if the existing
local WAL contains tenant events for the configured tenant. If a PostgreSQL
customer store is configured, switching away from its execution ledger also
refuses startup when it contains tenant events. These checks cannot detect a
missing ledger or authority stored only in an unconfigured backend.
When a local file execution ledger is enabled, startup now attempts a fenced
tenant import and snapshot repair. This path is experimental and incomplete:
local mutations of fields that do not require semantic or graph repair use a
revision-checked WAL commit, but changes that require those index repairs are
rejected until fenced index repair is implemented. Clears and other mutation
surfaces are not yet integrated with this commit path;
PostgreSQL tenant establishment is not implemented, and power-loss durability
of the evidence file's directory entry has not been established on Windows.
Do not treat this implementation as production-ready.
During an already-running process, snapshot-only saves, clears, and the internal
snapshot-write helpers refuse snapshot-only writes if
a refreshed ledger contains tenant events. This guard is not a concurrency boundary: another writer can
commit after the check. It is not a substitute for ledger-backed writes.

The proposed initialization order is: configure paths and select the tenant
store; construct and validate the chosen ledger backend and its writer
exclusion; import a validated legacy snapshot as the initial event only if no
authority has previously been established; replay the tenant stream and repair
its snapshot; then initialize caches, repair derived indexes, construct the
control plane using the same ledger instance, and admit writes or background
tasks. Boot now constructs and validates the control-plane ledger before memory
hydration and reuses it in the control plane, but migration, tenant replay and
repair are not implemented. Import must be serialized and restart-safe: an
existing authority must never silently fall back to an old snapshot, and the
migration must preserve evidence on failure.

The proposed write boundary validates and normalizes against the current
tenant revision, appends one durable mutation with its expected revision,
then publishes the resulting projection. The append is the commit point:
after it succeeds, failed snapshot or index updates require repair rather than
an unlogged rollback. Before it succeeds no durable projection may change;
an ambiguous append blocks further writes until verified recovery. A process
may die after commit before returning, so callers must reconcile by operation
ID rather than assume that an error means rollback. Explicit replacement,
clear, direct save, staged changes and startup migration must respect the same
boundary; session restore must not overwrite newer tenant state.

Compaction must retain a verified durable checkpoint and tail, preserve chain
continuity, and define retention of sensitive historical data.
Execution-ledger compaction is currently disabled: the previous in-memory
trimming did not compact the WAL or provide a durable replay checkpoint and
could break subsequent sequence and hash-chain recovery. The ledger can grow
without bound until safe retention is implemented; monitor disk and memory
usage and preserve existing logs and backups.
The session snapshot engine can reconstruct its session projection from a
hash-checked snapshot plus tail, and can compare a snapshot to a complete
ledger prefix. This is not a durable tenant-memory checkpoint, does not
authenticate a snapshot against an independently trusted head, and does not
make WAL compaction safe.
The existing local `FileLockMutex` now uses a persistent sibling `.guard` file
to serialize lock creation, stale-record recovery and release across cooperating
processes. Do not remove the guard while any process may be using the lock.
This helper is advisory, assumes processes use the same local filesystem and
protocol. Execution WAL reads and appends use this lock, but tenant memory
commits do not; it does not establish a memory authority or distributed fencing.

The turn-summary memory ledger uses an atomic SQLite transaction to append an
entry and update its chain head. `MemoryLedger.verify_chain()` detects malformed
entries and mismatched or missing tails; it does not authenticate timestamps or
protect against an attacker who can rewrite the entire ledger and head. This
ledger records finalized turns, not every memory mutation, and is not the
authority for replaying the memory store. Custom persistence adapters that only
implement the legacy get/set/list interface do not get atomic append semantics.
Turn checkpoints require a configured event tap and a valid session snapshot;
DadBot logs a warning when no tap is configured, and snapshot or checkpoint
errors are surfaced rather than replaced with an empty state. A configured tap
can still skip a checkpoint according to its checkpoint interval. Explicit
shutdown propagates a final memory-flush failure to its caller.
The optional execution ledger (`DADBOT_DURABLE_EXECUTION_LEDGER=1`) is separate
from the memory store. It defaults to a local file WAL; explicitly setting
`DADBOT_EXECUTION_LEDGER_BACKEND=postgres` selects a transactional execution
ledger in the configured PostgreSQL state store instead. This requires a
PostgreSQL durable store and rejects a nonempty local execution WAL because
migration between those ledgers is not implemented. PostgreSQL serializes
writers on a head row and commits the record and head in one transaction;
stale writers must reload. This backend does not yet log tenant-memory writes.
Its file WAL now refuses replay when a record is malformed
or the last record is incomplete; it also refuses to append after an incomplete
tail. Stop all writers and preserve a copy of the WAL before investigating or
restoring from a known-good backup. Do not remove a record merely to make
startup succeed: a missing record can change recovered state. This check is not
automatic recovery or proof of memory snapshot/ledger parity. Chained records
must retain their checksum fields; legacy unchained records remain readable.
Ledger reads and returned appended events are detached copies; modifying them
does not change the in-memory ledger. On replay, the next sequence accounts for
the highest persisted sequence, including gaps in older logs.
Without an independent trusted chain head, whole-file truncation or complete
replacement cannot reliably be detected.

### Optional checkpoint adapters

`SQLiteCheckpointer` remains the default standalone checkpoint backend.
`PostgresCheckpointer` and `RedisCheckpointer` are opt-in adapters and require
`trusted_local=True`; they are not automatically constructed by the application.
`RedisPostgresCheckpointer` uses PostgreSQL as the durable authority and Redis
only as a cache, so cache failure or stale/conflicting cache data cannot replace
the PostgreSQL result. These adapters are not supported for authenticated
production, which continues to require the durable file execution WAL.

File-backed ledger appends now use a shared advisory file lock and verify the
persisted chain before accepting a chained event; stale writers must reload
before retrying. Unchained legacy records can precede chained records but
cannot be appended after a chain starts. All writers must use this backend
and the same local filesystem lock; direct file edits and writers using older
versions do not participate. Verification currently scans the WAL on each
append, so write cost grows with WAL size. This does not yet make the ledger
the tenant memory authority.
If an execution-ledger append fails, its outcome can be uncertain even when
an error is returned. The ledger blocks subsequent reads and writes until
its backend is reloaded and validated. A failed reload leaves it blocked.
Buffered backends additionally refuse to retry or reload after an uncertain
flush, because retrying an already-persisted prefix could duplicate records;
the inner backend and pending batch require explicit reconciliation.
This protects the execution ledger's own state; it does not make memory
snapshots and execution events a single atomic transaction.
For a configured tenant document store, a failed tenant write fails the memory
mutation rather than accepting a local mirror that could later be overwritten
by stale tenant data. The local ephemeral document store also requires a
successful JSON write because its contents do not survive restart. This is
fail-closed behavior, not offline write reconciliation.
When both Redis and Postgres are configured, memory reads and writes go
directly to Postgres rather than relying on a potentially stale Redis copy.
An explicitly configured customer store that cannot initialize fails startup
instead of silently switching to an empty in-memory store.
Graph-store write errors are reported to the memory transaction coordinator,
which attempts to roll back the mutation; rollback itself can fail and is
reported as incomplete. This is not a crash-atomic transaction across stores.

Local snapshot loading tries the previous backup when the primary is malformed
or fails memory-store validation, including a malformed memory catalog or entry.
Invalid catalog mutations are rejected instead of silently discarding entries.
If neither snapshot is valid, loading fails
instead of silently starting with an empty store. A snapshot with a newer,
unsupported schema also fails without rewriting it; use a compatible version
or restore a valid backup. Editing a memory preserves its temporal and
reinforcement metadata, but changing a reviewed transcript removes its old
review evidence rather than treating edited text as reviewed.

`tests\eval\test_memory_long_horizon.py` is a small deterministic six-question
check over memories dated across six months. It exercises time-scoped lexical
retrieval, one no-answer case, access updates, and persistence; its five
answerable cases require perfect recall and precision at one. It uses mocked
semantic retrieval and does not simulate conversations or measure real-model
quality. This check, the no-answer cases, and the mocked 200-question
benchmark run on pushes and pull requests in the Semantic PGVector workflow;
the real-model cases remain opt-in.

Consolidation flags heuristic contradictions for review instead of automatically
superseding a memory: text overlap alone cannot establish that two statements
refer to the same person and time. A user can still explicitly choose which
consolidated memory to keep in the conflict review UI. Review can also mark
both statements as compatible or dismiss both. Compatibility decisions persist
with the pair and suppress the same heuristic conflict during later
consolidation; dismissing both marks them superseded rather than deleting them.

Controlled forgetting writes a uniquely named full-store snapshot under
`artifacts/backups` before modifying the active catalog, and no longer drops
retained entries solely to satisfy a fixed 50-item limit. These snapshots may
contain sensitive data and must be included in deletion and retention policies.
They are backups, not a fully replayable mutation ledger.
The memory-management screen can also run forgetting manually using the
current UTC time; this is a separate operation and does not fabricate a
committed conversation turn.

Newly forgotten memories are also stored in `archived_memories` in the same
memory store, with an archive ID, time, reason, and the original memory
record. The memory-management screen can restore or delete an archived record;
restoration and removal from the archive use one coordinated store mutation.
Archives are excluded from normal retrieval. Deleting an archive entry does not
erase older snapshots, conversation logs, or other copies; follow the full
deletion instructions below when erasure is required. Old backups made before
this archive feature are not automatically imported.

Reading the memory catalog no longer runs heuristic deduplication or rewrites
the store. Saving a catalog no longer drops entries after a fixed count or
merges different memories merely because they share a broad topic. Working
memory remains a bounded recent-turn window; episodic promotion no longer
silently discards old episodes at 256 entries. These retained projections and
the reversible archive can grow; there is no automatic archive retention limit.

### Local Model Context and Timing

Ollama requests use DadBot's configured `CONTEXT_TOKEN_BUDGET` (default 6000)
or the request's explicit `num_ctx`, capped by the model's context capacity.
There is no power-of-two rounding: a 6000-token budget requests 6000 tokens.
A Modelfile's default `num_ctx` is not treated as the architectural capacity.
Before transport, the prompt guard drops older history and trims oversized
messages, preserving the latest task where possible. Thinking-capable models
reserve at least 3072 tokens for generation, capped at half the request window,
unless the effective request explicitly disables thinking. Non-thinking
requests use `RESERVED_RESPONSE_TOKENS` (minimum 64), also capped at half the
window. Invalid `DADBOT_OLLAMA_CHAT_THINK` values fail configuration validation.

Reasoning-only responses that end with `done_reason=length` raise an explicit
budget-exhaustion error rather than succeeding with an empty answer. Other
empty completions also fail at the transport boundary with completion reason
and token counts; hidden reasoning is never substituted for the answer. This
does not guarantee unlimited reasoning fits; increase the configured context
within available memory when necessary. INFO-level Ollama timing logs include
request purpose, token counts, load/generation durations, and completion reason,
but not prompt, answer, or hidden-reasoning text. Mood detection, tool planning,
and chat can each incur a separate generation.
Canonical live-turn logs additionally report wall time and success/error status
through checkpoint completion; they do not include the conversation text.
Concurrent runtime builds serialize Chroma client construction so another
browser session cannot observe a shared Rust backend before startup completes.

Live conversational confluence keys include request identity as well as
tenant, session, wording, and attachments. Repeating the same wording in a new
request is valid. Retrying an existing request preserves its identity and
cached committed result; explicitly reusing its confluence key still enforces
the original execution contract. The request factory allocates an identity
once when the caller has not supplied one.

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

| Capability | Status | Evidence |
|---|---|---|
| Core chat and memory | Available; durability depends on configured backends | Canonical turn path and memory services |
| Voice | Partial; full-duplex streaming is not certified | `dadbot/speech/voice_loop.py`, `dadbot/ui/media/service.py` |
| Reminders and calendar | Local events available; feed sync is partial | `dadbot/agentic.py`, `dadbot/core/boot_mixin.py` |
| Proactive check-ins | Partial | `dadbot/core/boot_mixin.py` |
| Agent Driver Loop | Experimental API; hybrid memory, structured reflection, pre-dispatch plan gate, canonical goal state, and opt-in receipt-verified File-WAL resume; consolidation remains post-commit | `dadbot/runtime/agent_driver_loop.py`, `dadbot/core/agent_loop_recovery.py`, unit and integration tests |
| Avatar rendering | Partial | `dad_streamlit.py`, optional `heritage` dependencies |

Streamlit chat uses the canonical turn path. Restart durability depends on the
selected ledger and storage backends; display history and browser-local retries
are separate from memory authority. Local tests do not certify every deployment
or concurrent-user configuration.

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

The API's `/v1/health` route is a lightweight liveness/identity response.
`/v1/ready` returns 200 only when state storage, configured authority storage,
and workers are available; its checks are grouped into `dependencies`,
`runtime`, `model`, `vector`, and `memory`. Model, vector, and memory probes
are disabled by default, preserving the existing readiness behavior. Set
`DADBOT_READINESS_PROBES=model,vector,memory` to enable selected probes and
`DADBOT_READINESS_REQUIRED=model,vector,memory` to make the selected checks
gate readiness. Probe timeouts are capped at five seconds
(`DADBOT_READINESS_PROBE_TIMEOUT_MS`, default 1000 ms), and results are cached
for ten seconds by default (`DADBOT_READINESS_PROBE_CACHE_SECONDS`). The model
probe checks the configured Ollama model via `/api/tags`; the vector probe
checks `DADBOT_VECTOR_STORE_URL` as a PostgreSQL DSN or HTTP(S) endpoint.
The memory probe checks the configured semantic retrieval backend's connection
with `SELECT 1`; it does not generate or assess embeddings. Memory retrieval
status counts in `/v1/doctor` summarize the latest 100 in-process retrieval
status observations and include no queries, memory content, or exception text.
Compose does not enable these probes by default; production deployments should
require model readiness when Ollama availability is necessary to serve turns,
and require vector/memory readiness only when those backends are part of the
serving path.
The admin-scoped `/v1/doctor` route adds tool-registry, trace-provider, and
cached runtime-health diagnostics and reflects enabled probe results.
Pending-approval status is reported as not exposed until the API has an
approval-status provider.

Authenticated clients with session read access can request
`GET /v1/sessions/{session_id}/actions/explain?limit=10` for a bounded,
human-readable explanation of the latest turn's tool actions. The response is
tenant-scoped and built from checksum-validated audit events; it includes
authority decision metadata and input/output fingerprints, but never raw tool
arguments or results. It is an explanation of recorded action metadata, not a
replay or proof of the complete conversation.

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
- **Local voice setup** — Install the `voice` extra with `uv sync --extra voice`, configure a Piper model path in Voice preferences, and select local Piper. Audio is held transiently during capture and is not saved as memory.
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
