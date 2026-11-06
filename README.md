# 🧔 DadBot Sovereign — Your Local AI Dad

A warm, private AI companion with long-term memory that remembers you, grows with you, and feels like having a supportive dad in your corner.

**Local processing is available.** Ollama chat, local memory, local STT, and Piper TTS can run on your machine. Auto voice may use edge-tts or ElevenLabs and send reply text to external services; choose Piper or pyttsx3 for local speech.

Ollama chat clients use a 120-second transport timeout (including waiting for
response headers or the next stream chunk). This is an inactivity limit, not a
total-turn deadline; model retries and multi-step turns can take longer. A
five-second health probe may time out during a normal cold model load.

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
- 📅 **Calendar & Proactive Flows (Partial)** — Core reminder/check-in plumbing exists; full calendar-sync automation remains in progress
- 📬 **Proactive Flows** — Daily check-in is fully supported; calendar sync and advanced scheduling are in development.
- 🎨 **Avatar Support (Partial)** — Avatar rendering exists with fallback paths; generation workflows vary by environment
- 📱 **Mobile-Friendly** — PWA works on phone browsers
- 🔒 **Local modes available** — Auto voice may use cloud services; select a local backend for on-device speech

### Spatial Agent Live Avatar Integration

Dad-Bot can embed a live avatar frontend (for example a fork of josepeon/spatial-agent) directly in the chat presence stage.

Set in .env:

```bash
DADBOT_SPATIAL_AVATAR_URL=http://localhost:8080/index.html
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
$env:DADBOT_SPATIAL_AVATAR_URL = "http://localhost:8080/index.html"
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

### Story Mode (Password Protected)

Story Mode enables deeper personalization learning. Set a password before launch:

```bash
export DADBOT_STORY_MODE_PASSWORD="replace-with-local-secret"  # or set in .env
python launch.py
```

---

## 🧑‍💻 Development & Contribution

### Running Tests

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

---

## 📦 Packaging & Entrypoints

### Entrypoints
- **Primary**: `launch.py` (routes CLI, UI, and API startup)
- **Compatibility**: `Dad.py` (legacy import/launch shim)
- **API wrapper**: `api_entrypoint.py` (forwards API startup through `launch.py`)
- **UI wrapper/surface**: `launch_streamlit.py` and `dad_streamlit.py` (direct Streamlit entrypoints)

### Feature Groups

```bash
pip install -e .[service,voice,notifications,heritage,mcp]
```

---

## 🔒 Privacy & Security

- **Local mode** — Ollama chat, local STT, and Piper TTS can be configured to stay on-device.
- **Cloud voice mode** — edge-tts and ElevenLabs send reply text to their respective services; the Auto backend can choose them when available. Do not use Auto when strict local-only processing is required.
- **No telemetry** — We don't collect usage data
- **Story Mode Lock** — Optional password protection for sensitive contexts
- **Data export** — Full access to your memory files (JSON, SQLite)

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
