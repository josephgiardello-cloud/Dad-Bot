# 🧔 DadBot Sovereign — Your Local AI Dad

A warm, private AI companion with long-term memory that remembers you, grows with you, and feels like having a supportive dad in your corner.

**Runs 100% locally. No data leaves your machine.**

---

## Why Dad-Bot?

- **Actually remembers you**: Learns from your conversations over weeks/months. Tracks your goals, relationships, mood patterns.
- **Feels real**: Adaptive personality, emotional awareness, proactive check-ins. More than a chatbot.
- **Privacy-first**: Ollama-powered, runs on your computer. Your data is yours.
- **Voice support path**: Local voice stack integration points are included; setup maturity depends on your machine tooling.
- **No subscriptions**: Install once, use forever.

---

## ✨ Key Features

- 🧠 **Deep Long-Term Memory** — Remembers conversations, goals, relationships, and emotional context
- 💭 **Relationship Modeling** — Understands trust, emotional momentum, and personal history
- 🎤 **Voice Pipeline (Experimental)** — Local STT/TTS integration hooks are present; full turnkey setup is still maturing
- 📅 **Calendar & Proactive Flows (Partial)** — Core reminder/check-in plumbing exists; full calendar-sync automation remains in progress
- 📬 **Proactive Nudges** — Reminders and check-ins tailored to you
- 🎨 **Avatar Support (Partial)** — Avatar rendering exists with fallback paths; generation workflows vary by environment
- 📱 **Mobile-Friendly** — PWA works on phone browsers
- 🔒 **100% Private** — No cloud, no tracking, fully local

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

No screenshots are checked in yet. The first polished pass should include real captures for:

- Main chat view with an active thread
- Workshop status view with health and confluence visible
- Voice panel with local STT/TTS status visible
- Memory or relationship context panel showing continuity
- Mobile PWA chat view

Suggested file names for the first capture set:

- `docs/screenshots/01-chat-main.png`
- `docs/screenshots/02-workshop-status.png`
- `docs/screenshots/03-voice-panel.png`
- `docs/screenshots/04-memory-context.png`
- `docs/screenshots/05-mobile-pwa.png`

---

## 🎯 First-Run Setup

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.com) installed and running
- Recommended: `ollama pull llama3.2` and `ollama pull nomic-embed-text`

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

For regular stability validation, run the broader replay/durability/stress lane:

```bash
pytest --run-stress -m "durability or phase4 or stress" --durations=25 -q
```

### Test Lanes Quick Map

- `Test Lane: DEV` (`-m unit`) for fast local iteration
- `Test Lane: INTEGRATION` (`-m integration`) for service/storage boundaries
- `Test Lane: THIN-SPINE VS LEGACY` for parity checks on the migration seam
- `Test Lane: DURABILITY / PHASE4` (`-m "durability or phase4"`) for restart/commit integrity
- `Test Lane: DETERMINISM STRESS` (`--run-stress -m stress`) for replay and strict-equivalence pressure
- `Test Lane: STABILITY SWEEP` (`--run-stress -m "durability or phase4 or stress"`) for broad regular validation
- `Test Lane: CERT / REGRESSION` and `Test Lane: FULL CERT` for certification-oriented full passes

Source of truth: `.vscode/tasks.json`.

### Feature Maturity Matrix

| Capability | Maturity | Evidence |
|---|---|---|
| Core local chat + memory pipeline | Shipped | `dadbot/core/dadbot.py`, `dadbot/core/turn_mixin.py`, `dadbot/core/graph.py` |
| Voice stack hooks and optional deps | Experimental | `pyproject.toml` (`[project.optional-dependencies].voice`), `dad_streamlit.py`, `dadbot/agentic.py` |
| Local reminders + local calendar events | Shipped (local storage) | `dadbot/agentic.py` (`add_reminder`, `add_calendar_event`, `list_calendar_events`) |
| Calendar feed synchronization | Partial | `dadbot/core/boot_mixin.py` (`ical_feed_url`, `schedule_calendar_sync`) |
| Proactive heartbeat/nudges | Partial | `dadbot/core/boot_mixin.py` (`_run_proactive_heartbeat_loop`), `dadbot/assistant_runtime.py` |
| Avatar rendering support | Partial | `dad_streamlit.py`, `pyproject.toml` (`[project.optional-dependencies].heritage`) |

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

Proof-of-correctness runs under chaos/fuzz/replay pressure:

```bash
python tools/system_validation_at_scale.py --turns 1000
```

This validates long-horizon behavior, adversarial fuzzing, and replay equivalence.

### Runtime Health & Observability

- **Health Levels**: `green` (normal) → `yellow` (pressure) → `red` (critical)
- **Adaptive Guardrails**: Auto-throttling when memory/prompt pressure rises
- **Status Dashboard**: Real-time memory usage, trim counters, runtime health signals

---

## 📦 Packaging & Entrypoints

- **Core**: `dadbot/core/dadbot.py` (main DadBot class)
- **Canonical launcher**: `launch.py` (`python launch.py`, optional `--api` / `--ui`)
- **Compatibility**: `Dad.py` (legacy re-export for old imports only)
- **UI shell**: `dad_streamlit.py` (Streamlit surface used by `launch.py --ui`)

### Feature Groups

```bash
pip install -e .[service,voice,notifications,heritage]
```

---

## 🔒 Privacy & Security

- **No cloud calls** — All processing is local via Ollama
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
