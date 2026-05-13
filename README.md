# 🧔 DadBot — Your Local AI Dad

A warm, private AI companion with long-term memory that remembers you, grows with you, and feels like having a supportive dad in your corner.

**Runs 100% locally. No data leaves your machine.**

---

## Why Dad-Bot?

- **Actually remembers you**: Learns from your conversations over weeks/months. Tracks your goals, relationships, mood patterns.
- **Feels real**: Adaptive personality, emotional awareness, proactive check-ins. More than a chatbot.
- **Privacy-first**: Ollama-powered, runs on your computer. Your data is yours.
- **Voice support**: Talk to Dad like you would a real person (WebRTC + TTS).
- **No subscriptions**: Install once, use forever.

---

## ✨ Key Features

- 🧠 **Deep Long-Term Memory** — Remembers conversations, goals, relationships, and emotional context
- 💭 **Relationship Modeling** — Understands trust, emotional momentum, and personal history
- 🎤 **Voice Calls** — Talk naturally with WebRTC + Piper TTS + Whisper STT
- 📅 **Calendar Aware** — Knows your schedule and life events (iCal sync)
- 📬 **Proactive Nudges** — Reminders and check-ins tailored to you
- 🎨 **Custom Avatar** — Visual personality (generated)
- 📱 **Mobile-Friendly** — PWA works on phone browsers
- 🔒 **100% Private** — No cloud, no tracking, fully local

---

## 🚀 Quick Start

### Install and Run

```bash
git clone https://github.com/josephreisinger/dadbot.git
cd dadbot
python install.py
```

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

*(Coming soon: main chat interface, voice call screen, status dashboard, memory graph visualization)*

For now, try it yourself! It takes ~2 minutes to install.

---

## 🎯 First-Run Setup

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.com) installed and running
- Recommended: `ollama pull llama3.2` and `ollama pull nomic-embed-text`

The install script will guide you through the rest.

### Story Mode (Password Protected)

Story Mode enables deeper personalization learning. Set a password before launch:

```bash
export DADBOT_STORY_MODE_PASSWORD="your-password"  # or set in .env
python launch.py
```

---

## 🧑‍💻 Development & Contribution

### Running Tests

```bash
pytest tests/
```

### Docker (Dev & Production)

```bash
docker compose up
```

See `docker-compose.yml` for profile options.

---

## 📚 Technical Documentation

### Architecture & Internals

- **Execution Model** — `docs/explicit-execution-model.md`
- **Explainability** — `docs/system-level-explainability.md`
- **Failure Handling** — `docs/failure-model.md`
- **Confluence & Semantics** — `docs/confluence-law.md`

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
- **Compatibility**: `Dad.py` (legacy re-export, preserves old imports)
- **CLI**: `Dad.py` (console chat, API server)
- **UI**: `dad_streamlit.py` (web interface)

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
