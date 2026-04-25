# 🧔 DadBot — Your Local AI Dad

A warm, private, long-term memory AI companion that feels like a real dad.
Runs locally on your computer with Ollama.

**"That's my boy. I love hearing that, Tony."**

## ✨ Features

- Deep long-term memory + continuous learning (RLHF-style)
- Real relationship modeling (trust, openness, emotional momentum)
- Live voice calls (WebRTC + Piper TTS + Whisper STT)
- Calendar awareness (iCal feed sync)
- Proactive nudges & reminders
- Custom avatar generation
- First-run wizard + PIN protection
- Mobile-friendly PWA
- Fully local & private — no data leaves your machine

## 🚀 Quick Start

### 1. Install and Run (Recommended)

```bash
git clone https://github.com/yourusername/dadbot.git
cd dadbot
python install.py
```

This installs dependencies, prepares profile and memory files, checks Ollama, pulls baseline models, and launches Streamlit.

### 2. Quick Launch (after first install)

```bash
python launch.py
```

Or run Streamlit directly:

```bash
streamlit run dad_streamlit.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 3. Production-Oriented Install and Run

Install the package into your current environment:

```bash
pip install -e .
```

Optional feature groups:

```bash
pip install -e .[service,voice,notifications,heritage]
```

Primary runtime entrypoints:

- Streamlit UI: python -m streamlit run dad_streamlit.py
- Console chat runtime: python Dad.py
- API runtime: python Dad.py --serve-api

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- Recommended model: `ollama pull llama3.2` (and `nomic-embed-text` for semantic memory)

## 📸 Screenshots

*(Add 3–4 screenshots here: main chat, voice call, status tab, preferences)*

## 🐳 Docker Support

```bash
docker compose up
```

See `docker-compose.yml` for dev and production profiles.

## Packaging And Entrypoints

- Core DadBot implementation now lives in dadbot/core/dadbot.py.
- Dad.py is a minimal compatibility shim that re-exports DadBot and preserves legacy run behavior.
- Existing imports like from Dad import DadBot continue to work.
- New internal code should import from dadbot.core.dadbot where practical.

## Current Refactor Status

Phase 4 graph hardening is complete and the turn pipeline is now frozen around these invariants:

- DadBot graph execution is the default turn path.
- Sync, async, and stream entrypoints all preserve correlation and trace context.
- Graph failures are fail-closed: they emit `GRAPH_EXECUTION_FAILED`, return a controlled response, and do not invoke legacy fallback execution.
- Legacy `turn_service.process_user_message*` entrypoints are confined to the DadBot facade compatibility surface and are guarded by freeze tests.

This means current work is in system hardening and validation mode rather than graph redesign. Node ordering, Save ownership, reflection timing, and core graph wiring should be treated as frozen unless a concrete bug is found.

## Status/Dashboard Observability

The Status/Dashboard surfaces guardrails and runtime safety signals, including:

- Memory context token usage and pruning state.
- Prompt guard trim counters and last trim token sizes.
- Recent runtime degradation issues and fallback actions.
- Runtime health level (`green`, `yellow`, `red`) with adaptive controls.

## Runtime Health Levels

- `green`: normal operation, no immediate pressure signals.
- `yellow`: growing pressure (memory ratio, trim frequency, or fallback issues). Dad-Bot starts adaptive throttling.
- `red`: sustained high pressure. Non-critical maintenance is delayed and runtime guardrails tighten.

## Adaptive Guardrails

When health turns `yellow` or `red`, Dad-Bot now actively adjusts behavior:

- In light mode, effective background worker concurrency is reduced.
- Prompt-size guard budgets are tightened automatically when trim pressure rises.
- Memory-context budget is tightened to prune lower-priority sections earlier.
- Non-critical maintenance jobs can be deferred during heavy pressure.
- Optional quiet mode suppresses proactive nudges while pressure remains elevated.

Runtime health snapshots are refreshed automatically at key points:

- At the end of each completed user turn.
- At the end of post-turn maintenance.
- During status/orchestration checks with cached refresh windows (about every 5 minutes by default).

CLI quiet-mode commands:

- `/quiet on`
- `/quiet off`
- `/quiet status`

## Streamlit Quick Actions

The Status tab includes one-click operational controls:

- Clear semantic index
- Force memory consolidation
- Export memory
- Optimize for hardware (auto-tunes stream and thinking budgets from CPU count)
- Enable or disable quiet mode

The status view also includes rolling health trends so you can watch pressure over time.
