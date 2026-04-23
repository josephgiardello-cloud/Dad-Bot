# Mobile And PWA Notes

Dad Bot now exposes mobile-oriented shell metadata and a web manifest through Streamlit static serving.

## What Works Now
- Responsive tab shell with a dedicated Mobile tab.
- Web manifest and app icons served from `/app/static`.
- Theme color and mobile web-app meta tags injected into the live Streamlit document.
- Homescreen install flow for browsers that allow manifest-based install or manual Add to Home Screen.
- Voice chat controls in Chat tab with push-to-talk and always-listening modes.

## Voice Notes
- Speech-to-text and text-to-speech run locally when optional voice packages are installed.
- Push-to-talk is the best default for noisy environments and low-power devices.
- Always-listening mode can auto-send each captured utterance, which is best when the app is foregrounded and the microphone permission remains active.

## Limits Of The Current Stack
- Streamlit serves custom assets from `/app/static`, which does not give a clean root-scoped service worker path.
- Because of that, this setup improves installability and homescreen behavior, but it should not be treated as a fully offline-first PWA shell.
- If full offline caching and install-prompt control become required, the right next step is a small dedicated frontend shell or a reverse proxy route that serves a root-scoped service worker.

## Local Usage
1. Run the Streamlit app locally or over HTTPS.
2. Open the Mobile tab for platform-specific install instructions.
3. Use browser Add to Home Screen or Install App to pin the app.
