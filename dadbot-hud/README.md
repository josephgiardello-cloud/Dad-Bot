# DadBot HUD

Zero-G React + Vite cockpit for the DadBot API.

## Surface
- Main log: terminal-style conversation feed
- Subconscious sidebar: kernel-retrieved fragments
- Vitals HUD: live pulse stream over WebSocket

## Config
Set `VITE_DADBOT_API_BASE_URL` to the DadBot API root, for example:

```bash
VITE_DADBOT_API_BASE_URL=http://127.0.0.1:8010/v1
```

## Run
Install dependencies, then start Vite:

```bash
npm install
npm run dev
```
