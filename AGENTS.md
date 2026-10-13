# SENTINEL NEXUS AI — Base44 Dev Environment

## Stack
- **Backend**: Python FastAPI (port 8000), SQLite database, uvicorn with --reload
- **Frontend**: React 18 + Vite (port 3000 → 5173), Three.js / React Three Fiber for 3D, Framer Motion, Tailwind CSS
- **Database**: SQLite (stored in a named Docker volume `sentinel-data`)

## Running
```
docker compose -f docker-compose.base44.yml up -d --build
```
Frontend: http://localhost:3000
Backend API: http://localhost:8000/api

## Architecture
- Backend serves a REST API; frontend proxies API calls via `VITE_API_URL`.
- The 3D AI Core uses React Three Fiber (WebGL). A fallback is the canvas container itself.
- Seed data (agents, tools, models, demo assessment/findings) is created on first boot.
- No external secrets required to boot. AI model/agent connections are optional integrations.

## Key Files
- `backend/main.py` — FastAPI app, router registration
- `backend/models.py` — SQLAlchemy models (all entities)
- `backend/seed.py` — Initial catalog + demo data
- `backend/services/copilot.py` — AI Copilot response generation (local fallback or LLM)
- `frontend/src/App.jsx` — React Router setup
- `frontend/src/three/AICore.jsx` — 3D command center visualization
- `frontend/src/store/useStore.js` — Zustand global state

## Language Support
English and Persian (Farsi) via in-app language toggle (Settings or TopBar globe icon).
RTL is supported via `dir="rtl"` on Persian text.

## Notes
- Security tools (Nmap, Nuclei, etc.) are integration targets — they require an external runner.
- Ollama `localhost` refers to the backend container, not the user's machine.
- The app never fabricates tool results or claims tasks succeeded without confirmation.
