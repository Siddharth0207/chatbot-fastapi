# LangChain Diamonds — Developer Quickstart

Minimal developer-focused README with the common commands you need to run locally.
Checklist

- Activate the included venv and install deps
- Set runtime env vars (example `.env` or session vars)
- Run the backend (dev/local and LAN variants shown)
- Serve the static frontend locally
- Quick tests and basic troubleshooting commands

1. Activate the repository virtualenv (PowerShell)

```powershell
# from repo root
& .\lang\Scripts\Activate.ps1
python --version
```

1. Install requirements

```powershell
pip install -r .\requirements.txt
```

1. Environment variables (session-only example)

```powershell
#$env:DATABASE_URL must use asyncpg if using async SQLAlchemy
$env:DATABASE_URL = 'postgresql+asyncpg://postgres:password@localhost:5432/postgres'
$env:NVIDIA_API_KEY = 'sk-...'
```

Alternatively create a `.env` in the project root with those keys.

1. Run the backend (development - local only)

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

1. Run the backend (listen on LAN / other devices)

```powershell
# Make sure Windows Firewall allows the port if you do this
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
# For production (no reload) and multiple workers:
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

1. Serve the static frontend directory with Python (bind to localhost)

```powershell
# from the folder that contains home_cb.html
python -m http.server 5500 --bind 127.0.0.1
# then open http://127.0.0.1:5500/home_cb.html
```

1. Quick test requests (PowerShell)

```powershell
# simple GET
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/' -Method Get

# POST /query example
$body = @{ query = 'Find a 1 carat VVS1 diamond' } | ConvertTo-Json
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/query' -Method Post -ContentType 'application/json' -Body $body
```

1. Check if port 8000 is listening

```powershell
Get-NetTCPConnection -LocalPort 8000
# or
netstat -ano | findstr :8000
```

1. Windows Firewall rule (admin) — only when binding to 0.0.0.0 and you need inbound access

```powershell
New-NetFirewallRule -DisplayName "Allow Uvicorn 8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

Troubleshooting tips (short)

- If the browser shows ERR_CONNECTION_REFUSED to 192.x.x.x, confirm the machine IP with `ipconfig` and whether Uvicorn is listening on that interface. Common local IPs are `192.168.x.x` (not `192.167.x.x`) — check for typos.
- For local development, serving the frontend on `127.0.0.1` and running Uvicorn on `127.0.0.1` avoids firewall/network issues.
- The frontend uses `window.location.hostname` to build the backend URL; ensure both frontend and backend are bound to the same interface.

CORS setup that works across dev devices

- Do not use `allow_origins=["*"]` with `allow_credentials=True` (browsers reject this).
- Keep CORS in `.env` so every developer gets the same behavior.
- Start backend with:

```powershell
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- If frontend runs on a LAN URL (for mobile/other devices), add that exact origin (including port) to `CORS_ALLOW_ORIGINS`.

Where things live

- Backend: `app.py` (FastAPI endpoints)
- Core logic: `main.py` (DiamondFinder, extraction, SQL builder)
- Config: `utils/config.py` (reads `.env` via Pydantic settings)
- Prompts: `utils/prompts.py`
- Mappings: `utils/mappings.py`
- Frontend: `home_cb.html`

If you want, I can add a small PowerShell helper script to start both the backend and a static server concurrently.
