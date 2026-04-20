import os

from fastapi.middleware.cors import CORSMiddleware

# Browsers silently reject `Access-Control-Allow-Origin: *` paired with
# credentials, and the wildcard also signals "we haven't thought about
# origins." Read the allow-list from ALLOWED_ORIGINS (comma-separated);
# fall back to a conservative local-dev set so `npm run dev` still works
# out of the box. Production deploys MUST set ALLOWED_ORIGINS explicitly.
_DEFAULT_DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


def _parse_origins(raw: str) -> list[str]:
    return [o.strip() for o in raw.split(",") if o.strip()]


def add_cors_middleware(app):
    env_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
    origins = _parse_origins(env_origins) if env_origins else _DEFAULT_DEV_ORIGINS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Admin-Token"],
    )