# MediRAG — Pre-pilot security gaps

Findings from the 2026-04-20 pre-pilot sweep. JWT enforcement is
**deferred** by explicit decision to keep the testing loop fast; this
file exists so the gaps are not forgotten before the first real user
ever touches the system.

---

## 1. Rotate the leaked Cohere API key — DO THIS TODAY

Commit `364336d` (2026-04-14) committed `.env` to git with a live
`COHERE_API_KEY`. It was "removed" the same day in `6c0d121`, but the
key is still reachable in history — anyone who has ever cloned or
forked the repo has it.

**Action:** revoke the key in the Cohere dashboard, issue a new one,
update the deploy env. A `git filter-repo` / force-push does not help
here; assume the secret is public.

Verify with:

```
git log --all --full-history -p | grep -E "(API_KEY|SECRET|TOKEN)="
```

---

## 2. Backend has no authorization layer

FastAPI connects to Supabase with `SUPABASE_SERVICE_ROLE_KEY`, which
bypasses every RLS policy. The only authorization check on user-scoped
endpoints is "trust whatever `user_id` the client puts in the body /
form / query." RLS policies in `supabase/007…010` exist but do nothing
as long as every backend call uses the service role.

Unauthenticated endpoints (any caller can impersonate any user):

| Endpoint | File | Trust source |
| --- | --- | --- |
| `POST /session/start` | `app/RAG.py:250` | body `user_id` |
| `POST /upload` | `app/RAG.py:452` | form `user_id` |
| `GET /uploads` | `app/RAG.py:842` | query `user_id` |
| `DELETE /upload/{id}` | `app/RAG.py:850` | body `user_id` |
| `POST /query` | `app/RAG.py:1181` | **no `user_id` at all** — session UUID is the only handle |
| `POST /query/stream` | `app/RAG.py:1550` | **no `user_id` at all** |

Concrete attacks this enables during pilot:
- Anyone who observes or guesses a session UUID can drive that
  session through `/query/stream`, injecting messages into another
  user's history.
- Anyone who observes a `user_id` can list that user's uploads via
  `GET /uploads?user_id=<guess>`.
- A malicious client can create sessions for arbitrary `user_id`s
  via `POST /session/start`.

**Planned fix (post-pilot):** Supabase JWT verification FastAPI
dependency on all six endpoints above, reading the user id from the
decoded JWT instead of request inputs. Frontend already holds the
access token via `supabase.auth.getSession()` — wire it through
`buildRequestHeaders` in `frontend/src/app/App.tsx`.

---

## 3. CORS — tightened 2026-04-20

`app/middleware.py` now reads `ALLOWED_ORIGINS` (comma-separated) from
env and falls back to localhost dev origins only. Previously was
`allow_origins=["*"]` with `allow_credentials=True`, a combination
browsers reject outright. **Production deploys MUST set
`ALLOWED_ORIGINS`** — the dev fallback list is not safe to ship.

---

## 4. Admin / corpus ingestion path

`POST /upload_pdf` (`app/RAG.py:331`) is admin-gated via
`X-Admin-Token` against `ADMIN_UPLOAD_TOKEN` env. Endpoint returns 503
if the env var is unset, which is the correct fail-closed behaviour.
No change needed — just make sure `ADMIN_UPLOAD_TOKEN` is a strong
random value in every deploy.

---

## Items explicitly out of scope for pre-pilot

- Full JWT auth rollout (deferred by user decision, 2026-04-20).
- Moving FastAPI off the service-role key toward per-request
  `SET LOCAL role` + JWT propagation so RLS does the work.
- Secret scanning in CI (`gitleaks` / `trufflehog`).
- Rate-limit persistence (currently in-memory per process).
