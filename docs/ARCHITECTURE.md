# Amnesia - Architecture & Team Contract

> **Status:** v0.1 in active development. This document is the contract between
> the parallel workstreams. If a module needs to change a signature documented
> here, coordinate with the Lead first.

## 1. What we are building

Amnesia is a **local-first long-term memory layer for Ollama models**. A user
chats with a local model; Amnesia:

1. **Remembers who they are** - durable facts/preferences about the user
   (`memories` with kinds `identity`/`fact`/`preference`).
2. **Tracks what they do** - an activity timeline (`events`).
3. **Recalls what matters** - hybrid keyword + embedding search, injected into
   the model's prompt as a compact `<amnesia_memory>` block.
4. **Distills conversations automatically** - a local LLM extracts facts/events
   and summarizes sessions after they happen.

Everything lives in one SQLite file (`~/.amnesia/amnesia.db`). No cloud calls.

### Surfaced through four interfaces

| Interface | Entry point | Status |
|---|---|---|
| CLI chat + memory commands | `amnesia ...` (`amnesia/cli.py`) | workstream C |
| HTTP proxy (Ollama + OpenAI compatible) + web dashboard | `amnesia serve` (`amnesia/proxy.py`) | workstream B |
| MCP server (stdio) | `amnesia mcp` (`amnesia/mcp_server.py`) | workstream C |
| Python API | `amnesia.service.Amnesia` | core (done) |

## 2. Repository layout

```
amnesia/
  amnesia/
    __init__.py       # version
    config.py         # Settings, load_settings, env handling    [core, Lead]
    models.py         # dataclasses: Memory, Event, Session, ... [core, Lead]
    utils.py          # time/tokens/redaction/json helpers       [core, Lead]
    db.py             # SQLite schema, FTS5, migrations          [core, Lead]
    store.py          # sync persistence + FTS/vector primitives [core, Lead]
    embeddings.py     # Ollama + hash embedders                  [core, Lead]
    ollama.py         # async Ollama HTTP client                 [core, Lead]
    retrieval.py      # RRF hybrid search + context builder      [core, Lead]
    service.py        # async Amnesia facade                     [core, Lead]
    extraction.py     # LLM fact/event extraction + scheduler    [A]
    proxy.py          # FastAPI app: Ollama/OpenAI passthrough   [B]
    static/           # proxy-served dashboard assets            [B]
    cli.py            # Typer CLI (entry point amnesia.cli:app)  [C]
    mcp_server.py     # MCP stdio server                         [C]
  tests/              # pytest suites (see section 9)
  docs/               # ARCHITECTURE.md user docs, INTEGRATIONS, SCHEMA [D]
  scripts/            # setup/serve/chat PowerShell helpers      [D]
  README.md           # public README                            [D]
  pyproject.toml      # packaging + pytest config                [Lead]
```

**File ownership is strict**: only edit files in your workstream's scope.
If you need a change in a Lead-owned file, report it to the Lead instead of
editing it. Tests for a workstream live in the matching `tests/test_*.py`.

## 3. Environment facts (important)

- Windows 11, PowerShell. Repo root: `C:\Users\hp\OneDrive\Desktop\amnesia`.
- Ollama runs at `http://127.0.0.1:11434` with these models:
  - `R4C3R/qwen2.5-coder-7b-instruct-heretic:q4_k_m` (7.6B, tools-capable)
  - `dolphin3:latest` (8B, completion only)
  - `nomic-embed-text:latest` (768-dim embeddings; **installed**)
- Python 3.11 (`python` on PATH). All needed packages are already installed:
  fastapi, uvicorn, httpx, typer, rich, numpy, python-dotenv, pytest,
  pytest-asyncio, mcp. **Do not pip install** (PyPI is proxied/offline here).
- Run Python from the repo root with `$env:PYTHONPATH = (Get-Location).Path`.
- Do **not** push to GitHub (the Lead owns commits/pushes). Local commits are
  also unnecessary - just leave working-tree changes; the Lead reviews and
  commits.
- Never write outside the repo directory.
- Ollama is slow on CPU: prefer tiny prompts/timeouts in tests, and never make
  the test suite require the real Ollama daemon (mock it).

## 4. Core data model

Schema (see `amnesia/db.py` for the authoritative SQL):

- `sessions(id, title, model, client, started_at, ended_at, metadata, summary)`
- `messages(id, session_id, role, content, created_at, token_count, metadata)`
- `memories(id, kind, subject, key, content, importance, confidence, source,
  session_id, message_id, created_at, updated_at, last_accessed_at,
  access_count, tags, active, embedding, embedding_model, embedding_dim)`
  - kinds: `identity`, `fact`, `preference` (profile) and `episode`, `summary`,
    `note` (recallable moments)
  - `key` + kind/subject is unique among active rows -> upsert semantics
- `events(id, kind, title, details, project, tags, importance, occurred_at,
  created_at, session_id, message_id, source, embedding, ...)`
  - kinds: `conversation|project|task|decision|file|learning|note|other`
- `extractions(...)` - audit trail of distillation runs
- FTS5: `memories_fts`, `events_fts`, `messages_fts` (external content,
  triggers keep them in sync). All timestamps are **UTC ISO-8601 strings**;
  render local time via `utils.human_time`.

## 5. Core module API (Lead-owned; stable)

```python
# config.py
load_settings(home: Path|None, **overrides) -> Settings
Settings fields: home, db_path, ollama_url, proxy_host, proxy_port, api_token,
  request_timeout, chat_model, embed_model, extraction_model, context_window,
  memory_enabled, inject_context, extraction_enabled, extraction_every_n_turns,
  extraction_min_chars, summarization_enabled, redact_secrets,
  memory_token_budget, profile_token_budget, recent_turns, session_gap_minutes,
  log_level, extra
# derived: .database, .config_file, .effective_extraction_model

# store.py  (synchronous; MemoryStore)
initialize() / close()
create_session(...) -> Session; get_session; end_session; list_sessions;
  latest_session; delete_session; set_session_title; set_session_summary
add_message; add_messages; get_message; session_messages; recent_messages;
  count_messages
add_memory(...) -> Memory; upsert_memory_by_key(...) -> Memory; get_memory;
  update_memory; deactivate_memory; delete_memory;
  list_identity_memories; list_episode_memories; list_memories;
  find_memory_by_key; find_similar_memories; touch_memories
add_event(...) -> Event; get_event; update_event; delete_event; list_events;
  event_projects; events_by_day
search_memories_keyword(query, limit, kinds=, subject=) -> [(Memory, score)]
search_events_keyword(...) -> [(Event, score)]
search_messages_keyword(...) -> [(Message, score)]
search_memories_vector(embedding, limit, kinds=, embedding_model=) -> [(Memory, sim)]
search_events_vector(...)
memories_missing_embeddings; events_missing_embeddings
record_extraction; last_extraction; count_turns_since_extraction
stats(embed_model, chat_model) -> Stats; export_data(); import_data(payload);
  backup_to(path); purge_all(); purge_old_events(days=)

# embeddings.py
Embedder: .name, .dim, await embed(texts), await embed_one(text), await probe()
OllamaEmbedder, HashEmbedder, NullEmbedder, create_embedder(settings)

# ollama.py
OllamaClient.from_settings(settings)
  await version(); await tags(); await model_names(); await is_available()
  await resolve_chat_model(preferred) -> str
  await embeddings(model, input) -> list[vector]
  await chat(model, messages, options=, format_=, tools=) -> dict
  await chat_stream(...) -> AsyncIterator[dict]  # decoded NDJSON chunks
  await chat_text(...) -> str; await chat_json(...) -> dict
  await generate(model, prompt, system=, options=) -> dict
  await close()
OllamaError(message, status_code=, body=)

# retrieval.py
ContextBuilder(store, settings, default_budget, profile_budget)
  .build(query, query_embedding, budget=, profile_budget=, embedding_model=,
         session_summary=, include_events=, include_episodes=) -> MemoryContext
MemoryContext.text  # ready-to-inject system block
recall(store, query, query_embedding, limit=, include_events=, include_messages=,
       embedding_model=) -> list[RecallHit]

# service.py (async; the only thing CLI/proxy/MCP need)
Amnesia(settings, store=, embedder=, ollama=)
  await start(); await close()
  .settings .store .ollama .embedder .builder .chat_model .extraction_model
  .embeddings_enabled .embedding_model_id .ollama_available .warnings
  await embed(text) -> vector
  await remember(content, kind=, key=, subject=, importance=, confidence=,
                 tags=, source=, session_id=, message_id=, dedupe=True) -> Memory
  await forget(memory_id=None, key=None, kind=None, hard=False) -> bool
  profile(limit=) -> [Memory]; await profile_text(limit=) -> str
  await log_event(title, kind=, details=, project=, tags=, importance=,
                  occurred_at=, session_id=, message_id=, source=) -> Event
  timeline(days=, limit=, project=, kind=) -> [Event]; timeline_text(...)
  await recall(query, limit=, include_events=, include_messages=) -> [RecallHit]
  await context_for(user_message, budget=, session_id=, include_events=,
                    include_episodes=) -> MemoryContext
  @staticmethod augment_messages(messages, memory_block) -> list[dict]
  @staticmethod system_message(memory_block) -> dict
  start_session(...); get_session(...); await ensure_session(session_id=,
      model=, client=, gap_minutes=, metadata=) -> Session
  await add_turn(session_id, user_text, assistant_text, model=, metadata=)
      -> (Message, Message)
  recent_turns(session_id, limit=) -> [Message]
  end_session(session_id, summary=)
  await backfill_embeddings(limit=) -> {"memories": n, "events": n}
  stats() -> Stats; export(path); import_(path); backup(path); health() -> dict
```

`Amnesia.health()` is the canonical health/status payload (already
JSON-serializable) - `amnesia doctor` and `GET /amnesia/health` should use it.

## 6. Workstream A - `amnesia/extraction.py` (intelligence)

Own **only** `amnesia/extraction.py` + `tests/test_extraction.py`.

Deliver:

```python
@dataclass
class ExtractionResult:
    facts: list[dict]     # {"kind", "key"(opt), "content", "importance", "confidence"}
    events: list[dict]    # {"kind", "title", "details", "project", "importance", "occurred_at"}
    summary: str | None = None
    raw: dict | None = None

class MemoryExtractor:
    def __init__(service: Amnesia, *, model: str|None = None,
                 ollama: OllamaClient|None = None, settings: Settings|None = None)
    async def extract_exchange(user_text, assistant_text) -> ExtractionResult
    async def extract_session_summary(session_id) -> str | None
    async def distill_session(session_id, *, force=False, max_turns=12) -> dict
        # returns {"facts": n, "events": n, "summary": str|None, "skipped": bool}
    async def maybe_extract(session_id) -> None
        # respects settings.extraction_enabled,
        # settings.extraction_every_n_turns, settings.extraction_min_chars,
        # and store.count_turns_since_extraction

class ExtractionScheduler:
    def __init__(extractor: MemoryExtractor, *, max_concurrency: int = 1)
    def schedule(session_id) -> None      # fire-and-forget, never raises
    async def drain() -> None             # wait for pending work (tests/CLI)
    async def close() -> None
```

Requirements:
- Use `ollama.chat_json` (or `chat` with `format_="json"`) against the local
  extraction model (`service.extraction_model`, fallback `service.chat_model`).
- Robustly parse JSON (models wrap it in prose/fences). Tolerate/repair
  malformed candidates; never crash on bad model output.
- Prompts must instruct: only durable facts about the user (identity,
  preferences, projects, constraints); concrete activity events (what was done,
  decided, learned - not small talk); no secrets; canonical short statements
  ("Prefers concise answers", not "The user said they prefer...").
- Store via `service.remember(...)` / `service.log_event(...)` so redaction,
  embedding and dedupe apply. Event `occurred_at` defaults to now; extract a
  date only when the text clearly states one (ISO string).
- Update session summary via `service.store.set_session_summary`; add a
  `kind="summary"` memory (source `"extraction"`).
- Record every run via `store.record_extraction(...)` (status/items/error).
- Never recurse into the proxy; talk to Ollama directly.
- Tests: fake Ollama (see `tests/conftest.py`) returning canned JSON; cover
  happy path, fenced JSON, invalid JSON, no-op thresholds, dedupe upsert,
  scheduler drain.

## 7. Workstream B - `amnesia/proxy.py` (+ `amnesia/static/`)

Own **only** `amnesia/proxy.py`, `amnesia/static/**`, `tests/test_proxy.py`.

Deliver `create_app(service: Amnesia | None = None, *, settings: Settings | None = None, upstream_client: httpx.AsyncClient | None = None) -> FastAPI`
and a module-level app factory usable as `uvicorn.run(app, ...)`.
`upstream_client` exists for tests: when provided, forward all upstream Ollama
requests through it (it is a `MockTransport`-backed client); otherwise the proxy
creates its own `httpx.AsyncClient` from `settings.ollama_url`.

### Passthrough routes (memory injection + transcript capture + extraction)

- `POST /api/chat` - Ollama native, streaming + non-streaming. Inject memory
  context into the messages, forward to upstream Ollama (`service.settings.
  ollama_url`), tee the response to the client, persist the turn, schedule
  extraction.
- `POST /api/generate` - same idea for prompt/system.
- `GET /api/tags`, `GET /api/version`, `GET /api/ps`, `POST /api/show`,
  `POST /api/pull`, `POST /api/delete` - transparent passthrough.
- `POST /v1/chat/completions` - OpenAI-compatible, streaming (SSE) +
  non-streaming, including tools/tool_calls passthrough.
- `GET /v1/models` - proxy Ollama tags into OpenAI model list format.
- `POST /v1/embeddings` - passthrough to `/api/embed` (OpenAI shape).
- Never inject memory into system messages that already contain
  `<amnesia_memory>`; strip/ignore our own block on re-entry.
- Bypass controls: header `X-Amnesia-Memory: off`; body `metadata.amnesia_memory
  = false`; also skip for model names starting with `amnesia-extract`.
- Client timeout: streaming responses must not be buffered; use
  `StreamingResponse` with `media_type="application/x-ndjson"` (Ollama) or
  `text/event-stream` (OpenAI). Preserve upstream status/headers where sensible.
- If upstream is unreachable, return HTTP 502 with a helpful JSON error.

### Memory API (used by dashboard, CLI and tests)

- `GET /amnesia/health` -> `service.health()`
- `GET /amnesia/profile?limit=50` -> `{"facts": [Memory...]}`
- `GET /amnesia/recall?q=...&limit=10` -> `{"hits": [RecallHit...]}`
- `POST /amnesia/remember` (JSON: content, kind?, key?, importance?, tags?)
- `DELETE /amnesia/memories/{id}` -> forget (deactivate)
- `GET /amnesia/events?days=7&limit=50` -> `{"events": [...]}`
- `POST /amnesia/events` (JSON: title, kind?, details?, project?, importance?)
- `GET /amnesia/sessions?limit=20` -> `{"sessions": [...]}`
- `GET /amnesia/sessions/{id}` -> session + messages
- `GET /amnesia/stats` -> `service.stats().to_dict()`
- `GET /amnesia/export` -> JSON export (attachment)
- `GET /amnesia/ui` -> dashboard HTML (see below)
- `GET /` -> redirect to `/amnesia/ui`
- Optional auth: if `settings.api_token` is set, require
  `Authorization: Bearer <token>` (or `X-Amnesia-Token`) on all `/amnesia/*`
  routes except `/amnesia/health`; passthrough routes stay open (local).

### Session resolution

Header `X-Amnesia-Session` (or `X-Conversation-Id`) wins. Otherwise reuse the
latest session for the same `model` + `client` if idle < `session_gap_minutes`,
else create a new one. `client` = short label from `User-Agent`/route (e.g.
`openai`, `ollama`, `openwebui`). Persist `session_id` + `model` in message
metadata. Never fail a chat request because memory persistence failed - log and
continue.

### Dashboard

Single self-contained HTML page (vanilla JS/CSS, no CDN, no build step) served
at `/amnesia/ui` with tabs: **Profile**, **Timeline**, **Sessions**,
**Search**, **Health**. Calls the `/amnesia/*` endpoints. Dark theme, clean,
responsive; show stats cards (memories, events, sessions, DB size, embeddings
status). Nice-to-have: memory add form, event log form, forget buttons.

Tests (`tests/test_proxy.py`): use `httpx.AsyncClient(transport=ASGITransport
(app=app))`, a temp store + HashEmbedder + fake Ollama client (fixtures in
conftest) and a fake extraction pipeline object or disabled extraction. Cover
both chat formats (stream + non-stream), injection present/absent, transcript
persistence, memory endpoints, auth, dashboard HTML, upstream failure -> 502.

## 8. Workstream C - `amnesia/cli.py` + `amnesia/mcp_server.py`

Own **only** `amnesia/cli.py`, `amnesia/mcp_server.py`, `tests/test_cli.py`,
`tests/test_mcp.py`.

### CLI (Typer, entry point `amnesia.cli:app`, rich output)

Commands (naming is part of the contract; docs are written against it):

```
amnesia init [--interactive] [--force] [--model TEXT]
amnesia chat [MODEL] [--session ID] [--no-memory] [--no-stream]
amnesia ask TEXT [--model] [--no-memory]
amnesia remember TEXT [--kind truth|identity|preference|note] [--key] [--importance]
amnesia forget ID|--key KEY [--kind] [--hard]
amnesia recall QUERY [--limit N] [--events/--no-events] [--json]
amnesia profile [--limit N] [--json]
amnesia timeline [--days N] [--limit N] [--project] [--json]
amnesia note TEXT [--kind] [--project] [--importance]
amnesia sessions [--limit N]            # list
amnesia session show SESSION_ID
amnesia distill [--session ID] [--all] [--limit N]
amnesia serve [--host] [--port] [--no-extraction] [--reload]
amnesia mcp
amnesia doctor
amnesia stats [--json]
amnesia export PATH
amnesia import PATH [--no-merge]
amnesia backup PATH
amnesia config show
amnesia purge [--yes] [--events-older-than DAYS]
```

Behaviour:
- All commands load `Settings` via `load_settings()`, start an `Amnesia` service
  (`asyncio.run` via `utils.run_sync`), and close it.
- `chat`: interactive REPL with streaming output (`ollama.chat_stream`), memory
  context injected each turn (`service.context_for` + `augment_messages`),
  session persisted, extraction scheduled after replying (if enabled).
  Slash commands inside chat: `/help`, `/remember TEXT`, `/recall Q`,
  `/profile`, `/timeline`, `/events`, `/forget N`, `/new`, `/session`,
  `/model NAME`, `/memory on|off`, `/stats`, `/export PATH`, `/quit`.
- `init`: create `~/.amnesia`, initialize DB, print detected Ollama/models,
  optionally run an onboarding wizard storing identity facts (name, role,
  location, languages, projects, communication preferences) - one `remember`
  per answer, `source="onboarding"`. `--force` re-runs.
- `doctor`: show DB path/integrity, Ollama version/models, embedding model
  availability, extracted counts, warnings; exit code 1 on hard failures.
- Errors: friendly `rich` messages, non-zero exit code on failure; never show a
  raw traceback for expected errors (bad ID, Ollama down, missing file).
- `serve`: `uvicorn.run(create_app(service), host=..., port=...)`; start service
  first, print URL + dashboard URL.
- `mcp`: build service and run stdio MCP server.

### MCP server (`mcp_server.py`)

Use the installed `mcp` package (`mcp.server.fastmcp.FastMCP`) if importable;
expose tools that return concise JSON-serializable dicts:

```
memory_search(query, limit=8, include_events=True) -> {hits: [...]}
memory_remember(content, kind="fact", key=None, importance=0.6) -> {memory: {...}}
memory_profile(limit=30) -> {facts: [...]}
memory_log_event(title, kind="note", details=None, project=None) -> {event: {...}}
memory_timeline(days=7, limit=20) -> {events: [...]}
memory_stats() -> {...}
```

`main()`: create service, register tools, `mcp.run()` (stdio). Guard import so
`amnesia mcp` prints a helpful install hint if the `mcp` extra is missing.
Tests: call the underlying tool functions / server object directly; do not
require a client protocol handshake.

## 9. Workstream D - docs & ops

Own `README.md`, `docs/**` (except `docs/ARCHITECTURE.md` if the Lead created it -
check before editing), `scripts/**`, `.env.example`, `docker-compose.yml`,
`CONTRIBUTING.md`, `LICENSE`.

- README: compelling, accurate quickstart (Windows PowerShell + generic shell),
  feature list, architecture diagram (ASCII or mermaid), table of integrations,
  CLI command reference, privacy/security, troubleshooting (proxy, model
  naming, `OLLAMA_HOST`), roadmap. Keep it honest - no feature that does not
  exist in the code.
- `docs/INTEGRATIONS.md`: step-by-step for (a) `ollama run` via
  `OLLAMA_HOST=127.0.0.1:11435`, (b) Open WebUI (add OpenAI-compatible
  connection `http://host.docker.internal:11435/v1`), (c) LibreChat, (d)
  Continue / VS Code, (e) MCP clients (Claude Desktop / Cursor config JSON),
  (f) plain `curl`. Include the exact env var names/paths and a note that the
  proxy listens on 11435 while the Ollama daemon stays on 11434.
- `docs/SCHEMA.md`: schema, memory lifecycle, retention/export/import.
- `docs/ARCHITECTURE.md`: high-level user-facing architecture (code structure,
  data flow, extraction loop, privacy guarantees). May overlap section 1-5 but
  written for users.
- `scripts/setup.ps1`, `scripts/serve.ps1`, `scripts/chat.ps1`: idempotent
  helpers; `setup.ps1` creates a venv, installs `-e ".[dev,mcp]"` if network
  allows, checks Ollama, optionally pulls `nomic-embed-text`.
- `.env.example` with every `AMNESIA_*` var and comments (mirror config.py).
- `docker-compose.yml`: optional proxy container (`host.docker.internal` for
  Ollama), env comments, volume for `~/.amnesia`.
- `CONTRIBUTING.md` (dev setup, tests, style), MIT `LICENSE`.
- Also `docs/DEMO.md` or a "first 5 minutes" section with a scripted demo.

## 10. Workstream E - QA / end-to-end (spawned later)

Own `tests/test_integration.py`, `tests/test_e2e.py`, `tests/test_utils.py`
(if not already), plus a top-level `Makefile`/`pytest.ini`-free verification
report. Run full suite, cross-review modules, test against real Ollama
(in-scope models only, tiny prompts), verify proxy with a real `ollama` client
call if possible, and report findings to the Lead.

## 11. Tests & conventions

- `pytest` from repo root: `$env:PYTHONPATH=(Get-Location).Path; python -m pytest -q`
- Shared fixtures in `tests/conftest.py`:
  - `tmp_home`, `settings` (temp home, hash embeddings, extraction disabled by
    default unless requested), `store`, `service` (async, started without
    Ollama), `fake_ollama` (httpx.MockTransport-backed client + recorded
    requests), `client` (httpx ASGI client factory).
- No network in tests. No sleeps > 0.1s. Deterministic.
- Code style: Python 3.11, `from __future__ import annotations`, type hints,
  dataclasses, module loggers (`logging.getLogger("amnesia.<module>")`).
  Keep lines <= 100 chars; avoid new third-party dependencies.
- User-facing errors are human-readable; internal errors logged with context.
- All stored timestamps UTC; display local.
- Secret redaction (`settings.redact_secrets`) must be respected anywhere we
  persist model-extracted or user-provided text.

## 12. Definition of done (global)

1. `python -m pytest -q` is green (aim: 80+ meaningful tests).
2. `amnesia init --interactive` works on this machine (real Ollama).
3. `amnesia chat` remembers a fact across two separate runs.
4. `amnesia serve` proxy: `curl /api/chat` gets memory-injected model output;
   `/v1/chat/completions` works; `/amnesia/ui` renders.
5. MCP server starts (stdio handshake via `mcp` client or manual initialize).
6. README quickstart verified; docs accurate; no secrets or local DB committed.
7. Lead reviews diff, commits, pushes private repo `elalamiimed/amnesia`.
