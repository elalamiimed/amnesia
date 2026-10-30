# Amnesia

> **Your local models forgot. Amnesia remembers.**

Amnesia is a **local-first long-term memory layer for Ollama models**. It
remembers who you are and tracks what you do, then injects the relevant parts
back into every conversation - so your local model greets you like someone who
knows you instead of a stranger with a fresh context window.

Everything lives in one SQLite file on your disk:

- **No cloud, no telemetry, no accounts.** Amnesia talks only to your local
  Ollama daemon.
- **Drop-in.** It is an HTTP proxy speaking both the Ollama API and the
  OpenAI-compatible API, so existing apps gain memory by changing one URL.
- **Yours.** One file at `~/.amnesia/amnesia.db`; export it, back it up, forget
  a memory or purge the whole store at any time.

## Why

Local LLMs are private - and forgetful. Every restart starts from zero: no idea
who you are, what you are building, which decisions you already made, or how you
like your answers. The usual fixes are cloud memory services (privacy trade-off)
or ever-growing prompt dumps (does not scale, does not persist).

Amnesia keeps a durable memory next to the model:

| Problem | Amnesia's answer |
|---|---|
| "Who am I talking to?" | **Identity memory** - durable facts, roles and preferences about you |
| "What was I doing last week?" | **Activity timeline** - sessions, tasks, decisions, projects |
| "What did we discuss?" | **Hybrid recall** - keyword + embedding search over memories, events and past conversations |
| "I don't want to write my bio by hand" | **Automatic distillation** - a local LLM extracts durable facts and events from real conversations |
| "My tools can't talk to this" | **Drop-in proxy** - Ollama-native *and* OpenAI-compatible, streaming included |
| "Is my data safe?" | **Local-only storage**, secret redaction, optional bearer token, documented schema |

## Architecture

```text
┌──────────────────────┐   prompt + <amnesia_memory>   ┌──────────────────────┐    /api/chat        ┌─────────────────────┐
│  Chat clients        │ ────────────────────────────► │  Amnesia proxy       │ ──────────────────► │  Ollama daemon      │
│  CLI · Open WebUI    │                               │  127.0.0.1:11435     │   NDJSON / SSE      │  127.0.0.1:11434    │
│  LibreChat · IDE     │ ◄──────────────────────────── │  inject + record     │ ◄────────────────── │  local models       │
│  MCP agents · curl   │        streamed answer        │  + distill           │    model output     │                     │
└──────────────────────┘                               └──────────┬───────────┘                     └─────────────────────┘
                                                                  │
                                               ┌──────────────────▼───────────────────────────────┐
                                               │           ~/.amnesia/amnesia.db                  │
                                               │  sessions · messages · memories · events ·       │
                                               │  extractions · FTS5 indexes · float32 vectors    │
                                               └──────────────────────────────────────────────────┘
```

The loop:

1. **Capture** - a chat turn arrives from the CLI or the proxy; the session and
   the user/assistant messages are stored.
2. **Distill** - a local LLM extracts durable facts (`memories`), concrete
   activity (`events`) and a session summary; runs are audited in `extractions`.
3. **Retrieve** - for each new message, FTS5 keyword search and embedding
   cosine similarity are fused with reciprocal-rank fusion, then re-ranked by
   importance and recency.
4. **Inject** - the budgeted result becomes an `<amnesia_memory>` system block
   in front of the model; the model answers like it remembers.
5. **Control** - everything is inspectable and editable through the CLI,
   HTTP API, dashboard and one documented SQLite schema.

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and
[docs/SCHEMA.md](docs/SCHEMA.md).

## Quickstart

**Prerequisites**

- Python **3.10+** (3.11 recommended)
- [Ollama](https://ollama.com/download) running on `http://127.0.0.1:11434`
- A chat model and an embedding model:

  ```powershell
  ollama pull dolphin3:latest
  ollama pull nomic-embed-text
  ```

### Windows (PowerShell)

```powershell
# 1. From the repository root: virtual environment + editable install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,mcp]"

# 2. Create the local memory store (~/.amnesia) and run diagnostics
amnesia init
amnesia doctor

# 3. Chat with memory (streaming REPL; try /help, /remember, /recall, /quit)
amnesia chat dolphin3:latest

# 4. Or run the memory proxy and point any client at http://127.0.0.1:11435
amnesia serve
```

> If the `amnesia` command is not on your `PATH` (for example because the
> virtual environment is not activated), `python -m amnesia <command>` runs the
> exact same CLI from the source tree.

### macOS / Linux

```bash
cd amnesia
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,mcp]"

amnesia init
amnesia doctor
amnesia chat dolphin3:latest

# In another terminal, give memory to every Ollama/OpenAI client:
amnesia serve
```

### One-command setup (Windows)

The helper scripts wrap the steps above, are idempotent and fall back gracefully
when an offline machine already has the dependencies:

```powershell
.\scripts\setup.ps1 -Interactive -PullEmbedModel   # venv, install, Ollama check, onboarding
.\scripts\chat.ps1                                 # memory-augmented REPL
.\scripts\serve.ps1                                # proxy + dashboard
.\scripts\doctor.ps1                               # diagnostics
```

### First five minutes

```powershell
amnesia init --interactive        # answer a few onboarding questions (optional)
amnesia remember "Prefers concise answers" --kind preference --key answer_style
amnesia recall "how do I like answers?"
amnesia serve                     # then open http://127.0.0.1:11435/amnesia/ui
```

- Dashboard: <http://127.0.0.1:11435/amnesia/ui>
- Health JSON: <http://127.0.0.1:11435/amnesia/health>
- Ollama-native API: `http://127.0.0.1:11435/api/chat`
- OpenAI-compatible API: `http://127.0.0.1:11435/v1/chat/completions`

A complete, narrated walkthrough is in [docs/DEMO.md](docs/DEMO.md).

## Features

- **Identity memory.** Durable facts, identity and preferences about the user,
  stored canonically with optional keys (one row per slot), importance and
  confidence.
- **Activity timeline.** Concrete things that happened - conversations,
  projects, tasks, decisions, files, learnings - with project tags and dates.
- **Hybrid recall.** SQLite FTS5 (BM25) plus local embeddings
  (`nomic-embed-text`), fused with reciprocal-rank fusion and re-ranked with
  importance and recency; budgeted into the model's context window.
- **Automatic distillation.** After a few turns a local LLM extracts durable
  facts, concrete events and a session summary, deduplicates against what is
  already known, and records an audit trail. Runs never block the reply.
- **Drop-in HTTP proxy.** Ollama-native (`/api/chat`, `/api/generate`, tags,
  version, ps, show, pull, delete) and OpenAI-compatible
  (`/v1/chat/completions`, `/v1/models`, `/v1/embeddings`), with streaming
  (NDJSON/SSE) and tool-call passthrough.
- **Interactive CLI.** Streaming chat REPL with slash commands, plus explicit
  `remember` / `forget` / `recall` / `profile` / `timeline` / `note` / `stats`
  commands and `--json` output for scripting. Quiet by default, with `-v`/`-vv`
  for INFO/DEBUG logs.
- **MCP server.** Six stdio tools - `memory_search`, `memory_remember`,
  `memory_profile`, `memory_log_event`, `memory_timeline`, `memory_stats` - for
  Claude Desktop, Cursor and other MCP clients.
- **Web dashboard.** A self-contained page with Profile, Timeline, Sessions,
  Search and Health tabs plus stats cards; add memories, log events, forget
  entries and download an export directly from the browser. No CDN, no build
  step, served by the proxy.
- **Data control.** JSON export/import, SQLite online backup, soft and hard
  forget, event retention, full purge - and a documented schema.
- **Privacy by default.** Binds to `127.0.0.1`, no outbound network, secret
  redaction before storage, optional bearer token for the memory API.
- **Offline-safe.** If the embedding model is missing, a deterministic hash
  embedder keeps recall working; keyword search works even without FTS5.
- **Python API.** The async `amnesia.service.Amnesia` facade is the same API
  the CLI, proxy and MCP server use - embed Amnesia in your own tools.

## CLI reference

All commands load the configuration from `~/.amnesia/config.json` and the
environment (see [`.env.example`](.env.example)); run `amnesia config show` to
print the effective settings and `amnesia --version` for the installed version.
Short aliases exist for the common options (`-m`, `-n`, `-k`, `-d`, `-p`).

Commands are **quiet by default** (only warnings and errors); pass the global
`-v` for INFO logs or `-vv` for DEBUG (before the command name, e.g.
`amnesia -v doctor`), or set `AMNESIA_LOG_LEVEL`. `amnesia serve` logs INFO
by default.

| Command | What it does |
|---|---|
| `amnesia init [--interactive] [--force] [--model TEXT]` | Create `~/.amnesia`, initialize the DB, detect Ollama/models; `--interactive` runs the onboarding wizard |
| `amnesia chat [MODEL] [--session ID] [--no-memory] [--no-stream]` | Streaming REPL with memory injection and slash commands |
| `amnesia ask TEXT [--model] [--no-memory] [--no-stream]` | One question, one answer |
| `amnesia remember TEXT [--kind fact\|identity\|preference\|note\|episode\|summary] [--key] [--importance]` | Store a durable memory (upsert when `--key` repeats) |
| `amnesia forget ID\|--key KEY [--kind] [--hard]` | Soft-delete a memory (hard-delete with `--hard`); `--key` searches every memory kind unless `--kind` is given |
| `amnesia recall QUERY [--limit N] [--events/--no-events] [--json]` | Hybrid search over memory and activity |
| `amnesia profile [--limit N] [--json]` | Show the identity/profile memories |
| `amnesia timeline [--days N] [--limit N] [--project] [--json]` | Show the activity timeline |
| `amnesia note TEXT [--kind] [--project] [--importance]` | Log an activity event |
| `amnesia sessions [--limit N]` | List conversation sessions |
| `amnesia session show SESSION_ID [--limit N]` | Show one session with its messages |
| `amnesia distill [--session ID] [--all] [--limit N]` | Force fact/event/summary extraction |
| `amnesia serve [--host] [--port] [--no-extraction] [--reload]` | Run the HTTP proxy + dashboard |
| `amnesia mcp` | Run the MCP stdio server |
| `amnesia doctor` | Database integrity, Ollama, embeddings, counts, warnings; exit code 1 on hard failures |
| `amnesia stats [--json]` | Store statistics |
| `amnesia export PATH` | Export all sessions/memories/events to JSON |
| `amnesia import PATH [--no-merge]` | Import an export file |
| `amnesia backup PATH` | Safe SQLite backup copy |
| `amnesia config show [--json]` | Print the effective configuration |
| `amnesia purge [--yes] [--events-older-than DAYS]` | Delete old events or everything (destructive) |

Inside `amnesia chat`: `/help`, `/remember TEXT`, `/recall Q`, `/profile`,
`/timeline`, `/events`, `/forget N`, `/new`, `/session`, `/model NAME`,
`/memory on|off`, `/stats`, `/export PATH`, `/quit`.

## Integrations

| Client | How to connect | Address |
|---|---|---|
| `ollama run` / Ollama CLI | Set `OLLAMA_HOST` for the client (daemon stays on 11434) | `127.0.0.1:11435` |
| Open WebUI | OpenAI-compatible connection | `http://host.docker.internal:11435/v1` |
| LibreChat | `librechat.yaml` custom endpoint | `http://host.docker.internal:11435/v1` |
| Continue (VS Code / JetBrains) | OpenAI-compatible model config | `http://127.0.0.1:11435/v1` |
| Claude Desktop / Cursor / agents | MCP stdio server (`amnesia mcp`) | local process |
| Any OpenAI client | Chat Completions API | `http://127.0.0.1:11435/v1` |
| Any Ollama client | Ollama API | `http://127.0.0.1:11435` |
| Dashboard | Browser | `http://127.0.0.1:11435/amnesia/ui` |
| Python code | `amnesia.service.Amnesia` | import `amnesia` |

Step-by-step setup, `curl` examples and MCP config JSON:
**[docs/INTEGRATIONS.md](docs/INTEGRATIONS.md)**.

## Memory model

| Layer | Stored in | What it holds |
|---|---|---|
| Profile | `memories` with `kind in (identity, fact, preference)` | Stable facts about you: name, role, projects, preferences |
| Activity | `events` | What you did, when, in which project (`task`, `decision`, `learning`, ...) |
| Recall | `memories` with `kind in (episode, summary, note)` | Session summaries and notable moments |
| Transcripts | `sessions`, `messages` | Full conversation history, per client/model |
| Index | `memories_fts`, `events_fts`, `messages_fts`, vector BLOBs | Keyword + semantic retrieval |
| Audit | `extractions` | Every distillation run (status, item count, error) |

Retrieval combines keyword and semantic candidates, then budgets the result into
a compact system block:

```text
<amnesia_memory>
## Who the user is
- name: Sam
- Prefers concise answers (preference)
## Relevant facts about the user
- Building a local-first notes app
## Recent relevant activity (local time)
- 2026-10-28 00:31 [amnesia] Fixed the schema migration
</amnesia_memory>
```

Full tables, indexes, lifecycle and retention semantics:
**[docs/SCHEMA.md](docs/SCHEMA.md)**.

## Privacy and security

- **Nothing leaves the machine.** Amnesia makes requests only to your local
  Ollama daemon (`AMNESIA_OLLAMA_URL`, default `http://127.0.0.1:11434`) and
  writes only to `~/.amnesia/` (override with `AMNESIA_HOME` / `AMNESIA_DB`).
  No telemetry, no accounts, no cloud APIs.
- **Secret redaction before storage.** API keys, bearer tokens, passwords,
  connection-string credentials and private-key blocks are redacted by default
  (`AMNESIA_REDACT=true`) before model-extracted or user-provided text is
  persisted - including summaries, activity events, tags and chat transcripts.
  Treat this as best-effort defence, not a reason to paste real secrets into a chat.
- **Localhost bind.** The proxy listens on `127.0.0.1:11435`. If you bind it to
  `0.0.0.0`, you are exposing your memory; use a firewall/VPN and set a token.
- **Optional API token.** Set `AMNESIA_API_TOKEN` to require a token on
  `/amnesia/*` dashboard/memory API calls: send `Authorization: Bearer <token>`
  or `X-Amnesia-Token: <token>` (or `?token=...` on the URL). The dashboard
  page, `/amnesia/static/*` and `/amnesia/health` stay loadable; passthrough
  chat routes stay open by design (single-user, local), so a token is not a
  multi-user auth system.
- **Export contains everything.** JSON exports and SQLite backups include full
  transcripts and profile data (secrets are redacted before storage). They are
  as private as the database.
- **Never commit your store.** `*.db`, `*.db-wal`, `*.db-shm`, `.amnesia/` and
  `.env` are already in `.gitignore`.

## Troubleshooting

### Ollama is not running

`amnesia doctor` (or `GET /amnesia/health`) reports `ollama.available: false`;
`amnesia serve` still starts, but chat requests return **HTTP 502** with a JSON
error. Start the Ollama app or `ollama serve`, check
`http://127.0.0.1:11434/api/version`, then retry. No memory is lost - the
failure happens before anything is persisted.

### The embedding model is missing

Without `nomic-embed-text` Amnesia falls back to a deterministic **hash
embedder** (`hash:512`) and logs a warning: recall still works, but it is
lexical rather than semantic. Install the model and new memories get proper
vectors:

```powershell
ollama pull nomic-embed-text
```

Vectors are tagged with their `embedding_model` and never mixed during search.
Existing rows without vectors are filled by `Amnesia.backfill_embeddings()` (the
Python API). Set `AMNESIA_EMBED_MODEL=none` to run keyword-only on purpose.

### Port 11435 vs 11434

- **11434** = the Ollama daemon. Do not point clients here if you want memory.
- **11435** = the Amnesia proxy. This is the address clients should use.

`OLLAMA_HOST` is a *client* setting: set it to `127.0.0.1:11435` in a shell to
make `ollama run` go through Amnesia, but keep the daemon itself on 11434.
Amnesia connects to the daemon through `AMNESIA_OLLAMA_URL` (default
`http://127.0.0.1:11434`). If 11435 is already in use, an old proxy is probably
still running (`Get-NetTCPConnection -LocalPort 11435 -State Listen`) or pick
another port with `amnesia serve --port 11436`.

### Model names and models that are not installed

Use the exact tags from `ollama list` (for example `dolphin3:latest`, not
`dolphin3`). If `AMNESIA_MODEL` names a model that is not installed, Amnesia
logs a warning and falls back to the first installed non-embedding model; the
extraction model defaults to the chat model unless `AMNESIA_EXTRACTION_MODEL`
is set. `amnesia doctor` shows what was resolved.

### Windows long paths / OneDrive folders

Deep paths (especially inside OneDrive with long repo names) can hit the
Windows 260-character limit for SQLite side files. Fixes:

- Enable long paths: `Set-ItemProperty -Path HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem -Name LongPathsEnabled -Value 1` (admin, then reboot).
- Or keep the store in a short path: `$env:AMNESIA_HOME = "C:\amnesia"`.
- Quote paths containing spaces in commands.

The repo itself is safe to keep wherever you like; only the DB path needs to be
short.

### PowerShell refuses to run the helper scripts

Unsigned `.ps1` files can be blocked by the Windows execution policy
("... is not digitally signed"). Run the scripts for one session with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

You can also unblock files copied from another machine with
`Get-ChildItem .\scripts\*.ps1 | Unblock-File`. The `amnesia` CLI itself is not
affected - `python -m amnesia <command>` always works.

### Legacy Windows code pages (GBK, cp1252)

The CLI is hardened for non-UTF-8 consoles: it reconfigures its output streams
to replace unencodable characters and falls back to ASCII markers (for example
`+` instead of a check mark) when the terminal encoding cannot represent them.
Commands such as `amnesia doctor` and `amnesia backup` therefore work on
GBK/CP936 systems and when their output is redirected. If another Python tool
still raises `UnicodeEncodeError`, set UTF-8 I/O for the session (the helper
scripts do this automatically):

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

### Proxy authentication errors

If you set `AMNESIA_API_TOKEN`, memory API requests without a valid token
return **401**. Send it as `Authorization: Bearer <token>`, `X-Amnesia-Token:
<token>` or `?token=<token>` on the URL. The dashboard page and
`/amnesia/health` stay loadable; the dashboard asks for the token and remembers
it. `curl` examples are in
[docs/INTEGRATIONS.md](docs/INTEGRATIONS.md). Remove the variable and restart
`amnesia serve` to disable auth.

### Everything is slow

Local 7-8B models on CPU are slow. Amnesia's default upstream timeout is 600 s
(`AMNESIA_REQUEST_TIMEOUT`); lower-generation models like `dolphin3:latest` are
faster than coder models for conversation. The first request after a restart
also has to load the model into memory. If extraction makes chat feel slow, it
runs in the background - set `AMNESIA_EXTRACTION=false` before starting the
proxy to turn it off and run `amnesia distill --all` on demand.

### "Database is locked" or a corrupt store

Run a single `amnesia serve` instance per store. The database uses WAL and a
10-second busy timeout so CLI commands can run while the proxy serves requests,
but external tools must not write to the file while Amnesia is running. Check
integrity with `amnesia doctor` (`db_integrity` should be `ok`). To start over:
stop the proxy, delete `~/.amnesia/amnesia.db`, `-wal` and `-shm`, then run
`amnesia init` - or use a throwaway store via `AMNESIA_HOME`.

## Project layout

```text
amnesia/
  amnesia/
    config.py        settings + AMNESIA_* environment variables
    db.py            SQLite schema, FTS5 indexes, migrations
    store.py         synchronous persistence + search primitives
    models.py        dataclasses (Memory, Event, Session, Stats, ...)
    embeddings.py    Ollama embeddings + offline hash fallback
    ollama.py        async Ollama HTTP client (chat/stream/embed)
    retrieval.py     RRF hybrid search + prompt-context builder
    service.py       async Amnesia facade (remember/recall/context)
    extraction.py    LLM fact/event/summary distillation + scheduler
    proxy.py         FastAPI Ollama/OpenAI proxy + dashboard host
    static/          dashboard assets (self-contained HTML/CSS/JS)
    cli.py           Typer CLI (entry point: amnesia)
    mcp_server.py    MCP stdio server
  tests/             offline pytest suite (fake Ollama, temp stores)
  docs/              ARCHITECTURE, INTEGRATIONS, SCHEMA, DEMO
  scripts/           setup / serve / chat / doctor PowerShell helpers
  README.md · CONTRIBUTING.md · LICENSE
  pyproject.toml · .env.example · docker-compose.yml
```

Design notes and the module contract live in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); development setup, tests and style
are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Roadmap

Amnesia v0.1 is a complete single-user product. Ideas for later releases (not
implemented yet):

- Multiple user profiles in one store (the schema already carries `subject`).
- CLI commands for embedding backfill and event deletion.
- Faster vector search (ANN index) for stores with tens of thousands of rows.
- Encrypted export and optional at-rest database encryption.
- Prebuilt container image and a packaged Windows installer.
- More clients (native Ollama integrations, editor extensions) and additional
  MCP tools.

## Status

**v0.1 - beta, under active development.** Built for a single user on a local
workstation; Windows 11 is the primary platform, Linux/macOS work with the
Python package. Expect rough edges, keep a backup of your store
(`amnesia backup PATH`) before upgrading, and report issues with the
information listed in [CONTRIBUTING.md](CONTRIBUTING.md#reporting-bugs).

Tested with Ollama 0.40.x, Python 3.11 and these local models:
`dolphin3:latest`, `R4C3R/qwen2.5-coder-7b-instruct-heretic:q4_k_m` and
`nomic-embed-text:latest`. The offline test suite (330 passed, about 36 seconds)
plus 7 opt-in real-Ollama end-to-end tests (`AMNESIA_REAL_OLLAMA=1`)
run with `$env:PYTHONPATH=(Get-Location).Path; python -m pytest -q` - no
network or real Ollama daemon required.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - how the pieces fit together
- [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) - `ollama run`, Open WebUI,
  LibreChat, Continue, MCP clients, `curl`
- [docs/SCHEMA.md](docs/SCHEMA.md) - database schema and memory lifecycle
- [docs/DEMO.md](docs/DEMO.md) - five-minute scripted demo
- [CONTRIBUTING.md](CONTRIBUTING.md) - development setup, tests, style

## License

[MIT](LICENSE) © 2026 Mohamed El Alami
