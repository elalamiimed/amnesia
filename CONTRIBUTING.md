# Contributing to Amnesia

Thanks for helping improve Amnesia — a local-first long-term memory layer for
Ollama models. This guide covers development setup, tests, code style and the
pull-request checklist.

## Before you start

- **Python 3.10+** (3.11 recommended; development happens on 3.11).
- **Ollama** running locally at `http://127.0.0.1:11434`
  (<https://ollama.com/download>).
- At least one chat model installed, e.g. `ollama pull dolphin3:latest`, and
  for semantic recall an embedding model: `ollama pull nomic-embed-text`.
- Windows 11 / PowerShell is the primary development platform; Linux and macOS
  are supported by the code and scripts are documented for both shells.

Amnesia stores everything in `~/.amnesia/amnesia.db`. During development you
can point it elsewhere instead of touching your real memory store:

```powershell
$env:AMNESIA_HOME = "$PWD\.tmp-amnesia"   # PowerShell
```

```bash
export AMNESIA_HOME="$PWD/.tmp-amnesia"    # POSIX shells
```

## Set up a development environment

From the repository root:

```powershell
# 1. Create an isolated virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install the package in editable mode with dev + MCP extras
python -m pip install -e ".[dev,mcp]"

# 3. Create the local store and check Ollama/model availability
amnesia init
amnesia doctor
```

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,mcp]"
amnesia init
amnesia doctor
```

If you prefer not to install the package (for example on an offline machine
where the dependencies are already available system-wide), run everything from
the repository root with the source tree on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m amnesia doctor
```

On POSIX: `PYTHONPATH="$PWD" python -m amnesia doctor`.

### Environment variables

Every setting can be overridden from the environment or a `.env` file. Copy
[`.env.example`](.env.example) to `.env` for a documented template, and see
`amnesia/amnesia/config.py` for the authoritative list and defaults.
`amnesia config show` prints the effective values.

## Running the project

```powershell
amnesia init --interactive     # onboarding wizard: stores your identity facts
amnesia chat                   # memory-augmented streaming REPL
amnesia serve                  # HTTP proxy + dashboard on http://127.0.0.1:11435
```

Equivalent entry points that do not need the console script installed:

```powershell
python -m amnesia init
python -m amnesia serve
python -m amnesia.proxy         # proxy only, via uvicorn
```

Open <http://127.0.0.1:11435/amnesia/ui> for the dashboard. The proxy speaks
both the Ollama-native API (`/api/chat`) and the OpenAI-compatible API
(`/v1/chat/completions`); see [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) for
client setup and plain `curl` examples.

The CLI is quiet by default - pass the global `-v` (INFO) or `-vv` (DEBUG)
option before the command name (`amnesia -v doctor`) for logs. `amnesia serve`
logs INFO by default, and an `AMNESIA_LOG_LEVEL` set in the environment is
respected unless a `-v` flag overrides it.

## Tests

The test suite is offline and deterministic: it never talks to the real Ollama
daemon (a `httpx.MockTransport` fake lives in `tests/conftest.py`) and uses a
temporary home directory and SQLite store per test.

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m pytest -q                                  # whole suite
python -m pytest tests/test_store.py -q              # one file
python -m pytest -q -k "recall or profile"           # one subset
```

```bash
PYTHONPATH="$PWD" python -m pytest -q
```

The full suite is 416 offline tests and takes about 60 seconds on the
development machine; `integration`-marked real-model tests (7 of them) are
skipped unless `AMNESIA_REAL_OLLAMA=1` is set.

Rules for new tests:

- **No network, no real Ollama, no sleeps longer than ~0.1 s.**
- Use the fixtures in `tests/conftest.py`: `tmp_home`, `settings`, `store`,
  `service`, `fake_ollama`, `proxy_client`, `make_service`.
- Mock transports/queued replies instead of hitting a model; assert on the
  requests the code made (`fake_ollama.chat_payloads()`).
- Mark opt-in real-model smoke tests with `@pytest.mark.integration` and skip
  them unless `AMNESIA_REAL_OLLAMA=1` is set.
- Assert on behaviour and public API, not on private helper names.

## Code style

- Python 3.11, `from __future__ import annotations`, type hints on public
  functions, dataclasses for structured data.
- Module loggers only: `logging.getLogger("amnesia.<module>")`; never
  `print()` in library code (CLI output goes through `rich`).
- Keep lines ≤ 100 characters. `ruff` is configured in `pyproject.toml` (it is
  a development tool, not a runtime dependency - `python -m pip install ruff`
  if it is not already available):

  ```powershell
  python -m ruff check amnesia tests
  python -m ruff format --check amnesia tests
  ```

- All timestamps are stored as UTC ISO-8601 strings; display them through
  `amnesia.utils.human_time` / `relative_time`.
- Anything persisted from users or models must pass through
  `amnesia.utils.redact_secrets` when `AMNESIA_REDACT` is enabled.
- No new third-party runtime dependencies without discussing it first (the
  project deliberately depends only on fastapi, uvicorn, httpx, typer, rich,
  numpy and python-dotenv; `mcp` is an optional extra).
- User-facing errors should be friendly and actionable; log internal detail at
  `DEBUG`/`WARNING` and never leak raw tracebacks for expected errors.

## Project layout

| Path | Contents |
|---|---|
| `amnesia/config.py` | Settings, env variables, precedence |
| `amnesia/db.py` | SQLite schema, FTS5 indexes, migrations |
| `amnesia/store.py` | Synchronous persistence and search primitives |
| `amnesia/models.py` | Dataclasses shared by every layer |
| `amnesia/embeddings.py` | Ollama embeddings + offline hash fallback |
| `amnesia/ollama.py` | Async Ollama HTTP client (streaming) |
| `amnesia/retrieval.py` | Hybrid RRF search and prompt-context builder |
| `amnesia/service.py` | Async `Amnesia` facade used by CLI/proxy/MCP |
| `amnesia/extraction.py` | LLM fact/event/summary distillation |
| `amnesia/proxy.py` | FastAPI memory proxy + dashboard host |
| `amnesia/cli.py` | Typer CLI (`amnesia` command) |
| `amnesia/mcp_server.py` | MCP stdio server |
| `docs/` | Architecture, integrations, schema, demo |
| `scripts/` | PowerShell operations helpers |

`docs/ARCHITECTURE.md` documents the module contracts and data flow; read it
before changing a public signature. Update the user docs when behaviour that
users see changes: [README.md](README.md), [docs/SCHEMA.md](docs/SCHEMA.md),
[docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) and [docs/DEMO.md](docs/DEMO.md).

## Pull-request checklist

1. Branch from `main` with a short descriptive name (`feat/recall-rerank`,
   `fix/session-gap`).
2. Keep changes focused; include tests for new behaviour and a regression test
   for every bug fix.
3. Run the full suite from the repository root:
   `$env:PYTHONPATH=(Get-Location).Path; python -m pytest -q`.
4. Run `python -m ruff check amnesia tests` and fix what you introduced.
5. Do not commit local data or secrets. The `.gitignore` already excludes
   `*.db`, `*.db-wal`, `*.db-shm`, `.amnesia/`, `.env`, virtualenvs and caches.
   Never paste an `AMNESIA_API_TOKEN` or other secret into code, docs, tests or
   an issue.
6. Update docs when the CLI, API routes, environment variables or schema change.
7. Describe **what** changed, **why**, and anything reviewers should test by
   hand (especially model-dependent behaviour).

## Reporting bugs

Include the Amnesia version (`amnesia --version`), OS, Python version, Ollama
version and installed model names (`ollama list`), the exact command that failed
and its output, and the relevant part of `amnesia doctor` / `amnesia config
show`. Never include memory contents or secrets you would not share publicly —
redact them first.

## License

Amnesia is released under the [MIT License](LICENSE). By contributing you agree
that your contributions are licensed under the same terms.
