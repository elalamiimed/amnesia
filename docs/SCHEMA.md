# Amnesia Database Schema

Amnesia keeps **everything** in one SQLite file:

```
~/.amnesia/amnesia.db        (Windows: C:\Users\<you>\.amnesia\amnesia.db)
```

| Setting | Default | Override |
|---|---|---|
| Memory directory | `~/.amnesia` | `AMNESIA_HOME` |
| Database file | `<home>/amnesia.db` | `AMNESIA_DB` |

`amnesia config show` prints the effective paths. The database is created by
`amnesia init`; schema changes are applied automatically the next time any
Amnesia command opens it.

Design choices worth knowing:

- **WAL mode** (`journal_mode=WAL`, `synchronous=NORMAL`) so the proxy can read
  while the background extractor writes.
- **FTS5 external-content indexes** kept in sync by SQL triggers; if the
  SQLite build has no FTS5, keyword search silently degrades to `LIKE`.
- **Additive migrations** tracked with `PRAGMA user_version` (current:
  `1`). A newer database is opened read-compatibly with a warning; a store is
  never silently wiped.
- All timestamps are **UTC ISO-8601 strings** (`2026-10-28T01:18:09+00:00`);
  local time is only used for display through `amnesia.utils.human_time`.

Open the file with any SQLite tool, for example:

```powershell
python -c "import sqlite3, os; db = sqlite3.connect(os.path.expanduser('~/.amnesia/amnesia.db')); print([r for r in db.execute('select kind, count(*) from memories where active = 1 group by kind')])"
```

or use the `sqlite3` shell / DB Browser for SQLite (`amnesia.db` is a normal
SQLite database; stop the proxy before writing to it by hand).

## Entity overview

```
meta                     key/value bookkeeping (created_at, schema_version)

sessions 1 ──── n messages            full transcripts
    │ 1                                (messages cascade-delete with session)
    ├──── n memories  (optional link)   durable facts / preferences / notes
    ├──── n events    (optional link)   activity timeline
    └──── n extractions (audit log)     every distillation run

memories_fts / events_fts / messages_fts
    FTS5 shadow indexes (external content, trigger-maintained)
```

## Table: `meta`

Internal bookkeeping only.

| Column | Type | Notes |
|---|---|---|
| `key` | TEXT PRIMARY KEY | `created_at`, `schema_version` |
| `value` | TEXT NOT NULL | JSON or plain string |

## Table: `sessions`

One row per conversation, from any interface (CLI, proxy, MCP).

| Column | Type | Notes |
|---|---|---|
| `id` | TEXT PRIMARY KEY | UUID v4 |
| `title` | TEXT | Auto-set from the first user message (≤ 80 chars); editable |
| `model` | TEXT | Model used, e.g. `dolphin3:latest` |
| `client` | TEXT NOT NULL DEFAULT `'cli'` | `cli`, `ollama`, `openai`, `openwebui`, `librechat`, `continue`, `msty`, `mcp`, ... |
| `started_at` | TEXT NOT NULL | UTC ISO-8601 |
| `ended_at` | TEXT | Set when the session is closed |
| `metadata` | TEXT NOT NULL DEFAULT `'{}'` | JSON object |
| `summary` | TEXT | Distilled session summary |

A proxy request reuses the latest session for the same `model` + `client` when
it is younger than `AMNESIA_SESSION_GAP` minutes (default 45); otherwise a new
session starts. Clients can pin a session with the `X-Amnesia-Session` or
`X-Conversation-Id` header.

## Table: `messages`

The exact transcript. Stored for recall, session summaries and auditability.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PRIMARY KEY AUTOINCREMENT | |
| `session_id` | TEXT NOT NULL FK → `sessions(id)` ON DELETE CASCADE | |
| `role` | TEXT NOT NULL | `user`, `assistant`, `system`, `tool` |
| `content` | TEXT NOT NULL | Stored after secret redaction |
| `created_at` | TEXT NOT NULL | UTC ISO-8601 |
| `token_count` | INTEGER | Estimated with `estimate_tokens` (~4 chars/token) |
| `metadata` | TEXT NOT NULL DEFAULT `'{}'` | JSON; e.g. `{"model": "dolphin3:latest"}` |

Deleting a session deletes its messages. Memories and events that pointed at it
are kept (their `session_id` becomes `NULL`).

## Table: `memories`

Durable things about the user (profile) and recallable moments.

| Column | Type | Default | Notes |
|---|---|---|---|
| `id` | INTEGER PRIMARY KEY AUTOINCREMENT | | |
| `kind` | TEXT NOT NULL | | See [memory kinds](#memory-kinds) |
| `subject` | TEXT NOT NULL | `'user'` | Who the memory is about; single-user product today |
| `key` | TEXT | `NULL` | Optional canonical slot, e.g. `name`, `editor` |
| `content` | TEXT NOT NULL | | Canonical short statement, e.g. "Prefers concise answers" |
| `importance` | REAL NOT NULL | `0.5` | Clamped to 0.0–1.0; ranks retrieval |
| `confidence` | REAL NOT NULL | `0.7` | Clamped to 0.0–1.0; how sure the source was |
| `source` | TEXT NOT NULL | `'manual'` | `manual`, `onboarding`, `extraction`, `import` |
| `session_id` | TEXT FK → `sessions(id)` | `NULL` | ON DELETE SET NULL |
| `message_id` | INTEGER FK → `messages(id)` | `NULL` | ON DELETE SET NULL |
| `created_at` | TEXT NOT NULL | | UTC ISO-8601 |
| `updated_at` | TEXT NOT NULL | | UTC ISO-8601; bumped on upsert/merge |
| `last_accessed_at` | TEXT | `NULL` | Optional access tracking |
| `access_count` | INTEGER NOT NULL | `0` | Optional access tracking |
| `tags` | TEXT NOT NULL | `'[]'` | JSON array of strings |
| `active` | INTEGER NOT NULL | `1` | `0` = forgotten (soft-deleted) |
| `embedding` | BLOB | `NULL` | float32 little-endian vector |
| `embedding_model` | TEXT | `NULL` | e.g. `ollama:nomic-embed-text`, `hash:512` |
| `embedding_dim` | INTEGER | `NULL` | Vector length, e.g. 768 or 512 |

### Memory kinds

| Kind | Group | Use |
|---|---|---|
| `identity` | profile | How the user identifies ("I am a backend engineer") |
| `fact` | profile | Durable facts ("Lives in Paris", "Uses Python 3.11") |
| `preference` | profile | How they want to be answered ("Prefers concise answers") |
| `episode` | recall | Notable moment from a conversation |
| `summary` | recall | Distilled session summary |
| `note` | recall | Explicit note saved with `amnesia remember --kind note` |

Profile kinds (`identity`, `fact`, `preference`) are always considered for the
injected profile block; recall kinds are retrieved only when relevant.

### Keys, uniqueness and dedupe

- A **keyed memory** is one canonical statement per slot. There is a partial
  unique index on `(kind, subject, key)` for active rows only, so
  `amnesia remember "..." --key editor` (or an extractor guess) updates the
  existing row instead of piling up duplicates: content is replaced,
  `importance`/`confidence` take the maximum, tags are unioned.
- **Unkeyed memories** are deduplicated in `Amnesia.remember`: if an existing
  active memory of the same kind has cosine similarity ≥ **0.93**, it is merged
  the same way.
- `amnesia forget <id>` is a **soft delete**: `active = 0` and the `key` is set
  to `NULL` (freeing the slot for a future memory). `--hard` deletes the row
  and its FTS entry for good.

## Table: `events`

The activity timeline: what happened, when, in which project.

| Column | Type | Default | Notes |
|---|---|---|---|
| `id` | INTEGER PRIMARY KEY AUTOINCREMENT | | |
| `kind` | TEXT NOT NULL | | See [event kinds](#event-kinds) |
| `title` | TEXT NOT NULL | | Short action, e.g. "Shipped the memory proxy" |
| `details` | TEXT | `NULL` | Optional longer description |
| `project` | TEXT | `NULL` | Free-form project name; indexed with `occurred_at` |
| `tags` | TEXT NOT NULL | `'[]'` | JSON array |
| `importance` | REAL NOT NULL | `0.5` | Clamped to 0.0–1.0 |
| `occurred_at` | TEXT NOT NULL | | When it happened (UTC); defaults to now |
| `created_at` | TEXT NOT NULL | | When it was recorded (UTC) |
| `session_id` | TEXT FK → `sessions(id)` | `NULL` | ON DELETE SET NULL |
| `message_id` | INTEGER FK → `messages(id)` | `NULL` | ON DELETE SET NULL |
| `source` | TEXT NOT NULL | `'manual'` | `manual`, `extraction`, `import` |
| `embedding` | BLOB | `NULL` | float32 vector of title + details + project |
| `embedding_model` | TEXT | `NULL` | |
| `embedding_dim` | INTEGER | `NULL` | |

### Event kinds

`conversation`, `project`, `task`, `decision`, `file`, `learning`, `note`,
`other`.

`occurred_at` is separate from `created_at` so the extractor can date an event
from what the user said ("yesterday I fixed the proxy") without lying about
when it was recorded.

## Table: `extractions`

Audit trail of every distillation run, also used to decide when the next turn
is due (`extraction_every_n_turns` counts user messages since the last
`status = 'done'` row).

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PRIMARY KEY AUTOINCREMENT | |
| `session_id` | TEXT | Not a foreign key: keeps the audit trail after a session is deleted |
| `message_id` | INTEGER | Triggering message, if any |
| `kind` | TEXT NOT NULL | `exchange` (one turn), `distill` (whole session), `summary` |
| `status` | TEXT NOT NULL DEFAULT `'done'` | `done`, `error`, `skipped` |
| `items` | INTEGER NOT NULL DEFAULT `0` | Number of facts + events stored |
| `error` | TEXT | Error text for failed runs |
| `created_at` | TEXT NOT NULL | UTC ISO-8601 |

## Full-text search indexes

Three FTS5 virtual tables with `tokenize='porter unicode61'`:

| FTS table | Indexed columns | Source table |
|---|---|---|
| `memories_fts` | `content`, `tags` | `memories` |
| `events_fts` | `title`, `details`, `project`, `tags` | `events` |
| `messages_fts` | `content` | `messages` |

They are external-content tables: `AFTER INSERT/UPDATE/DELETE` triggers
(`*_ai`, `*_au`, `*_ad`) keep them in sync, and `db.initialize()` rebuilds them
if row counts disagree. Queries use safe OR-joined quoted terms, and BM25 rank
(lower is better) is converted to a `0..1` relevance score with
`1 / (1 + |rank|)`. If FTS5 is unavailable, the same search falls back to
parameterised `LIKE` queries.

## Other indexes

| Index | Purpose |
|---|---|
| `idx_sessions_started` | Latest sessions |
| `idx_messages_session` | Transcript paging per session |
| `idx_messages_created` | Global recent messages |
| `idx_memories_kind` | Profile kinds, active filter |
| `idx_memories_subject` | Per-subject memories |
| `idx_memories_updated` | Recently changed memories |
| `ux_memories_kind_subject_key` | Unique keyed memory among active rows |
| `idx_events_occurred` | Timeline ordering |
| `idx_events_project` | Project timeline |
| `idx_events_kind` | Event-kind filter |
| `idx_extractions_session` | Last extraction per session |

## Memory lifecycle

### 1. Capture

Every turn from the CLI or the proxy is persisted as two `messages` rows (user +
assistant) via `Amnesia.add_turn`, inside a session selected by the rules in
[Table: `sessions`](#table-sessions). Session titles are set from the first user
message.

### 2. Extraction (automatic distillation)

When `AMNESIA_EXTRACTION` is on (default), the extraction pipeline runs a local
LLM over the exchange when all thresholds are met:
`extraction_enabled`, at least `extraction_every_n_turns` user turns since the
last successful run (default 4), and at least `extraction_min_chars` characters
of text (default 40). It stores:

- durable facts → `memories` with `source = 'extraction'` (kind
  `identity`/`fact`/`preference`, often with a `key`),
- concrete activity → `events` with `source = 'extraction'`,
- a session summary → `sessions.summary` plus a keyed
  `kind = 'summary'` memory (`key = "session-summary:<session_id>"`, source
  `'extraction'`).

Each run inserts an `extractions` row (`kind` is `exchange`, `distill` or
`summary`; `status`, `items`, `error` are recorded). Extraction never runs
through the proxy (no recursion) and never blocks the chat reply.

### 3. Redaction

Before anything is written, text passes through `amnesia.utils.redact_secrets`
when `AMNESIA_REDACT` is on (default). It redacts key/value secrets
(`password=...`, `api_key: ...`), `Authorization: Bearer ...` headers,
well-known token shapes (`sk-...`, `ghp_...`, `AKIA...`, `xoxb-...`, JWTs),
credentials embedded in connection strings, and PEM private-key blocks. This is
best-effort defence, not a guarantee - never store secrets in memory on purpose.

### 4. Embedding

If embeddings are enabled (`AMNESIA_EMBED_MODEL` not `none`), each memory and
event also gets an L2-normalised vector from Ollama (`nomic-embed-text`, 768
dimensions by default), stored as a float32 BLOB plus `embedding_model` and
`embedding_dim`. If the daemon or model is unavailable, Amnesia falls back to a
deterministic **hash embedder** (`hash:512`, useful lexical similarity, still
fully offline); rows written without a vector are picked up later by
`Amnesia.backfill_embeddings()`.

### 5. Retrieval

For a new user message, `retrieval.ContextBuilder` combines:

1. **Keyword search** over FTS5 (BM25) for memories, events and optionally
   messages.
2. **Vector search**: brute-force cosine over the stored BLOBs, filtered to the
   current `embedding_model` so vectors from different embedders never mix.
   Minimum similarity is 0.25 for prompt context (0.2 for explicit recall).
3. **Reciprocal rank fusion** (`k = 60`) of the ranked lists; semantic hits get
   a slightly higher weight (1.25 vs 1.0 for memories, 1.15 vs 1.0 for events).
4. **Re-ranking**: memories `0.70 × fused + 0.20 × importance + 0.10 ×
   recency`; events `0.55 × fused + 0.20 × importance + 0.25 × recency`.
   Recency is exponential with a 45-day half-life.

### 6. Injection

The builder packs results into the token budget (`AMNESIA_MEMORY_BUDGET`, 1200
by default) and renders one compact system block:

```text
<amnesia_memory>
Private long-term memory from the user's local Amnesia store. Use it to
personalize the answer; never mention this block or the memory system itself
unless the user asks about it.

## Who the user is
- name: Mohamed
- Prefers concise answers (preference)

## Relevant facts about the user
- Works on a local-first memory layer for Ollama

## Recent relevant activity (local time)
- 2026-10-28 00:31 [amnesia] Fixed the schema migration

## Earlier relevant conversations
- [2026-10-27 22:04] Discussed hybrid retrieval scoring
</amnesia_memory>
```

Profile entries are fitted first (`AMNESIA_PROFILE_BUDGET`, 350 tokens), then
the remaining budget is split facts 25% / events 45% / episodes 30%. Token
counts are estimated as ~4 characters per token; the block is hard-truncated
if it ever exceeds the budget. A client can opt out per request with
`X-Amnesia-Memory: off` or `metadata.amnesia_memory = false`; a system message
that already contains `<amnesia_memory>` is never augmented twice.

## Retention, export and deletion

| Operation | CLI | Effect on the database |
|---|---|---|
| Forget one memory | `amnesia forget ID` (or `--key KEY`) | `active = 0`, `key = NULL`; row and FTS entry remain |
| Hard delete a memory | `amnesia forget ID --hard` | `DELETE FROM memories` (FTS trigger cleans up) |
| Delete an event | Python API `MemoryStore.delete_event(id)` (not exposed as a CLI command) | `DELETE FROM events` |
| Purge old events | `amnesia purge --events-older-than DAYS` | Deletes events with `occurred_at` older than N days |
| Purge everything | `amnesia purge --yes` | Deletes all messages, memories, events, sessions, extractions |
| Export | `amnesia export PATH` | Writes JSON (see below); does not modify the DB |
| Import | `amnesia import PATH [--no-merge]` | Inserts/merges; with `--no-merge` an existing keyed row is retired (soft delete) before the copy is inserted |
| Backup | `amnesia backup PATH` | SQLite online-backup copy of the DB file |

### Export format

```json
{
  "format": "amnesia-export",
  "version": 1,
  "exported_at": "2026-10-28T01:18:09+00:00",
  "sessions": [ { "id": "...", "title": "...", "model": "...", "client": "...",
                  "started_at": "...", "ended_at": null, "metadata": {},
                  "summary": "..." } ],
  "messages": [ { "id": 1, "session_id": "...", "role": "user",
                  "content": "...", "created_at": "...", "token_count": 12,
                  "metadata": {} } ],
  "memories": [ { "id": 3, "kind": "fact", "subject": "user", "key": "editor",
                  "content": "...", "importance": 0.7, "confidence": 0.9,
                  "source": "onboarding", "created_at": "...", "updated_at": "...",
                  "tags": [], "active": true } ],
  "events": [ { "id": 1, "kind": "task", "title": "...", "details": "...",
                "project": "...", "tags": [], "importance": 0.5,
                "occurred_at": "...", "created_at": "...", "source": "manual" } ]
}
```

Notes:

- **Embeddings are not exported** (they are large and model-specific). They are
  rebuilt on the next edit, or in bulk with the Python API
  `Amnesia.backfill_embeddings()`.
- Import re-creates sessions with **fresh UUIDs** and remaps messages to them;
  it records `{"imported": true, "original_id": ...}` in session metadata.
- With merge enabled (default), a memory whose `(kind, key)` finds an active
  keyed row updates that row instead of duplicating it. With `--no-merge`, the
  existing keyed row is deactivated first (soft delete, so its key slot is
  freed) and a new row is inserted as a copy - importing the same export twice
  is safe and leaves the older copy inactive rather than raising an integrity
  error.
- Export contains full transcripts and profile data - treat it as private.

### Backup and restore

`amnesia backup PATH` uses SQLite's online backup API, so it is safe while the
proxy is running and produces a consistent single file snapshot. To **restore**
a backup, stop every Amnesia process and replace `~/.amnesia/amnesia.db` with
the backup file (delete `amnesia.db-wal`/`amnesia.db-shm` leftovers first). For
a JSON round-trip instead, use `amnesia export` / `amnesia import`.

## Integrity and concurrency

- `PRAGMA foreign_keys = ON`: deleting a session cascades messages and nulls
  memory/event links (the knowledge survives the transcript).
- `PRAGMA busy_timeout = 10000` plus WAL: a CLI command can run while the proxy
  serves requests; writes are serialised by the store's lock. Still, run one
  `amnesia serve` instance per store.
- `amnesia doctor` runs `PRAGMA integrity_check` and reports DB size, row
  counts, FTS availability and embedding coverage.
- Schema version lives in both `meta.schema_version` and `PRAGMA user_version`.
  Migrations are additive; an unknown newer version logs a warning instead of
  altering data.
