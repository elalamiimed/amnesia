# Amnesia

> Your local models forgot. **Amnesia** remembers.

Amnesia is a **local-first long-term memory layer for Ollama models**. It remembers who you are, what you are working on, and everything you have done with your local LLMs — and injects the relevant parts back into every conversation. No cloud, no telemetry, no accounts: one SQLite file on your disk.

```
        +-------------------+        +-------------------+        +----------------------+
        |  Ollama / OpenAI  |  --->  |      Amnesia      |  --->  |   Ollama (11434)     |
        |  compatible app   |        |   memory proxy    |        |   local models       |
        +-------------------+        +---------+---------+        +----------------------+
                                               |
                                     +---------v----------+
                                     |  SQLite + FTS5 +   |
                                     |  vector embeddings |
                                     |  profile / events  |
                                     +--------------------+
```

## Why

Local models start every conversation as a stranger. Amnesia gives them a durable, private memory:

- **Identity memory** — your name, role, projects, preferences, the way you like to be answered.
- **Activity memory** — a searchable timeline of what you did: sessions, decisions, tasks, projects, files.
- **Semantic recall** — hybrid keyword + embedding search (via `nomic-embed-text`) over everything that mattered.
- **Zero cloud** — runs against your own Ollama instance; the memory store never leaves your machine.

## Quickstart

```powershell
# 1. Install (from the repo root)
pip install -e ".[dev,mcp]"

# 2. Set up the local store (~/.amnesia), verify Ollama, seed your profile
amnesia init --interactive

# 3. Chat with memory (any of your Ollama models)
amnesia chat qwen2.5-coder-7b-instruct-heretic:q4_k_m

# 4. Or run the memory proxy and point any OpenAI-compatible client at it
amnesia serve            # http://127.0.0.1:11435
```

## Features

- **Memory-augmented CLI chat** with streaming, slash commands, and automatic fact/activity extraction.
- **Drop-in HTTP proxy** speaking both Ollama-native (`/api/chat`) and OpenAI-compatible (`/v1/chat/completions`) APIs, so existing apps gain memory by changing one base URL.
- **MCP server** exposing `memory_search`, `memory_remember`, `memory_profile`, `memory_log_event`, `memory_timeline` to MCP clients.
- **Web dashboard** at `/amnesia/ui` to browse your profile, timeline, sessions, and search results.
- **Hybrid retrieval** — FTS5 BM25 + local embeddings, fused with reciprocal-rank fusion, budgeted into the model's context window.
- **Automatic distillation** — local LLM extracts durable facts and activity events from conversations, dedupes/merges them, and summarizes sessions.
- **Full control** — `remember`, `forget`, `recall`, `timeline`, `export`, `import`, and a documented SQLite schema.

## Memory model

| Layer | Table | What it holds |
|---|---|---|
| Identity | `memories` where `kind in (fact, identity, preference)` | Stable facts about you (`subject=user`) |
| Activity | `events` | What you did, when, in which project |
| Episodes | `memories` where `kind in (summary, episode, note)` | Session summaries and notable moments |
| Transcripts | `sessions`, `messages` | Full conversation history |
| Index | `*_fts`, embedding BLOBs | Keyword + semantic retrieval |

## Privacy

Everything is stored locally in `~/.amnesia/amnesia.db` (override with `AMNESIA_HOME`). Nothing is uploaded. Secret-shaped strings (API keys, passwords, tokens) are redacted before storage by default. Use `amnesia export`, `amnesia import`, `amnesia forget`, and `amnesia purge` to control your data.

## Documentation

- `docs/ARCHITECTURE.md` — how the pieces fit together
- `docs/INTEGRATIONS.md` — Open WebUI, LM Studio, LibreChat, Continue, `ollama run`, MCP clients
- `docs/SCHEMA.md` — database schema and memory lifecycle

## Status

Under active development. Built for a single-user, local workstation.

## License

MIT
