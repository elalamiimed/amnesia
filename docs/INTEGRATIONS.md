# Integrations

Amnesia is a **drop-in memory proxy** in front of your local Ollama daemon. Any
client that speaks the Ollama API or the OpenAI Chat Completions API can gain
long-term memory by changing one URL - no plugins, no code.

## Two ports, one machine

| Port | Process | What talks to it |
|---|---|---|
| **11434** | Ollama daemon | Amnesia itself (upstream); clients that want *no* memory |
| **11435** | Amnesia proxy (`amnesia serve`) | Your chat clients - they get memory |

- The **Ollama daemon never moves**; it stays on `127.0.0.1:11434`.
- The **proxy** listens on `127.0.0.1:11435` by default
  (`AMNESIA_PROXY_HOST` / `AMNESIA_PROXY_PORT`).
- Amnesia reads the upstream address from `AMNESIA_OLLAMA_URL`, default
  `http://127.0.0.1:11434`.

Keep the daemon running (`ollama serve`, or the Ollama desktop app) and start
the proxy in another terminal:

```powershell
.\scripts\serve.ps1            # or: amnesia serve
```

```bash
amnesia serve
```

Then:

| URL | Purpose |
|---|---|
| `http://127.0.0.1:11435/api/chat` | Ollama-native chat with memory |
| `http://127.0.0.1:11435/v1/chat/completions` | OpenAI-compatible chat with memory |
| `http://127.0.0.1:11435/v1/models` | OpenAI model list (from Ollama tags) |
| `http://127.0.0.1:11435/amnesia/ui` | Web dashboard |
| `http://127.0.0.1:11435/amnesia/health` | Health / status JSON |

### What the proxy adds to every request

1. **Memory injection** - relevant profile facts, events and past episodes are
   added as an `<amnesia_memory>` system block.
2. **Transcript capture** - the user/assistant exchange is stored in a session
   (resumed from `X-Amnesia-Session` / `X-Conversation-Id`, or reusing the most
   recent session for the same model/client when it is younger than
   `AMNESIA_SESSION_GAP` minutes).
3. **Automatic distillation** - a local LLM extracts durable facts and activity
   events from the conversation in the background when thresholds are met.
4. **Activity log** - chats become part of the timeline and the dashboard.

Raw Ollama on port 11434 does none of this - which is handy for A/B testing:
same model, same prompt, with and without Amnesia.

You can opt out per request with the header `X-Amnesia-Memory: off` (or
`"metadata": {"amnesia_memory": false}` in an Ollama request body). For
debugging, chat responses also carry `X-Amnesia-Session` (the resolved session
id), `X-Amnesia-Memory-Injected` (`1`/`0`) and `X-Amnesia-Memory-Chars`.

---

## 1. `ollama run` via `OLLAMA_HOST`

The Ollama CLI talks to whatever `OLLAMA_HOST` points at. Point the **client**
at the proxy and every `ollama run` prompt gets memory injected:

**Windows PowerShell**

```powershell
# Terminal 1 - the daemon stays on 11434 (skip if the Ollama app is running)
ollama serve

# Terminal 2 - the CLI now talks to the Amnesia proxy
$env:OLLAMA_HOST = "127.0.0.1:11435"
ollama run dolphin3:latest
```

**macOS / Linux**

```bash
ollama serve &                      # or use the Ollama app
OLLAMA_HOST=127.0.0.1:11435 ollama run dolphin3:latest
```

Notes:

- `OLLAMA_HOST` is a **client-side environment variable** here. The daemon
  itself must keep listening on `127.0.0.1:11434` - Amnesia connects to it.
- `ollama list` / `ollama ps` work through the proxy (passthrough), so the
  session behaves normally. The simplest way to pull a new model is to run
  `ollama pull <model>` in a terminal **without** `OLLAMA_HOST` set (i.e.
  against port 11434).
- Streaming works: the proxy forwards Ollama's NDJSON stream unbuffered.
- Model names must be the full Ollama tags (`dolphin3:latest`,
  `R4C3R/qwen2.5-coder-7b-instruct-heretic:q4_k_m`), exactly as `ollama list`
  shows them.
- Each CLI client shows up as `client="ollama"` in the session list.

---

## 2. Open WebUI (Docker)

Open WebUI speaks the OpenAI API. Add Amnesia as an OpenAI-compatible
connection:

1. Make sure Open WebUI and the proxy are running. If Open WebUI is in Docker,
   start it with the usual command, then add the connection in the UI (no
   special env vars required):

   ```powershell
   docker run -d --name open-webui -p 3000:8080 `
     -v open-webui:/app/backend/data --restart unless-stopped `
     ghcr.io/open-webui/open-webui:main
   ```

2. Open <http://localhost:3000> and sign in as admin.
3. Go to **Admin Panel → Settings → Connections → OpenAI API**, click **+**
   and fill in:
   - **Base URL**: `http://host.docker.internal:11435/v1`
   - **API Key**: any non-empty value (for example `amnesia-local`). Use your
     `AMNESIA_API_TOKEN` if you set one. The proxy ignores unknown bearer
     tokens on chat routes but protects `/amnesia/*`.
   - **Model**: fetch the list (or leave it to auto-discover `/v1/models`).
4. Save. The Ollama models now appear in the model selector, served through
   Amnesia - so conversations are remembered and logged.

Notes:

- `host.docker.internal` is the host machine as seen from a Docker container.
  Docker Desktop provides it automatically on Windows and macOS. On Linux
  Docker Engine, start the container with
  `--add-host=host.docker.internal:host-gateway`.
- If Open WebUI runs **natively** (not in Docker), use
  `http://127.0.0.1:11435/v1` instead.
- If the **proxy** runs in Docker (see `docker-compose.yml`) and Open WebUI is
  also in Docker, put both on the same Docker network and use the service name
  (`http://amnesia-proxy:11435/v1`) or keep the proxy's published port
  `http://host.docker.internal:11435/v1`.
- Open WebUI streams with SSE; the proxy supports it.

---

## 3. LibreChat

LibreChat supports custom OpenAI-compatible endpoints through `librechat.yaml`.
Mount the file into the container (`CONFIG_PATH=/app/librechat.yaml`).

```yaml
version: 1.1.5
cache: true
endpoints:
  custom:
    - name: "Amnesia (Ollama)"
      apiKey: "amnesia-local"            # or your AMNESIA_API_TOKEN
      baseURL: "http://host.docker.internal:11435/v1"
      models:
        default:
          - "dolphin3:latest"
          - "R4C3R/qwen2.5-coder-7b-instruct-heretic:q4_k_m"
        fetch: true                      # discover models from /v1/models
      titleConvo: true
      modelDisplayLabel: "Amnesia"
```

Restart LibreChat, pick the "Amnesia (Ollama)" endpoint in the model dropdown,
and chat. Conversations go through the proxy, so they are persisted and
distilled like CLI chats.

- Linux Docker Engine: add
  `extra_hosts: ["host.docker.internal:host-gateway"]` to the LibreChat
  service.
- The `models.default` list must use the exact Ollama tags.
- If `AMNESIA_API_TOKEN` is set, it protects the dashboard/memory API only -
  LibreChat's key can be any string for chat.

---

## 4. Continue (VS Code / JetBrains) and other OpenAI clients

Continue is an IDE assistant; configure it as an OpenAI-compatible provider.
Continue may store its config as JSON (older) or YAML (newer); both forms are
shown.

**`~/.continue/config.json`**

```json
{
  "models": [
    {
      "title": "Amnesia (dolphin3)",
      "provider": "openai",
      "model": "dolphin3:latest",
      "apiBase": "http://127.0.0.1:11435/v1",
      "apiKey": "amnesia-local"
    },
    {
      "title": "Amnesia (qwen coder)",
      "provider": "openai",
      "model": "R4C3R/qwen2.5-coder-7b-instruct-heretic:q4_k_m",
      "apiBase": "http://127.0.0.1:11435/v1",
      "apiKey": "amnesia-local"
    }
  ]
}
```

**`~/.continue/config.yaml`** (newer Continue)

```yaml
name: amnesia-local
version: 0.0.1
models:
  - name: Amnesia (dolphin3)
    provider: openai
    model: dolphin3:latest
    apiBase: http://127.0.0.1:11435/v1
    apiKey: amnesia-local
```

Notes:

- Use your token as `apiKey` if `AMNESIA_API_TOKEN` is set. Chat routes accept
  any value, but keeping one value everywhere avoids confusion.
- Continue uses the OpenAI Chat Completions API, which the proxy implements,
  including streaming. If a Continue version cannot stream from a local
  endpoint, set the model's `streaming` option if available, or use non-streaming
  mode.
- Any other OpenAI-compatible tool (SillyTavern, LobeChat, Obsidian plugins,
  custom scripts) is configured the same way: base URL
  `http://127.0.0.1:11435/v1`, any API key, the Ollama model tag as model name.

---

## 5. MCP clients (Claude Desktop, Cursor, ...)

Amnesia ships an MCP stdio server (`amnesia mcp`) exposing memory tools:
`memory_search`, `memory_remember`, `memory_profile`, `memory_log_event`,
`memory_timeline`, `memory_stats`.

Requirements: install the MCP extra - `python -m pip install -e ".[mcp]"` (or
`pip install "amnesia[mcp]"`).

### Claude Desktop

Edit `claude_desktop_config.json`:

- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "amnesia": {
      "command": "C:\\Users\\<you>\\.venv\\Scripts\\amnesia.exe",
      "args": ["mcp"],
      "env": {
        "AMNESIA_HOME": "C:\\Users\\<you>\\.amnesia"
      }
    }
  }
}
```

### Cursor

Global `~/.cursor/mcp.json` (or project `.cursor/mcp.json`) with the same
`mcpServers` object:

```json
{
  "mcpServers": {
    "amnesia": {
      "command": "C:\\Users\\<you>\\.venv\\Scripts\\amnesia.exe",
      "args": ["mcp"],
      "env": {
        "AMNESIA_HOME": "C:\\Users\\<you>\\.amnesia"
      }
    }
  }
}
```

Notes:

- Use an **absolute path** to the launcher: GUI applications often do not
  inherit your shell `PATH`. If `amnesia` is installed globally and available
  on `PATH`, `"command": "amnesia"` is enough. On macOS/Linux the path looks
  like `/Users/<you>/.venv/bin/amnesia` or `/home/<you>/.venv/bin/amnesia`.
- Without the console script (running from source), use
  `"command": "python"` and `"args": ["-m", "amnesia", "mcp"]` with
  `"PYTHONPATH": "C:\\path\\to\\amnesia"` in the `env` object.
- `AMNESIA_HOME` must point at the same store the CLI/proxy use, otherwise the
  MCP tools see an empty memory. Add any other `AMNESIA_*` variables you use
  (for example `AMNESIA_MODEL`) in the same `env` object.
- Restart the client after editing the config; the tools then appear in the
  client's tool picker and can be called by the model.
- The MCP server is local stdio: it reads and writes your SQLite store through
  the same code as the CLI. It does not open a network port.

---

## 6. Plain `curl` / HTTP

All examples assume `amnesia serve` is running on `127.0.0.1:11435`.

### Health and stats

```bash
curl -s http://127.0.0.1:11435/amnesia/health
curl -s http://127.0.0.1:11435/amnesia/stats
```

```powershell
Invoke-RestMethod http://127.0.0.1:11435/amnesia/health | ConvertTo-Json -Depth 5
Invoke-RestMethod http://127.0.0.1:11435/amnesia/stats | ConvertTo-Json -Depth 5
```

`health` reports the DB path/size/integrity, Ollama availability, embedding
backend, row counts and warnings. It is always reachable, even with an API
token set.

### Ollama-native chat (non-streaming)

```bash
curl -s http://127.0.0.1:11435/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dolphin3:latest",
    "messages": [{"role": "user", "content": "What do you remember about me?"}],
    "stream": false
  }'
```

```powershell
$body = @{
  model    = "dolphin3:latest"
  stream   = $false
  messages = @(@{ role = "user"; content = "What do you remember about me?" })
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:11435/api/chat -Method Post `
  -ContentType "application/json" -Body $body
```

### Ollama-native chat (streaming)

```bash
curl -N http://127.0.0.1:11435/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dolphin3:latest",
    "messages": [{"role": "user", "content": "Summarize what we did this week."}],
    "stream": true
  }'
```

Each line is one NDJSON chunk (`message.content`, then `done: true`).

### OpenAI-compatible chat

```bash
curl -s http://127.0.0.1:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dolphin3:latest",
    "messages": [{"role": "user", "content": "What am I working on?"}],
    "stream": false
  }'
```

```powershell
$body = @{
  model    = "dolphin3:latest"
  messages = @(@{ role = "user"; content = "What am I working on?" })
  stream   = $false
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:11435/v1/chat/completions -Method Post `
  -ContentType "application/json" -Body $body
```

### OpenAI model list

```bash
curl -s http://127.0.0.1:11435/v1/models
```

### Memory API

```bash
# Remember a fact (upsert by key when it already exists)
curl -s -X POST http://127.0.0.1:11435/amnesia/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "Prefers concise answers with code examples", "kind": "preference", "key": "answer_style", "importance": 0.8}'

# Who is the user?
curl -s "http://127.0.0.1:11435/amnesia/profile?limit=50"

# Hybrid recall over memories + events
curl -s "http://127.0.0.1:11435/amnesia/recall?q=proxy%20design&limit=10"

# Activity timeline
curl -s "http://127.0.0.1:11435/amnesia/events?days=7&limit=50"

# Log an activity event
curl -s -X POST http://127.0.0.1:11435/amnesia/events \
  -H "Content-Type: application/json" \
  -d '{"title": "Reviewed the retrieval reranker", "kind": "task", "project": "amnesia", "importance": 0.6}'

# Sessions and a full export
curl -s "http://127.0.0.1:11435/amnesia/sessions?limit=20"
curl -s http://127.0.0.1:11435/amnesia/export -o amnesia-export.json
```

If `AMNESIA_API_TOKEN` is set, add the bearer token to the memory API calls:

```bash
curl -s http://127.0.0.1:11435/amnesia/profile \
  -H "Authorization: Bearer $AMNESIA_API_TOKEN"
```

```powershell
$headers = @{ Authorization = "Bearer $env:AMNESIA_API_TOKEN" }
Invoke-RestMethod -Uri http://127.0.0.1:11435/amnesia/profile -Headers $headers
```

### Dashboard

Open <http://127.0.0.1:11435/amnesia/ui> (or `/` which redirects there). Tabs:
**Profile**, **Timeline**, **Sessions**, **Search**, **Health**, with stats
cards for memories, events, sessions, DB size and embedding status.

---

## Which integration should I use?

| You want to... | Use |
|---|---|
| Chat in a terminal with memory | `amnesia chat` / `.\scripts\chat.ps1`, or `ollama run` with `OLLAMA_HOST` |
| Use an existing chat UI | Open WebUI / LibreChat via the OpenAI-compatible base URL |
| Use an IDE assistant | Continue (or any OpenAI-compatible IDE plugin) |
| Let a coding agent use your memory | MCP server (`amnesia mcp`) |
| Build your own client | OpenAI `/v1/chat/completions` or Ollama `/api/chat` |
| Inspect or curate memory | Dashboard, `amnesia profile/recall/timeline/remember/forget` |

## Security notes

- Amnesia is designed for a **single user on a trusted machine**. It binds to
  `127.0.0.1` by default; do not expose port 11435 to an untrusted network.
- `AMNESIA_API_TOKEN` protects `/amnesia/*` (the dashboard page and
  `/amnesia/static/*` stay loadable, but their API calls need the token).
  Accepted forms: `Authorization: Bearer <token>`, `X-Amnesia-Token: <token>`,
  or `?token=<token>` on the URL. Passthrough chat routes stay open by design,
  so a token is not a full multi-user auth system. If you bind to `0.0.0.0`,
  put the proxy behind a reverse proxy/VPN and restrict access.
- The proxy returns HTTP **502** with a JSON error when Ollama is unreachable.
  Start the daemon and retry; nothing is lost.
- No integration in this document sends your memory to the cloud. Amnesia talks
  only to your local Ollama daemon and writes to your local SQLite file.
