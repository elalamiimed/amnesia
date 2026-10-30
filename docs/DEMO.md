# Five-minute demo

A scripted walkthrough that shows the whole product: onboarding, explicit
memory, chat with no cross-process context, activity timeline, the HTTP proxy,
the dashboard and export. Everything runs against local Ollama - no network, no
account.

**Audience:** someone who has the repo and Ollama installed.
**Show time:** ~5 minutes of talking.
**OS:** Windows PowerShell commands are shown; POSIX equivalents are included
where they differ.

## 0. Pre-flight (before the clock starts)

```powershell
ollama list                      # expect a chat model + nomic-embed-text
amnesia doctor                   # DB, Ollama, embeddings: all green
```

If Amnesia is not set up yet: `.\scripts\setup.ps1` (or
`pip install -e ".[dev,mcp]"` followed by `amnesia init`).

**Optional: a clean demo store.** Do the demo in a throwaway memory home so
your real memory stays untouched (Windows PowerShell):

```powershell
$env:AMNESIA_HOME = "$env:TEMP\amnesia-demo"
Remove-Item -Recurse -Force "$env:AMNESIA_HOME" -ErrorAction SilentlyContinue
```

POSIX: `export AMNESIA_HOME="/tmp/amnesia-demo"; rm -rf "$AMNESIA_HOME"`.

Keep this terminal's environment for every step below (or prefix each command
with `$env:AMNESIA_HOME=...` / `AMNESIA_HOME=...`).

> 0:00 - "Local models start every conversation as a stranger. Amnesia gives
> them a durable, private memory: your profile, your activity, and what you
> talked about - in one SQLite file on this machine."

## 1. Initialize and onboard (0:15)

```powershell
amnesia init
amnesia remember "My name is Sam Rivera" --kind identity --key name --importance 0.9
amnesia remember "I am a backend engineer building a local-first app" --kind fact --key role --importance 0.8
amnesia remember "I prefer concise answers with code examples" --kind preference --key answer_style --importance 0.8
amnesia remember "My main project is Amnesia, a local memory layer for Ollama" --kind fact --key project --importance 0.8
```

Alternative for real users: `amnesia init --interactive` asks for name, role,
location, languages, projects and communication preferences, and stores each
answer as a memory (`source="onboarding"`).

```powershell
amnesia profile
```

- "Those are **identity memories** - durable statements about me, stored in
  `memories` with kinds `identity`, `fact` and `preference`. They're always
  eligible for the profile block injected into the model."

## 2. Give it something to remember (1:00)

```powershell
amnesia note "Release 0.1 needs docs, ops scripts and a green test suite" --kind task --project amnesia
amnesia recall "release checklist"
```

- "`note` writes an **activity event** to the timeline; `recall` runs hybrid
  keyword + embedding search. This is the same retrieval the model gets."

## 3. Chat session 1 (1:30)

```powershell
amnesia chat dolphin3:latest
```

Prompt 1:

```
Based on what you remember about me, what should I focus on next for Amnesia?
```

- "The model got a `<amnesia_memory>` block behind the scenes: my name, role,
  preferences and project. It answers like someone who knows me."

Prompt 2 (state a new durable fact + decision):

```
Decision: the release branch needs a changelog generated from the docs.
```

Then distill the session explicitly so the demo does not depend on the
background threshold:

```
/quit
amnesia distill --all
```

- "Extraction runs a local LLM over the session: durable facts become
  `memories`, concrete actions become `events`, plus a session summary.
  `amnesia distill` forces a run immediately."

## 4. A completely new process with memory (2:45)

```powershell
amnesia chat dolphin3:latest
```

```
What do you know about me and what am I building?
```

- "New process, new terminal, **no context in RAM** - and it still knows my
  name, my role, my project and the decision I just made. That is the whole
  point of Amnesia."

Optionally show the retrieval inside the REPL:

```
/recall release
/events
/quit
```

## 5. Timeline and sessions (3:30)

```powershell
amnesia timeline --days 7
amnesia sessions
```

- "Every chat is a session; every decision/task can be an event. The timeline
  is the second memory layer: what I *do*, not just who I *am*."

## 6. Proxy and dashboard (4:00)

In a second terminal:

```powershell
amnesia serve
```

Open <http://127.0.0.1:11435/amnesia/ui>:

- **Profile** tab: the identity memories.
- **Timeline** tab: the activity event from step 2 and extracted events.
- **Sessions** tab: both chat sessions with messages.
- **Search** tab: type `release` - hybrid recall.
- **Health** tab: DB path/size/integrity, Ollama and embedding status.

Prove the OpenAI-compatible surface with one request:

```powershell
$body = @{
  model    = "dolphin3:latest"
  messages = @(@{ role = "user"; content = "What is my name?" })
  stream   = $false
} | ConvertTo-Json -Depth 5
(Invoke-RestMethod -Uri http://127.0.0.1:11435/v1/chat/completions `
  -Method Post -ContentType "application/json" -Body $body).choices[0].message.content
```

```bash
curl -s http://127.0.0.1:11435/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dolphin3:latest","messages":[{"role":"user","content":"What is my name?"}]}'
```

- "Any OpenAI-compatible client (Open WebUI, LibreChat, Continue, a script)
  gets memory by changing one base URL to `http://127.0.0.1:11435/v1`.
  See `docs/INTEGRATIONS.md`."

Stop the proxy with `Ctrl+C` when done.

## 7. Export and control (4:45)

```powershell
amnesia export demo-export.json
amnesia forget --key answer_style
amnesia profile
```

- "Everything is yours: export the whole store to JSON, forget a memory
  (soft delete; `--key` searches every kind unless you pass `--kind`), or
  `amnesia purge --yes` to wipe it. The database is a single SQLite file at
  `~/.amnesia/amnesia.db`."

Clean up the throwaway store (optional):

```powershell
Remove-Item -Recurse -Force "$env:AMNESIA_HOME"
Remove-Item -Force demo-export.json
```

## Closing line

> "Amnesia is not a bigger context window. It is a memory that survives the
> process: identity, activity, hybrid recall, automatic distillation - all
> local, all in one SQLite file."

## Backup plan for a slow CPU

If the machine is slow (7B models on CPU can take a while):

- Use `dolphin3:latest` (smaller) and keep prompts to one line.
- Replace step 3's chat with:

  ```powershell
  amnesia ask "What should I focus on next for Amnesia?" --model dolphin3:latest
  ```

- Skip step 6's live chat request and show `/amnesia/health` +
  `/amnesia/profile` instead - they are instant:

  ```powershell
  Invoke-RestMethod http://127.0.0.1:11435/amnesia/health | ConvertTo-Json -Depth 4
  ```

## What each step proves

| Step | Capability |
|---|---|
| 1 | Onboarding / identity memory (`memories`) |
| 2 | Activity timeline (`events`) + hybrid recall |
| 3 | Memory injection, streaming chat, transcript capture, distillation |
| 4 | Memory persists **across processes** |
| 5 | Timeline + session history (`sessions`, `messages`) |
| 6 | Proxy: Ollama-native and OpenAI-compatible APIs, dashboard |
| 7 | Export, forget, local ownership |
