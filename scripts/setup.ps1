#Requires -Version 5.1
<#
.SYNOPSIS
    One-command setup for Amnesia on Windows: virtual environment, editable
    install, Ollama checks and the local memory store.

.DESCRIPTION
    Idempotent - re-running only fills in whatever is missing. Creates:
      * <repo>\.venv with Amnesia installed in editable mode (when pip can
        reach a package index; otherwise falls back to the already-installed
        system Python),
      * <repo>\.env copied from .env.example (if missing),
      * the local memory store in %USERPROFILE%\.amnesia (via `amnesia init`).

.PARAMETER Interactive
    Run the onboarding wizard (`amnesia init --interactive`) and store your
    name, role, projects and preferences as identity memories.

.PARAMETER PullEmbedModel
    Pull the embedding model (`nomic-embed-text` by default) with Ollama when
    it is not installed yet. Without this switch a hint is printed instead.

.PARAMETER SkipInstall
    Do not create/update the virtual environment or install packages.

.PARAMETER SkipInit
    Do not initialize the memory store.

.PARAMETER SystemPackages
    Create the virtual environment with `--system-site-packages` (useful on
    offline machines where dependencies are already installed globally).

.PARAMETER EmbedModel
    Embedding model name to check/pull. Default: nomic-embed-text.

.PARAMETER Python
    Python executable used to create the virtual environment. Default: python.

.EXAMPLE
    .\scripts\setup.ps1

.EXAMPLE
    .\scripts\setup.ps1 -Interactive -PullEmbedModel
#>
[CmdletBinding()]
param(
    [switch]$Interactive,
    [switch]$PullEmbedModel,
    [switch]$SkipInstall,
    [switch]$SkipInit,
    [switch]$SystemPackages,
    [string]$EmbedModel = "nomic-embed-text",
    [string]$Python = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "_common.ps1")

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([string]$Message)
    Write-Host "    [ok] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "    [!] $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "    $Message"
}

Write-Host ""
Write-Host "Amnesia setup" -ForegroundColor Magenta
Write-Host "Repository: $script:RepoRoot"

# ---------------------------------------------------------------------------
# 1. Python
# ---------------------------------------------------------------------------
Write-Step "Checking Python"
$pythonCommand = Get-Command $Python -ErrorAction SilentlyContinue
if (-not $pythonCommand -or -not $pythonCommand.Source) {
    throw "Python executable '$Python' was not found on PATH. Install Python 3.10+ from https://www.python.org/downloads/ and re-run."
}
$pythonPath = $pythonCommand.Source
$versionText = (& $pythonPath --version 2>&1 | Out-String).Trim()
if ($versionText -notmatch "(\d+)\.(\d+)") {
    throw "Could not determine the Python version from '$versionText'."
}
$majorVersion = [int]$Matches[1]
$minorVersion = [int]$Matches[2]
if ($majorVersion -lt 3 -or ($majorVersion -eq 3 -and $minorVersion -lt 10)) {
    throw "Amnesia requires Python 3.10 or newer; found $versionText at $pythonPath."
}
Write-Ok "$versionText at $pythonPath"

# ---------------------------------------------------------------------------
# 2. Virtual environment + install
# ---------------------------------------------------------------------------
Write-Step "Preparing the virtual environment"
if (Test-Path -LiteralPath $script:VenvPython) {
    Write-Ok "Using the existing virtual environment: $script:VenvDir"
} elseif ($SkipInstall) {
    Write-Warn "Virtual environment does not exist and -SkipInstall was given"
} else {
    $venvArguments = @("-m", "venv")
    if ($SystemPackages) {
        $venvArguments += "--system-site-packages"
    }
    $venvArguments += $script:VenvDir
    & $pythonPath @venvArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the virtual environment at $script:VenvDir (exit code $LASTEXITCODE)."
    }
    Write-Ok "Created virtual environment: $script:VenvDir"
}

$venvReady = $false
if (-not $SkipInstall) {
    Write-Step "Installing Amnesia with the dev + MCP extras"
    & $script:VenvPython -m pip install --upgrade pip --disable-pip-version-check --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Could not upgrade pip (offline?); continuing with the bundled version."
    }

    & $script:VenvPython -m pip install -e ".[dev,mcp]" --disable-pip-version-check
    if ($LASTEXITCODE -eq 0) {
        Write-Ok "Installed amnesia[dev,mcp] in editable mode"
    } else {
        Write-Warn "Editable install failed (no access to a package index?)."
        Write-Info "Retrying without build isolation (uses installed build tools)..."
        & $script:VenvPython -m pip install -e ".[dev,mcp]" --no-build-isolation --disable-pip-version-check
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Installed amnesia[dev,mcp] without build isolation"
        }
    }

    & $script:VenvPython -c "import amnesia" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $venvReady = $true
    }
}

if (-not $venvReady) {
    Write-Step "Checking the fallback (system Python)"
    & $pythonPath -c "import fastapi, httpx, typer, rich, numpy, uvicorn, dotenv" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Warn "The virtual environment is not usable, but the system Python has the dependencies."
        Write-Info "The scripts will run Amnesia from source with: `$env:PYTHONPATH = `"$script:RepoRoot`""
    } else {
        throw "Amnesia could not be installed and the system Python is missing dependencies. Connect to the network and run: python -m pip install -e `".[dev,mcp]`""
    }
}

# ---------------------------------------------------------------------------
# 3. Environment template
# ---------------------------------------------------------------------------
Write-Step "Preparing the environment file"
$envFile = Join-Path $script:RepoRoot ".env"
$envExample = Join-Path $script:RepoRoot ".env.example"
if (Test-Path -LiteralPath $envFile) {
    Write-Ok "Keeping the existing .env"
} elseif (Test-Path -LiteralPath $envExample) {
    Copy-Item -LiteralPath $envExample -Destination $envFile
    Write-Ok "Created .env from .env.example (all values commented out by default)"
} else {
    Write-Info "No .env.example found; skipping (defaults are fine)"
}

# ---------------------------------------------------------------------------
# 4. Ollama
# ---------------------------------------------------------------------------
Write-Step "Checking Ollama"
$ollamaStatus = Get-OllamaStatus
if (-not $ollamaStatus.Ok) {
    Write-Warn "Ollama is not reachable at $($ollamaStatus.Url)"
    Write-Info $ollamaStatus.Error
    Write-Info "Start the Ollama app (or run 'ollama serve' in another terminal), then re-run this script."
    Write-Info "Amnesia still works without it, but chat/serve need a running daemon."
} else {
    Write-Ok "Ollama $($ollamaStatus.Version) at $($ollamaStatus.Url)"
    $chatModels = @($ollamaStatus.Models | Where-Object {
            $_ -notmatch "embed|bge|e5-|gte|minilm|nomic|mxbai"
        })
    if ($chatModels.Count -eq 0) {
        Write-Warn "No chat model is installed. Pull one, for example: ollama pull dolphin3:latest"
    } else {
        Write-Ok "Chat models: $($chatModels -join ', ')"
    }

    $embedInstalled = @($ollamaStatus.Models | Where-Object { $_ -like "$EmbedModel*" }).Count -gt 0
    if ($embedInstalled) {
        Write-Ok "Embedding model installed: $EmbedModel"
    } elseif (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Write-Warn "Embedding model '$EmbedModel' not found and the ollama CLI is not on PATH."
        Write-Info "Install it from the Ollama app: it improves recall with semantic search."
    } elseif ($PullEmbedModel) {
        Write-Info "Pulling the embedding model ($EmbedModel)..."
        & ollama pull $EmbedModel
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Pulled $EmbedModel"
        } else {
            Write-Warn "Could not pull $EmbedModel; Amnesia will use the deterministic hash embedder."
        }
    } else {
        Write-Warn "Embedding model '$EmbedModel' is not installed."
        Write-Info "Optional: ollama pull $EmbedModel   (or re-run with -PullEmbedModel)"
        Write-Info "Without it, recall still works via keyword search + hash embeddings."
    }
}

# ---------------------------------------------------------------------------
# 5. Memory store
# ---------------------------------------------------------------------------
Write-Step "Initializing the local memory store"
if ($SkipInit) {
    Write-Warn "Skipped (-SkipInit)"
} else {
    $initArguments = @("init")
    if ($Interactive) {
        $initArguments += "--interactive"
    }
    Invoke-Amnesia -Arguments $initArguments
    if ($script:AmnesiaLastExitCode -eq 0) {
        Write-Ok "Memory store ready"
    } else {
        Write-Warn "`amnesia init` returned exit code $script:AmnesiaLastExitCode"
        Write-Info "Run 'amnesia doctor' (or .\scripts\doctor.ps1) to inspect the setup."
    }
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
$homeDirectory = $env:AMNESIA_HOME
if ([string]::IsNullOrWhiteSpace($homeDirectory)) {
    $homeDirectory = Join-Path $HOME ".amnesia"
}

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "  Memory store : $homeDirectory\amnesia.db"
Write-Host "  Config file  : $homeDirectory\config.json"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  .\scripts\chat.ps1                 start a memory-augmented chat"
Write-Host "  .\scripts\serve.ps1                run the proxy on http://127.0.0.1:11435"
Write-Host "  .\scripts\doctor.ps1               check the installation"
Write-Host "  http://127.0.0.1:11435/amnesia/ui  dashboard (while the proxy runs)"
Write-Host ""
Write-Host "Docs: README.md  docs/INTEGRATIONS.md  docs/SCHEMA.md  docs/DEMO.md"
