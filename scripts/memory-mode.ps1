#Requires -Version 5.1
<#
.SYNOPSIS
    Install and control Amnesia's always-on "memory mode" on Windows.

.DESCRIPTION
    Memory mode starts Amnesia's zero-config takeover server hidden at Windows
    sign-in, so plain `ollama run` gets long-term memory without opening a
    terminal and without setting OLLAMA_HOST.

    -Install writes a small launcher to your per-user Startup folder
    ("Amnesia Memory Mode.vbs"). At sign-in it runs
    `python -m amnesia serve --takeover --background` with no console window,
    waits until the server is healthy, and logs to ~/.amnesia/logs/serve.log.
    Optionally it also disables Ollama's own Startup shortcut (reversibly) so
    the Ollama tray app does not race Amnesia for port 11434.

    Manage it any time:
        -Start     start memory mode now (background)
        -Stop      stop it gracefully - Amnesia restores Ollama
        -Status    check whether memory mode is running
        -Open      open the dashboard profile tab
        -Uninstall remove the Startup launcher and restore Ollama autostart

    Amnesia is the only thing that starts or stops the server: this script calls
    `python -m amnesia ...` and never kills processes directly, so a graceful
    stop always restores the Ollama daemon / desktop app.

.PARAMETER Install
    Write the Startup launcher and (unless -NoStart) start memory mode now.

.PARAMETER Uninstall
    Stop memory mode (unless -NoStop), delete the Startup launcher, and restore
    Ollama's Startup shortcut if it was disabled.

.PARAMETER Start
    Start memory mode now in the background; prints the dashboard URL.

.PARAMETER Stop
    Stop the running server gracefully with `python -m amnesia stop`.

.PARAMETER Status
    Report whether memory mode is running (health endpoint 11434) and the
    managed Ollama daemon version (private endpoint 11435). Never throws when
    nothing is running.

.PARAMETER Open
    With -Start or -Status: open the dashboard profile tab (#profile).

.PARAMETER DisableOllamaAutostart
    With -Install: rename Ollama's Startup shortcut to
    "Ollama.lnk.amnesia-disabled" (never deleted; -Uninstall restores it).

.PARAMETER NoStart
    With -Install: install the launcher only, do not start memory mode now.

.PARAMETER NoStop
    With -Uninstall: remove the launcher without stopping a running server.

.PARAMETER StartupFolder
    Startup folder override (default: the current user's Startup folder).
    Useful for testing against a temporary folder.

.PARAMETER RepoRoot
    Amnesia checkout root override (default: the parent folder of this script).

.EXAMPLE
    .\scripts\memory-mode.ps1 -Install

.EXAMPLE
    .\scripts\memory-mode.ps1 -Install -DisableOllamaAutostart

.EXAMPLE
    .\scripts\memory-mode.ps1 -Status -Open

.EXAMPLE
    .\scripts\memory-mode.ps1 -Stop

.EXAMPLE
    .\scripts\memory-mode.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Start,
    [switch]$Stop,
    [switch]$Status,
    [switch]$Open,
    [switch]$DisableOllamaAutostart,
    [switch]$NoStart,
    [switch]$NoStop,
    [string]$StartupFolder = "",
    [string]$RepoRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Output helpers (ASCII only)
# ---------------------------------------------------------------------------

function Write-Section {
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

function Show-Usage {
    Write-Host ""
    Write-Host "Amnesia always-on memory mode" -ForegroundColor Magenta
    Write-Host ""
    Write-Host "Usage: .\scripts\memory-mode.ps1 <action> [options]"
    Write-Host ""
    Write-Host "Actions (exactly one is required):"
    Write-Host "  -Install      Install the hidden Startup launcher and start memory mode"
    Write-Host "  -Uninstall    Stop memory mode, remove the launcher, restore Ollama autostart"
    Write-Host "  -Start        Start memory mode now (background)"
    Write-Host "  -Stop         Stop it gracefully (Amnesia restores Ollama)"
    Write-Host "  -Status       Show whether memory mode is running"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Open                     Open the dashboard profile tab"
    Write-Host "  -DisableOllamaAutostart   With -Install: rename Ollama's Startup shortcut (reversible)"
    Write-Host "  -NoStart                  With -Install: do not start memory mode now"
    Write-Host "  -NoStop                   With -Uninstall: leave a running server alone"
    Write-Host "  -StartupFolder <path>     Startup folder override (default: current user's)"
    Write-Host "  -RepoRoot <path>          Amnesia checkout override (default: parent of this script)"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\scripts\memory-mode.ps1 -Install"
    Write-Host "  .\scripts\memory-mode.ps1 -Install -DisableOllamaAutostart"
    Write-Host "  .\scripts\memory-mode.ps1 -Status -Open"
    Write-Host "  .\scripts\memory-mode.ps1 -Stop"
    Write-Host ""
}

# ---------------------------------------------------------------------------
# Work out what to do
# ---------------------------------------------------------------------------

$primaryActions = @()
if ($Install) { $primaryActions += "Install" }
if ($Uninstall) { $primaryActions += "Uninstall" }
if ($Start) { $primaryActions += "Start" }
if ($Stop) { $primaryActions += "Stop" }
if ($Status) { $primaryActions += "Status" }

if ($primaryActions.Count -ne 1) {
    if ($primaryActions.Count -gt 1) {
        Write-Warn ("Pick exactly one action, got: " + ($primaryActions -join ", "))
    }
    Show-Usage
    exit 1
}
$Action = $primaryActions[0]

if ($DisableOllamaAutostart -and $Action -ne "Install") {
    Write-Warn "-DisableOllamaAutostart only applies to -Install; ignoring it."
}
# -NoStart / -NoStop are safety overrides; ignore them silently when irrelevant.


# ---------------------------------------------------------------------------
# Resolve paths and the Python interpreter
# ---------------------------------------------------------------------------

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = Split-Path -Parent $PSScriptRoot
}
if ([string]::IsNullOrWhiteSpace($RepoRoot) -or
    -not (Test-Path -LiteralPath (Join-Path $RepoRoot "amnesia\__init__.py"))) {
    throw "RepoRoot does not look like an Amnesia checkout: '$RepoRoot'. Pass -RepoRoot <path>."
}
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path

if ([string]::IsNullOrWhiteSpace($StartupFolder)) {
    $StartupFolder = [Environment]::GetFolderPath("Startup")
}
$StartupFolder = [System.IO.Path]::GetFullPath($StartupFolder)

$venvPython = Join-Path (Join-Path $RepoRoot ".venv") "Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    $python = $venvPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $pythonCommand -or [string]::IsNullOrWhiteSpace($pythonCommand.Source)) {
        throw "Python was not found. Install Python 3.10+ (or run .\scripts\setup.ps1) and try again."
    }
    $python = $pythonCommand.Source
}

$pythonwCandidate = Join-Path (Split-Path -Parent $python) "pythonw.exe"
if (Test-Path -LiteralPath $pythonwCandidate) {
    $pythonw = $pythonwCandidate
} else {
    $pythonw = $python
}

# Child processes must import the checkout even without an editable install.
$env:PYTHONPATH = $RepoRoot

$launcherName = "Amnesia Memory Mode.vbs"
$launcherPath = Join-Path $StartupFolder $launcherName
$ollamaShortcut = Join-Path $StartupFolder "Ollama.lnk"
$ollamaDisabledShortcut = "$ollamaShortcut.amnesia-disabled"

$amnesiaHome = $env:AMNESIA_HOME
if ([string]::IsNullOrWhiteSpace($amnesiaHome)) {
    $amnesiaHome = Join-Path $HOME ".amnesia"
}
$logsFolder = Join-Path $amnesiaHome "logs"
$serveLog = Join-Path $logsFolder "serve.log"
$servePid = Join-Path $logsFolder "serve.pid"

$dashboardUrl = "http://127.0.0.1:11434/amnesia/ui#profile"
$healthUrl = "http://127.0.0.1:11434/amnesia/health"
$upstreamVersionUrl = "http://127.0.0.1:11435/api/version"

# Exit code of the last `python -m amnesia ...` process.
$script:CliExitCode = 0

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

function Invoke-AmnesiaCli {
    <# Run the Amnesia CLI from the repo root; output goes to the console. #>
    param([string[]]$Arguments = @())
    Push-Location -LiteralPath $RepoRoot
    try {
        # Out-Host keeps native-command output visible instead of letting it
        # become part of the function's return value.
        & $python -m amnesia @Arguments | Out-Host
        $script:CliExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
}

function Test-PsutilAvailable {
    <# Takeover needs psutil to stop the Ollama app/daemon it replaces. #>
    & $python -c "import psutil" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Open-Dashboard {
    Start-Process -FilePath $dashboardUrl
    Write-Ok "Opened the dashboard: $dashboardUrl"
}

function ConvertTo-VbsString {
    <# Quote a PowerShell string as a VBScript string literal. #>
    param([string]$Value)
    return '"' + ($Value -replace '"', '""') + '"'
}

function Write-Launcher {
    <# Generate the hidden Startup launcher for the current paths. #>
    $commandText = (ConvertTo-VbsString $pythonw) + " -m amnesia serve --takeover --background"
    $lines = @(
        "' Amnesia memory mode - generated by scripts/memory-mode.ps1",
        "' Starts the Amnesia takeover server hidden in the background at sign-in.",
        "' Remove with: .\scripts\memory-mode.ps1 -Uninstall",
        "Option Explicit",
        "Dim sh, cmd",
        "Set sh = CreateObject(""WScript.Shell"")",
        ("sh.Environment(""PROCESS"")(""PYTHONPATH"") = " + (ConvertTo-VbsString $RepoRoot)),
        ("sh.CurrentDirectory = " + (ConvertTo-VbsString $RepoRoot)),
        ("cmd = " + (ConvertTo-VbsString $commandText)),
        "sh.Run cmd, 0, False"
    )
    $text = ($lines -join "`r`n") + "`r`n"

    New-Item -ItemType Directory -Force -Path $StartupFolder | Out-Null
    if ($text -match "[^\x00-\x7F]") {
        # Non-ASCII path: write UTF-16 with BOM, which Windows Script Host reads.
        [System.IO.File]::WriteAllText(
            $launcherPath,
            $text,
            [System.Text.UnicodeEncoding]::new($false, $true)
        )
    } else {
        [System.IO.File]::WriteAllText(
            $launcherPath,
            $text,
            [System.Text.ASCIIEncoding]::new()
        )
    }
}

function Start-MemoryMode {
    <# Start the background takeover server and print the dashboard/log hints. #>
    param([switch]$OpenDashboard)

    if (-not (Test-PsutilAvailable)) {
        Write-Warn "Takeover mode needs psutil. Install it with: python -m pip install psutil"
        Write-Info "(or reinstall Amnesia with the takeover extra: python -m pip install -e `".[takeover]`")"
    }

    Write-Info "Starting memory mode in the background ..."
    Invoke-AmnesiaCli -Arguments @("serve", "--takeover", "--background")
    if ($script:CliExitCode -ne 0) {
        Write-Warn "amnesia serve --takeover --background failed (exit code $script:CliExitCode)."
        Write-Info "Log: $serveLog"
        Write-Info "Try running it in the foreground to see the error: amnesia serve --takeover"
        return $false
    }

    Write-Ok "Memory mode is running in the background."
    Write-Info "Dashboard: $dashboardUrl"
    Write-Info "Log:       $serveLog"
    Write-Info "Stop with: .\scripts\memory-mode.ps1 -Stop   (or: amnesia stop)"
    if ($OpenDashboard) {
        Open-Dashboard
    }
    return $true
}

function Test-MemoryModeHealth {
    <# Return the health payload, or $null when memory mode is not running. #>
    try {
        return Invoke-RestMethod -Uri $healthUrl -Method Get -TimeoutSec 3 -ErrorAction Stop
    } catch {
        return $null
    }
}

function Get-ObjectProperty {
    <# Safe property access that works under Set-StrictMode. #>
    param(
        $InputObject,
        [string]$Name,
        $Default = $null
    )
    if ($null -eq $InputObject) { return $Default }
    $property = $InputObject.PSObject.Properties[$Name]
    if ($null -eq $property) { return $Default }
    return $property.Value
}

function Format-ByteSize {
    param($Value)
    if ($null -eq $Value) { return "?" }
    try {
        $size = [double]$Value
    } catch {
        return "?"
    }
    foreach ($unit in @("B", "KB", "MB", "GB")) {
        if ($size -lt 1024 -or $unit -eq "GB") {
            return ("{0:N1} {1}" -f $size, $unit)
        }
        $size = $size / 1024
    }
    return ("{0:N1} GB" -f $size)
}

function Show-Status {
    <# Report whether memory mode is running, using read-only HTTP probes. #>
    $health = Test-MemoryModeHealth
    Write-Section "Amnesia memory mode"

    if ($null -eq $health) {
        Write-Host "    Memory mode is not running." -ForegroundColor Yellow
        Write-Info "Start it now:  .\scripts\memory-mode.ps1 -Start"
        Write-Info "Install it at sign-in: .\scripts\memory-mode.ps1 -Install"
        return 1
    }

    Write-Ok "Running - dashboard: $dashboardUrl"

    $memoryEnabled = Get-ObjectProperty (Get-ObjectProperty $health "settings") "memory_enabled"
    if ($null -eq $memoryEnabled) {
        $memoryEnabled = Get-ObjectProperty $health "started"
    }
    Write-Info ("Memory injection : " + $(if ($memoryEnabled) { "on" } else { "off" }))

    $embeddings = Get-ObjectProperty $health "embeddings"
    $embedModel = Get-ObjectProperty $embeddings "model" "unknown"
    $embedDim = Get-ObjectProperty $embeddings "dim" "?"
    Write-Info "Embeddings       : $embedModel ($embedDim dims)"
    if ("$embedModel".StartsWith("hash:")) {
        Write-Warn "Semantic embeddings are degraded - Ollama was not ready when memory mode started."
        Write-Info "Fix: .\scripts\memory-mode.ps1 -Stop ; then .\scripts\memory-mode.ps1 -Start"
    }

    $dbPath = Get-ObjectProperty $health "db_path" "(unknown)"
    Write-Info "Database         : $dbPath"

    $dbSize = Format-ByteSize (Get-ObjectProperty $health "db_size_bytes")
    Write-Info "Database size    : $dbSize"

    $counts = Get-ObjectProperty $health "counts"
    if ($null -ne $counts) {
        $memories = Get-ObjectProperty $counts "memories" 0
        $events = Get-ObjectProperty $counts "events" 0
        $sessions = Get-ObjectProperty $counts "sessions" 0
        $messages = Get-ObjectProperty $counts "messages" 0
        Write-Info "Stored           : memories=$memories events=$events sessions=$sessions messages=$messages"
    }

    try {
        $upstream = Invoke-RestMethod -Uri $upstreamVersionUrl -Method Get -TimeoutSec 2 -ErrorAction Stop
        $version = Get-ObjectProperty $upstream "version" "unknown"
        Write-Info "Private Ollama   : version $version (takeover active on 11435)"
    } catch {
        Write-Info "Private Ollama   : not detected on 11435 (takeover may not be active)"
    }

    Write-Info "Stop with        : .\scripts\memory-mode.ps1 -Stop   (or: amnesia stop)"

    if ($Open) {
        Open-Dashboard
    }
    return 0
}

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

switch ($Action) {
    "Install" {
        Write-Section "Installing Amnesia memory mode"
        Write-Info "Repository : $RepoRoot"
        Write-Info "Python     : $python"
        Write-Info "Launcher   : $pythonw (hidden window)"
        Write-Info "Startup    : $StartupFolder"

        Write-Launcher
        Write-Ok "Startup launcher written: $launcherPath"

        if ($DisableOllamaAutostart) {
            if (Test-Path -LiteralPath $ollamaShortcut) {
                Move-Item -LiteralPath $ollamaShortcut -Destination $ollamaDisabledShortcut -Force
                Write-Ok "Disabled Ollama's Startup shortcut (renamed to Ollama.lnk.amnesia-disabled)."
            } elseif (Test-Path -LiteralPath $ollamaDisabledShortcut) {
                Write-Ok "Ollama's Startup shortcut is already disabled."
            } else {
                Write-Info "No Ollama.lnk in the Startup folder - nothing to disable."
            }
        }

        $started = $false
        $exitCode = 0
        if ($NoStart) {
            Write-Warn "Skipped the initial start (-NoStart); memory mode will start at next sign-in."
        } else {
            $started = Start-MemoryMode
            if (-not $started) {
                $exitCode = 1
            }
        }

        Write-Section "Memory mode installed"
        Write-Info "Launcher  : $launcherPath"
        Write-Info "Log       : $serveLog"
        Write-Info "PID file  : $servePid"
        Write-Info "Dashboard : $dashboardUrl"
        if ($started) {
            Write-Ok "Memory mode is running now; it also starts hidden at every Windows sign-in."
        } elseif (-not $NoStart) {
            Write-Warn "The launcher is installed, but memory mode did not start just now."
        } else {
            Write-Info "At the next Windows sign-in, Amnesia starts memory mode hidden in the background."
        }
        if ($DisableOllamaAutostart -and (Test-Path -LiteralPath $ollamaDisabledShortcut)) {
            Write-Info "Ollama autostart: disabled (reversible with -Uninstall)."
        }
        Write-Info "Desktop app: while memory mode runs, use `ollama run` in a terminal or the dashboard."
        Write-Info "Stop now  : .\scripts\memory-mode.ps1 -Stop"
        Write-Info "Remove    : .\scripts\memory-mode.ps1 -Uninstall"
        exit $exitCode
    }

    "Uninstall" {
        Write-Section "Uninstalling Amnesia memory mode"

        if (-not $NoStop) {
            Write-Info "Stopping memory mode gracefully (amnesia stop) ..."
            Invoke-AmnesiaCli -Arguments @("stop")
            if ($script:CliExitCode -eq 0) {
                Write-Ok "Stop requested; Ollama is restored by Amnesia (desktop app if it was running)."
            } else {
                Write-Warn "amnesia stop returned exit code $script:CliExitCode; continuing with the uninstall."
            }
        } else {
            Write-Warn "Skipped stopping the server (-NoStop)."
        }

        if (Test-Path -LiteralPath $launcherPath) {
            Remove-Item -LiteralPath $launcherPath -Force
            Write-Ok "Removed the Startup launcher: $launcherPath"
        } else {
            Write-Info "Startup launcher was not present: $launcherPath"
        }

        if (Test-Path -LiteralPath $ollamaDisabledShortcut) {
            if (Test-Path -LiteralPath $ollamaShortcut) {
                Write-Warn "Both Ollama.lnk and Ollama.lnk.amnesia-disabled exist; leaving the disabled copy for manual cleanup."
            } else {
                Move-Item -LiteralPath $ollamaDisabledShortcut -Destination $ollamaShortcut -Force
                Write-Ok "Restored Ollama's Startup shortcut."
            }
        } else {
            Write-Info "No disabled Ollama Startup shortcut to restore."
        }

        Write-Section "Memory mode removed"
        Write-Info "Amnesia no longer starts at sign-in."
        Write-Info "Logs remain in: $logsFolder"
        exit 0
    }

    "Start" {
        $started = Start-MemoryMode -OpenDashboard:$Open
        if (-not $started) {
            exit 1
        }
        exit 0
    }

    "Stop" {
        Write-Section "Stopping Amnesia memory mode"
        Invoke-AmnesiaCli -Arguments @("stop")
        if ($script:CliExitCode -ne 0) {
            Write-Warn "amnesia stop returned exit code $script:CliExitCode."
            Write-Info "Log: $serveLog"
            exit $script:CliExitCode
        }
        Write-Ok "Memory mode stopped; Ollama is restored (desktop app if it was running before)."
        exit 0
    }

    "Status" {
        exit (Show-Status)
    }
}
