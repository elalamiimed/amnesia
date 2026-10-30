#Requires -Version 5.1
<#
.SYNOPSIS
    Chat with a local Ollama model through Amnesia's long-term memory (Windows).

.DESCRIPTION
    Starts the interactive streaming REPL (`amnesia chat`) with the memory
    block injected into each turn, or runs a single question with `-Ask`.

    Inside the REPL these slash commands are available:
      /help /remember TEXT /recall Q /profile /timeline /events /forget N
      /new /session /model NAME /memory on|off /stats /export PATH /quit

.PARAMETER Model
    Chat model to use, as shown by `ollama list`
    (for example dolphin3:latest). Defaults to the configured model.

.PARAMETER Session
    Resume a specific session id (list them with `amnesia sessions`).

.PARAMETER NoMemory
    Disable memory injection for this chat - useful for a clean comparison.

.PARAMETER NoStream
    Wait for the full answer instead of streaming tokens.

.PARAMETER Ask
    Ask one question and exit (uses `amnesia ask`) instead of the REPL.

.EXAMPLE
    .\scripts\chat.ps1

.EXAMPLE
    .\scripts\chat.ps1 -Model dolphin3:latest

.EXAMPLE
    .\scripts\chat.ps1 -Ask "What do you remember about my projects?"
#>
[CmdletBinding()]
param(
    [string]$Model = "",
    [string]$Session = "",
    [switch]$NoMemory,
    [switch]$NoStream,
    [string]$Ask = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "_common.ps1")

Write-Host ""
Write-Host "Amnesia chat" -ForegroundColor Magenta

$ollamaStatus = Get-OllamaStatus
if ($ollamaStatus.Ok) {
    if ($ollamaStatus.Models.Count -gt 0) {
        Write-Host "  Ollama   : $($ollamaStatus.Url)  ($($ollamaStatus.Models -join ', '))"
    } else {
        Write-Host "  Ollama   : $($ollamaStatus.Url)  (no models installed yet)"
        Write-Host "             Pull one first, for example: ollama pull dolphin3:latest" -ForegroundColor Yellow
    }
} else {
    Write-Host "  Ollama   : unreachable at $($ollamaStatus.Url)" -ForegroundColor Yellow
    Write-Host "             Start the Ollama app first; chat needs the local daemon." -ForegroundColor Yellow
}

if (-not $Ask) {
    Write-Host "  Memory   : $(if ($NoMemory) { 'disabled (-NoMemory)' } else { 'injected automatically' })"
    Write-Host "  Tip      : type /help for slash commands, /quit to leave"
}
Write-Host ""

if ($Ask) {
    $cliArguments = @("ask", $Ask)
    if ($Model) {
        $cliArguments += @("--model", $Model)
    }
    if ($NoMemory) {
        $cliArguments += "--no-memory"
    }
} else {
    $cliArguments = @("chat")
    if ($Model) {
        $cliArguments += $Model
    }
    if ($Session) {
        $cliArguments += @("--session", $Session)
    }
    if ($NoMemory) {
        $cliArguments += "--no-memory"
    }
    if ($NoStream) {
        $cliArguments += "--no-stream"
    }
}

Invoke-Amnesia -Arguments $cliArguments
exit $script:AmnesiaLastExitCode
