param(
    [string]$Model = "llama3.2",
    [string]$OllamaHost = "http://localhost:11434",
    [string]$WorkspaceRoot = ".",
    [switch]$SkipModelPull
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

function Set-Or-AppendEnvValue {
    param(
        [string]$Path,
        [string]$Key,
        [string]$Value
    )

    $escapedKey = [Regex]::Escape($Key)
    $pattern = "^$escapedKey=.*$"
    $replacement = "$Key=$Value"

    if (Test-Path $Path) {
        $lines = Get-Content -Path $Path
    }
    else {
        $lines = @()
    }

    $updated = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $pattern) {
            $lines[$i] = $replacement
            $updated = $true
        }
    }

    if (-not $updated) {
        $lines += $replacement
    }

    Set-Content -Path $Path -Value $lines -Encoding UTF8
}

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

Write-Step "Checking Python"
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    throw "Python is required but was not found on PATH. Install Python 3.10+ and re-run install.ps1."
}

$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Step "Creating virtual environment"
    & python -m venv .venv
}

Write-Step "Installing Dad-Bot dependencies"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e ".[voice,service]"

Write-Step "Ensuring runtime folders and profile files"
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "static") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "session_logs") | Out-Null

$profilePath = Join-Path $RepoRoot "dad_profile.json"
$templatePath = Join-Path $RepoRoot "dad_profile.template.json"
if (-not (Test-Path $profilePath)) {
    if (Test-Path $templatePath) {
        Copy-Item -Path $templatePath -Destination $profilePath -Force
    }
    else {
        $fallbackProfile = @{
            name = "Dad"
            voice = @{
                tts_backend = "pyttsx3"
                piper_model_path = ""
            }
            avatar = @{}
            ical_feed_url = ""
        } | ConvertTo-Json -Depth 4
        Set-Content -Path $profilePath -Value $fallbackProfile -Encoding UTF8
    }
}

$memoryPath = Join-Path $RepoRoot "dad_memory.json"
if (-not (Test-Path $memoryPath)) {
    "{}" | Set-Content -Path $memoryPath -Encoding UTF8
}

Write-Step "Preparing .env"
$envPath = Join-Path $RepoRoot ".env"
$templateEnvPath = Join-Path $RepoRoot ".env.template"
if (-not (Test-Path $envPath) -and (Test-Path $templateEnvPath)) {
    Copy-Item -Path $templateEnvPath -Destination $envPath -Force
}
Set-Or-AppendEnvValue -Path $envPath -Key "WORKSPACE_ROOT" -Value $WorkspaceRoot
Set-Or-AppendEnvValue -Path $envPath -Key "OLLAMA_HOST" -Value $OllamaHost
Set-Or-AppendEnvValue -Path $envPath -Key "DADBOT_LLM_MODEL" -Value $Model

$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollamaCmd) {
    if (-not $SkipModelPull) {
        Write-Step "Pulling model $Model from Ollama"
        & ollama pull $Model
    }
}
else {
    Write-Warning "Ollama CLI not found. Install/start Ollama and pull model '$Model' manually if needed."
}

Write-Step "Install complete"
Write-Host "Run Dad-Bot with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe Dad.py --model $Model" -ForegroundColor Green
Write-Host "Run sovereign daemon with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\python.exe Dad.py heartbeat-daemon" -ForegroundColor Green
