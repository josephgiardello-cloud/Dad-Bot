Write-Host "--- IGNITING DADBOT SOVEREIGN SYSTEM ---" -ForegroundColor Cyan

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

# 1. Start the Backend API in the background
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $pythonCmd = $venvPython
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} else {
    Write-Host "[ERROR] Python not found. Activate venv or install Python." -ForegroundColor Red
    return
}

Start-Process -FilePath $pythonCmd -ArgumentList "-m", "dadbot.app_runtime.main", "--serve-api" -WorkingDirectory $repoRoot
Write-Host "[KERNEL] API ignited at http://127.0.0.1:8010/v1" -ForegroundColor Green

# 2. Check for Node Path
if (!(Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Node.js not found. Extract/install Node and update PATH." -ForegroundColor Red
    return
}

# 3. Print HUD Access Links
Write-Host "`n--- ACCESS COCKPIT ---" -ForegroundColor Yellow
Write-Host "LIVE MODE:    http://localhost:5173"
Write-Host "FIXTURE MODE: http://localhost:5173?fixtures=1"

# 4. Start HUD Dev Server
Set-Location (Join-Path $repoRoot "dadbot-hud")
if ($env:DADBOT_SECRET_KEY) {
    $env:VITE_DADBOT_SECRET_KEY = $env:DADBOT_SECRET_KEY
    Write-Host "[HUD] Handshake key forwarded to Vite env (VITE_DADBOT_SECRET_KEY)." -ForegroundColor Green
} else {
    Write-Host "[WARN] DADBOT_SECRET_KEY not set. Handshake enforcement is disabled unless backend env is configured separately." -ForegroundColor Yellow
}
npm run dev
