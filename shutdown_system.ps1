param(
    [switch]$Force,
    [switch]$DryRun
)

Write-Host "--- SHUTTING DOWN DADBOT SOVEREIGN SYSTEM ---" -ForegroundColor Cyan

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

function Stop-SelectedProcesses {
    param(
        [string]$Label,
        [scriptblock]$Filter
    )

    $stoppedAny = $false

    Get-CimInstance Win32_Process |
        Where-Object { & $Filter $_ } |
        Select-Object ProcessId, Name, CommandLine |
        ForEach-Object {
            $proc = $_
            $stoppedAny = $true
        if ($DryRun) {
            Write-Host "[$Label] DRY RUN would stop PID $($proc.ProcessId) ($($proc.Name))" -ForegroundColor Yellow
            return
        }
        try {
            Stop-Process -Id $proc.ProcessId -Force:$Force
            Write-Host "[$Label] Stopped PID $($proc.ProcessId) ($($proc.Name))" -ForegroundColor Green
        } catch {
            Write-Host "[$Label] Failed to stop PID $($proc.ProcessId): $($_.Exception.Message)" -ForegroundColor Red
        }
    }

    if (-not $stoppedAny) {
        Write-Host "[$Label] No matching processes found." -ForegroundColor DarkYellow
    }
}

# 1. Stop backend API started via dadbot app runtime
Stop-SelectedProcesses -Label "KERNEL" -Filter {
    param($p)
    $cmd = [string]$p.CommandLine
    $name = [string]$p.Name
    $isPython = [regex]::IsMatch($name, '^python(\.exe)?$')
    $isApi = $cmd -like '*dadbot.app_runtime.main*' -and $cmd -like '*--serve-api*'
    $inRepo = $cmd -like "*$repoRoot*"
    return $isPython -and $isApi -and $inRepo
}

# 2. Stop HUD dev server (Vite) processes
Stop-SelectedProcesses -Label "HUD" -Filter {
    param($p)
    $cmd = [string]$p.CommandLine
    $name = [string]$p.Name
    $isNode = [regex]::IsMatch($name, '^node(\.exe)?$')
    $isVite = $cmd -like '*vite*' -or $cmd -like '*npm run dev*'
    $inHud = $cmd -like '*dadbot-hud*'
    return $isNode -and $isVite -and $inHud
}

Write-Host "Shutdown sweep complete." -ForegroundColor Cyan
