# Backend Launcher for Plugin Brain v5
$base = "C:\Users\dimbe\Documents\plugin brain v3"
$v5 = "$base\plug-in brain v5"

Write-Host "Starting Plugin Brain v5 Backend..." -ForegroundColor Cyan
Set-Location $v5

# Activate venv
if (Test-Path "$base\venv\Scripts\Activate.ps1") {
    & "$base\venv\Scripts\Activate.ps1"
} else {
    & "$v5\.venv\Scripts\Activate.ps1"
}

# Start server
Write-Host "Backend running at http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
uvicorn api.api_server:app --reload --host 127.0.0.1 --port 8000
