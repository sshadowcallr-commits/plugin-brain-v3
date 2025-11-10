# Frontend Launcher for Plugin Brain v5
$v5 = "C:\Users\dimbe\Documents\plugin brain v3\plug-in brain v5"

Write-Host "Starting Plugin Brain v5 Frontend..." -ForegroundColor Cyan
Set-Location "$v5\plugin-brain-ui"

if (Test-Path "package.json") {
    npm run electron
} else {
    Write-Host "ERROR: package.json not found. Run 'npm install' first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
