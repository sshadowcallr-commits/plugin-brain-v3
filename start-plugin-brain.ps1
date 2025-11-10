# Launcher for Plugin Brain v3 - DEBUG MODE
# This script will open all consoles for debugging.
# It assumes it is being run from: C:\Users\dimbe\Documents\plugin brain v3\

Write-Host "--- PLUGIN BRAIN v3 LAUNCHER (DEBUG MODE) ---"

Write-Host "1. Starting Python Backend API Server..."
# Starts the FastAPI server IN A NEW, VISIBLE window
# The path is relative to this script
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host '--- BACKEND (Python) ---'; .\venv\Scripts\Activate.ps1; python api_server.py"

Write-Host "2. Starting React Frontend UI..."
# Starts the React server IN A NEW, VISIBLE window
# The path is relative to this script
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Write-Host '--- FRONTEND (React) ---'; cd '.\plugin-brain-ui'; npm run dev"

# Wait 5 seconds for the React server (Vite) to build
Write-Host "3. Waiting 5 seconds for frontend to build..."
Start-Sleep -Seconds 5

Write-Host "4. Opening GUI in your default browser at http://localhost:5173/"
# Opens the URL in your default browser
Start-Process "http://localhost:5173/"

Write-Host "--- LAUNCH COMPLETE ---"