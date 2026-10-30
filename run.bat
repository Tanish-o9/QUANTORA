@echo off
REM Quantora Engineering Platform CLI & Dev Runner

IF "%1"=="dev" GOTO dev
IF "%1"=="test" GOTO test
IF "%1"=="api" GOTO api
IF "%1"=="web" GOTO web
IF "%1"=="bench" GOTO bench

echo ========================================================
echo   QUANTORA - LLM Performance Engineering Lab
echo ========================================================
echo   Usage: run.bat [command]
echo.
echo   Commands:
echo     dev     - Launch both Python REST API and React Web UI
echo     test    - Run complete PyTest suite
echo     api     - Start Quantora REST API server (port 8000)
echo     web     - Start React Vite frontend (port 5173)
echo     bench   - Run quick benchmark on default model
echo ========================================================
GOTO end

:dev
echo Starting Quantora API and Web UI...
start "Quantora API (Port 8000)" cmd /k python server.py
start "Quantora Web Dashboard (Port 5173)" cmd /k "cd web && npm run dev"
echo Both servers started in background windows.
GOTO end

:test
echo Running PyTest suite...
python -m pytest tests/
GOTO end

:api
python server.py
GOTO end

:web
cd web && npm run dev
GOTO end

:bench
python cli.py benchmark --model Qwen/Qwen2.5-0.5B-Instruct --precision fp16 --trials 1 --max-tokens 32
GOTO end

:end
