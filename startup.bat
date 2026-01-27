@echo off
REM EcoPredict Production Startup Script (Windows)

setlocal enabledelayedexpansion

echo ================================
echo EcoPredict Production Startup
echo ================================

REM Load environment variables
if exist .env (
    for /f "delims== tokens=1,*" %%A in (.env) do (
        if not "%%A"=="" (
            if not "%%A:~0,1%%"=="#" (
                set "%%A=%%B"
            )
        )
    )
    echo [OK] Environment variables loaded
) else (
    echo [ERROR] .env file not found
    exit /b 1
)

REM Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found
    exit /b 1
)
for /f "tokens=2" %%A in ('python --version 2^>^&1') do set PYTHON_VERSION=%%A
echo [OK] Python %PYTHON_VERSION% available

REM Create necessary directories
echo Creating directories...
if not exist data\raw mkdir data\raw
if not exist data\processed mkdir data\processed
if not exist models\trained mkdir models\trained
if not exist models\metrics mkdir models\metrics
if not exist outputs mkdir outputs
if not exist logs mkdir logs
echo [OK] Directories created

REM Create virtual environment if not exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    python -m pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    echo [OK] Virtual environment created
) else (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
)

REM Validate configuration
echo Validating configuration...
python -c "
import yaml
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print(f'Project: {config[\"project\"][\"name\"]}')
    print(f'Version: {config[\"project\"][\"version\"]}')
" || exit /b 1
echo [OK] Configuration valid

echo.
echo ================================
echo Starting Services
echo ================================
echo.

REM Start Docker Compose or individual services
if "%RUN_MODE%"=="api" (
    echo Starting API server...
    python -m uvicorn src.api.main:app ^
        --host %API_HOST:0.0.0.0% ^
        --port %API_PORT:8000% ^
        --workers %API_WORKERS:4% ^
        --access-log
) else if "%RUN_MODE%"=="dashboard" (
    echo Starting Dashboard...
    streamlit run src/dashboard/app.py ^
        --server.port=%DASHBOARD_PORT:8501% ^
        --server.address=0.0.0.0
) else (
    echo Starting all services with Docker Compose...
    docker-compose up -d
    echo.
    echo ================================
    echo Services Started Successfully
    echo ================================
    echo.
    echo API Documentation: http://localhost:8000/docs
    echo Dashboard: http://localhost:8501
    echo Health Check: curl http://localhost:8000/health
    echo.
)

echo [OK] Startup completed successfully
pause
