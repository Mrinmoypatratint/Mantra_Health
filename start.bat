@echo off
echo ========================================
echo X-ray Enhancement AI - Quick Start
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "backend\venv\" (
    echo [ERROR] Virtual environment not found!
    echo Please run: cd backend ^&^& python -m venv venv
    pause
    exit /b 1
)

REM Check if node_modules exists
if not exist "frontend\node_modules\" (
    echo [ERROR] Node modules not found!
    echo Please run: cd frontend ^&^& npm install
    pause
    exit /b 1
)

echo [1/3] Starting Backend Server...
start cmd /k "cd backend && venv\Scripts\activate && uvicorn app.main:app --reload"
timeout /t 3 /nobreak > nul

echo [2/3] Starting Frontend Server...
start cmd /k "cd frontend && npm start"
timeout /t 3 /nobreak > nul

echo [3/3] Opening Browser...
timeout /t 5 /nobreak > nul
start http://localhost:3000

echo.
echo ========================================
echo Application Started Successfully!
echo ========================================
echo Frontend: http://localhost:3000
echo Backend:  http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo ========================================
echo.
echo Press any key to exit this window...
pause > nul
