@echo off
echo ========================================
echo X-ray Enhancement AI - Installation
echo ========================================
echo.

echo [Step 1/4] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

echo.
echo [Step 2/4] Setting up Backend...
cd backend
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
echo Installing Python dependencies...
pip install --upgrade pip
pip install -r requirements.txt
cd ..

echo.
echo [Step 3/4] Checking Node.js...
node --version
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found! Please install Node.js 18+
    pause
    exit /b 1
)

echo.
echo [Step 4/4] Setting up Frontend...
cd frontend
echo Installing Node dependencies...
call npm install
cd ..

echo.
echo Creating necessary directories...
mkdir checkpoints 2>nul
mkdir logs 2>nul
mkdir data\train 2>nul
mkdir data\val 2>nul

echo.
echo Creating environment file...
if not exist ".env" (
    copy .env.example .env
    echo .env file created. Please edit with your configuration.
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next Steps:
echo 1. Edit .env file with your configuration
echo 2. Add training images to data/train and data/val
echo 3. Run: start.bat
echo.
echo Or use Docker:
echo   docker-compose up -d
echo.
echo See NEXT_STEPS.md for detailed instructions.
echo ========================================
pause
