#!/bin/bash
# PythonAnywhere Setup Script
# Run this script in PythonAnywhere Bash console

echo "==================================="
echo "PythonAnywhere Deployment Setup"
echo "==================================="

# Get username
USERNAME=$(whoami)
PROJECT_DIR="/home/$USERNAME/xray-healthcare-ai"

echo "Username: $USERNAME"
echo "Project Directory: $PROJECT_DIR"

# Clone repository if not exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Cloning repository..."
    cd /home/$USERNAME
    git clone https://github.com/Mrinmoypatratint/Mantra_Health.git xray-healthcare-ai
else
    echo "Repository already exists. Pulling latest changes..."
    cd "$PROJECT_DIR"
    git pull
fi

cd "$PROJECT_DIR/backend"

# Create virtual environment
echo "Creating virtual environment..."
python3.10 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
if [ -f "requirements-pythonanywhere.txt" ]; then
    pip install -r requirements-pythonanywhere.txt
else
    pip install -r requirements.txt
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p "$PROJECT_DIR/checkpoints"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/data"

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Configure your web app in PythonAnywhere Web tab"
echo "2. Set WSGI file to: $PROJECT_DIR/backend/wsgi.py"
echo "3. Set virtualenv to: $PROJECT_DIR/backend/venv"
echo "4. Reload your web app"
echo ""
