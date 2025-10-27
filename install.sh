#!/bin/bash

echo "========================================"
echo "X-ray Enhancement AI - Installation"
echo "========================================"
echo ""

# Check Python
echo "[Step 1/4] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found! Please install Python 3.10+"
    exit 1
fi
python3 --version

# Setup Backend
echo ""
echo "[Step 2/4] Setting up Backend..."
cd backend
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
deactivate
cd ..

# Check Node.js
echo ""
echo "[Step 3/4] Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo "[ERROR] Node.js not found! Please install Node.js 18+"
    exit 1
fi
node --version

# Setup Frontend
echo ""
echo "[Step 4/4] Setting up Frontend..."
cd frontend
echo "Installing Node dependencies..."
npm install
cd ..

# Create directories
echo ""
echo "Creating necessary directories..."
mkdir -p checkpoints logs data/train data/val

# Create .env file
echo ""
echo "Creating environment file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ".env file created. Please edit with your configuration."
fi

# Make scripts executable
chmod +x start.sh
chmod +x install.sh

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Next Steps:"
echo "1. Edit .env file with your configuration"
echo "2. Add training images to data/train and data/val"
echo "3. Run: ./start.sh"
echo ""
echo "Or use Docker:"
echo "  docker-compose up -d"
echo ""
echo "See NEXT_STEPS.md for detailed instructions."
echo "========================================"
