#!/bin/bash

echo "========================================"
echo "X-ray Enhancement AI - Quick Start"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "backend/venv" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please run: cd backend && python -m venv venv"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "[ERROR] Node modules not found!"
    echo "Please run: cd frontend && npm install"
    exit 1
fi

echo "[1/3] Starting Backend Server..."
cd backend
source venv/bin/activate
uvicorn app.main:app --reload &
BACKEND_PID=$!
cd ..

sleep 3

echo "[2/3] Starting Frontend Server..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

sleep 5

echo "[3/3] Opening Browser..."
if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:3000
elif command -v open > /dev/null; then
    open http://localhost:3000
fi

echo ""
echo "========================================"
echo "Application Started Successfully!"
echo "========================================"
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop all services..."

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
