# Quick Start Guide

Get up and running with X-ray Enhancement AI in 5 minutes!

## Prerequisites

- Python 3.10+
- Node.js 18+
- 16GB RAM minimum

## Quick Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/xray-healthcare-ai.git
cd xray-healthcare-ai
```

### 2. Backend Setup (2 minutes)

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Frontend Setup (2 minutes)

```bash
cd ../frontend
npm install
```

### 4. Run Application (1 minute)

**Terminal 1 - Backend:**
```bash
cd backend
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
uvicorn app.main:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

### 5. Access Application

Open browser: **http://localhost:3000**

## Using Docker (Even Faster!)

```bash
docker-compose up -d
```

That's it! 🎉

---

## Test the App

1. **Upload X-ray Image**: Drag and drop or click to select
2. **View Results**: See before/after comparison
3. **Check Metrics**: View PSNR, SSIM scores
4. **Ask Chatbot**: Click chat icon for questions

## Next Steps

- Read the full [README.md](README.md) for details
- Follow [SETUP_GUIDE.md](SETUP_GUIDE.md) for training
- Check [API Docs](http://localhost:8000/docs) for endpoints

## Need Help?

- Check [Troubleshooting](#troubleshooting) section
- Open [GitHub Issue](https://github.com/yourusername/xray-healthcare-ai/issues)

---

## Troubleshooting

**Backend won't start:**
```bash
# Check Python version
python --version  # Should be 3.10+

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Frontend errors:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
```

**Port already in use:**
```bash
# Backend: Change port
uvicorn app.main:app --reload --port 8001

# Frontend: Change port
PORT=3001 npm start
```

---

**Ready to enhance X-rays! 🏥🤖**
