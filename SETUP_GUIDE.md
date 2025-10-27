# Complete Setup Guide

This guide provides step-by-step instructions for setting up the X-ray Enhancement AI application from scratch.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Local Development Setup](#local-development-setup)
3. [Google Colab Setup](#google-colab-setup)
4. [Cloud Deployment](#cloud-deployment)
5. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS 11+
- **RAM**: 16 GB
- **Storage**: 50 GB free space
- **Python**: 3.10 or higher
- **Node.js**: 18.0 or higher

### Recommended for Training
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **CUDA**: 11.8 or higher
- **cuDNN**: 8.0 or higher
- **RAM**: 32 GB
- **Storage**: 100 GB SSD

---

## Local Development Setup

### Step 1: Install Python and Dependencies

#### Windows

```powershell
# Check Python version
python --version

# Should be 3.10 or higher. If not, download from:
# https://www.python.org/downloads/

# Install Visual C++ Build Tools (required for some packages)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### Linux (Ubuntu/Debian)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10
sudo apt install python3.10 python3.10-venv python3-pip -y

# Install system dependencies
sudo apt install build-essential libssl-dev libffi-dev python3-dev -y
sudo apt install libgl1-mesa-glx libglib2.0-0 -y
```

#### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.10
brew install python@3.10
```

### Step 2: Install CUDA and PyTorch (GPU Training)

#### Windows/Linux with NVIDIA GPU

```bash
# Check CUDA version
nvidia-smi

# Install PyTorch with CUDA support
# Visit: https://pytorch.org/get-started/locally/
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Verify Installation

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
```

### Step 3: Clone and Setup Backend

```bash
# Clone repository
git clone https://github.com/yourusername/xray-healthcare-ai.git
cd xray-healthcare-ai

# Create virtual environment
cd backend
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p ../checkpoints ../data/train ../data/val ../logs
```

### Step 4: Setup Frontend

```bash
# Install Node.js (if not installed)
# Download from: https://nodejs.org/

# Navigate to frontend directory
cd ../frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env
```

Edit `frontend/.env`:
```env
REACT_APP_API_URL=http://localhost:8000
```

### Step 5: Configure Environment Variables

Create `.env` file in root directory:

```bash
cd ..
cp .env.example .env
```

Edit `.env`:
```env
# Optional: OpenAI API key for chatbot
OPENAI_API_KEY=your_api_key_here

# Model path
MODEL_PATH=./checkpoints/best_model.pth

# Backend configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
```

### Step 6: Download or Train Model

#### Option A: Download Pre-trained Model
```bash
# Download from your model repository
# Place in ./checkpoints/best_model.pth
```

#### Option B: Train Your Own Model

1. **Prepare Dataset**:
```bash
# Download NIH ChestX-ray14 or other datasets
# Organize in data/train and data/val folders

data/
├── train/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── val/
    ├── image1.png
    └── ...
```

2. **Start Training**:
```bash
cd training
python train.py
```

3. **Monitor with TensorBoard**:
```bash
tensorboard --logdir=../logs
# Open browser: http://localhost:6006
```

### Step 7: Run the Application

#### Terminal 1: Backend
```bash
cd backend
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Terminal 2: Frontend
```bash
cd frontend
npm start
```

Application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

---

## Google Colab Setup

For training on Google Colab with free GPU:

### 1. Create New Colab Notebook

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/yourusername/xray-healthcare-ai.git
%cd xray-healthcare-ai

# Install dependencies
!pip install -r backend/requirements.txt
```

### 2. Upload Dataset

```python
# Option 1: Upload from local
from google.colab import files
uploaded = files.upload()

# Option 2: Download from Kaggle
!pip install kaggle
!mkdir ~/.kaggle
!cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
!unzip chest-xray-pneumonia.zip -d data/
```

### 3. Organize Data

```python
import os
import shutil

# Organize dataset
source_dir = 'data/chest_xray'
train_dir = 'data/train'
val_dir = 'data/val'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Copy files (adjust paths based on your dataset)
# ... (dataset-specific code)
```

### 4. Train Model

```python
# Set GPU runtime: Runtime > Change runtime type > GPU

# Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Train
%cd training
!python train.py
```

### 5. Download Trained Model

```python
from google.colab import files

# Download checkpoint
files.download('checkpoints/best_model.pth')

# Or save to Drive
shutil.copy('checkpoints/best_model.pth', '/content/drive/MyDrive/xray_model.pth')
```

---

## Cloud Deployment

### Google Cloud Run

#### 1. Install Google Cloud SDK

```bash
# Download from: https://cloud.google.com/sdk/docs/install

# Login
gcloud auth login

# Set project
gcloud config set project YOUR_PROJECT_ID
```

#### 2. Build and Deploy Backend

```bash
# Build image
cd backend
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/xray-backend

# Deploy to Cloud Run
gcloud run deploy xray-backend \
  --image gcr.io/YOUR_PROJECT_ID/xray-backend \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --allow-unauthenticated \
  --set-env-vars MODEL_PATH=/app/checkpoints/best_model.pth
```

#### 3. Deploy Frontend (Netlify)

```bash
cd ../frontend

# Update API URL in .env
REACT_APP_API_URL=https://xray-backend-xxx-uc.a.run.app

# Build
npm run build

# Deploy to Netlify
npm install -g netlify-cli
netlify login
netlify deploy --prod --dir=build
```

### AWS Deployment

#### 1. Setup AWS CLI

```bash
# Install AWS CLI
# Windows: https://aws.amazon.com/cli/
# Linux: sudo apt install awscli

# Configure
aws configure
```

#### 2. Create ECR Repository

```bash
aws ecr create-repository --repository-name xray-backend --region us-east-1
```

#### 3. Build and Push Docker Image

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

# Build image
docker build -t xray-backend ./backend

# Tag and push
docker tag xray-backend:latest ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/xray-backend:latest
docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/xray-backend:latest
```

#### 4. Deploy to ECS

1. Create ECS cluster
2. Create task definition (use the pushed image)
3. Create service
4. Configure load balancer

### Docker Compose (Simple Deployment)

```bash
# Clone repository on server
git clone https://github.com/yourusername/xray-healthcare-ai.git
cd xray-healthcare-ai

# Copy and configure environment
cp .env.example .env
nano .env  # Edit configuration

# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Troubleshooting

### Backend Issues

**1. ModuleNotFoundError**
```bash
# Ensure virtual environment is activated
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**2. CUDA Out of Memory**
```python
# Edit training/train.py
config['batch_size'] = 4  # Reduce from 8
```

**3. Model Loading Error**
```bash
# Check model path
ls -la checkpoints/best_model.pth

# Verify in .env
echo $MODEL_PATH
```

### Frontend Issues

**1. npm install fails**
```bash
# Clear cache
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

**2. CORS Error**
```python
# Check backend CORS settings in app/main.py
# Ensure allow_origins includes frontend URL
```

**3. API Connection Error**
```javascript
// Check API URL in frontend
console.log(process.env.REACT_APP_API_URL)

// Update .env
REACT_APP_API_URL=http://localhost:8000
```

### Training Issues

**1. Dataset Not Found**
```bash
# Check directory structure
ls -la data/train
ls -la data/val

# Ensure images exist
find data/train -type f | wc -l
```

**2. Low GPU Utilization**
```python
# Increase batch size
config['batch_size'] = 16  # If GPU has enough memory

# Increase num_workers
config['num_workers'] = 8
```

**3. Training Too Slow**
```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# In training loop:
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Docker Issues

**1. Container Won't Start**
```bash
# Check logs
docker-compose logs backend

# Rebuild
docker-compose build --no-cache
docker-compose up
```

**2. Port Already in Use**
```bash
# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use different host port
```

---

## Performance Optimization

### Training Optimization

1. **Use Mixed Precision**:
```python
from torch.cuda.amp import autocast, GradScaler
```

2. **Data Loading**:
```python
config['num_workers'] = 4  # Adjust based on CPU cores
config['pin_memory'] = True  # For GPU training
```

3. **Gradient Accumulation**:
```python
# For small GPU memory
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Inference Optimization

1. **Model Quantization**:
```python
import torch
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

2. **TorchScript**:
```python
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")
```

3. **ONNX Export**:
```python
torch.onnx.export(model, dummy_input, "model.onnx")
```

---

## Next Steps

After successful setup:

1. **Test the Application**: Upload sample X-ray images
2. **Train Your Model**: Use your own dataset
3. **Customize UI**: Modify React components
4. **Deploy to Cloud**: Follow deployment guides
5. **Monitor Performance**: Use logging and metrics

For more help, refer to:
- [README.md](README.md) - Main documentation
- [API Documentation](http://localhost:8000/docs) - API reference
- [GitHub Issues](https://github.com/yourusername/xray-healthcare-ai/issues) - Report problems

---

**Happy coding! 🚀**
