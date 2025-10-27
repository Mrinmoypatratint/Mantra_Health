# X-ray Enhancement AI

Advanced medical image enhancement system using **UNet + Attention Mechanism + GAN (Pix2Pix)** with a full-stack web application for real-time X-ray image enhancement and healthcare chatbot assistance.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![React](https://img.shields.io/badge/React-18.2-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the Backend](#running-the-backend)
  - [Running the Frontend](#running-the-frontend)
- [Dataset Preparation](#dataset-preparation)
- [Model Architecture](#model-architecture)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Core Features
- **Advanced Image Enhancement**: UNet architecture with attention gates for superior X-ray enhancement
- **GAN-based Training**: Pix2Pix GAN for realistic texture generation
- **Quality Metrics**: Real-time PSNR, SSIM, and LPIPS calculation
- **Attention Visualization**: Interactive attention map visualization
- **Healthcare Chatbot**: AI-powered medical assistant for Q&A and result explanation

### Web Application
- **Drag-and-Drop Upload**: Easy image upload interface
- **Before/After Comparison**: Interactive slider for comparing results
- **Real-time Processing**: Fast image enhancement with progress tracking
- **Responsive Design**: Modern UI built with React and Tailwind CSS
- **Floating Chat Widget**: Integrated healthcare assistant

---

## Architecture

### Model Architecture

```
Input X-ray (256x256) → [Encoder] → Bottleneck → [Decoder + Attention] → Enhanced X-ray
                                                           ↓
                                                    [Discriminator]
```

**Components:**
1. **Generator**: UNet with Attention Gates
   - Encoder: 4 downsampling blocks (64, 128, 256, 512)
   - Bottleneck: 1024 channels
   - Decoder: 4 upsampling blocks with attention gates
   - Attention: Focus on relevant features at each scale

2. **Discriminator**: PatchGAN
   - 70x70 patch classification
   - Encourages high-frequency detail preservation

3. **Loss Functions**:
   - Adversarial Loss (BCE/LSGAN)
   - L1 Loss (λ = 100)
   - Perceptual Loss (optional)

### Tech Stack

**Backend:**
- PyTorch 2.6.0
- FastAPI
- OpenCV
- scikit-image

**Frontend:**
- React 18.2
- Tailwind CSS
- Framer Motion
- Axios

**Deployment:**
- Docker & Docker Compose
- Nginx
- Google Cloud Run / AWS ECS

---

## Installation

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA 11.8+ (for GPU training)
- Docker & Docker Compose (optional)

### Local Setup

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/xray-healthcare-ai.git
cd xray-healthcare-ai
```

#### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env
```

#### 4. Environment Configuration

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
OPENAI_API_KEY=your_openai_api_key_here  # Optional
MODEL_PATH=./checkpoints/best_model.pth
```

### Docker Setup

```bash
# Build and run all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Usage

### Training the Model

#### 1. Prepare Dataset

Download datasets (NIH ChestX-ray14, Kaggle Pneumonia, etc.) and organize:

```
data/
├── train/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── val/
    ├── image1.png
    ├── image2.png
    └── ...
```

#### 2. Train the Model

```bash
cd training
python train.py
```

**Training Configuration** (in `train.py`):

```python
config = {
    'train_dir': './data/train',
    'val_dir': './data/val',
    'img_size': 256,
    'batch_size': 8,
    'num_epochs': 150,
    'lr': 2e-4,
    'lambda_L1': 100.0,
}
```

#### 3. Monitor Training

```bash
# View TensorBoard logs
tensorboard --logdir=./logs
```

### Running the Backend

```bash
cd backend

# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### Running the Frontend

```bash
cd frontend

# Development mode
npm start

# Production build
npm run build
npm install -g serve
serve -s build
```

Frontend will be available at: `http://localhost:3000`

---

## Dataset Preparation

### Recommended Datasets

1. **NIH ChestX-ray14**
   - 112,120 frontal-view X-ray images
   - Download: [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)

2. **Kaggle Chest X-ray Pneumonia**
   - 5,856 images (train + test)
   - Download: [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

3. **RSNA Pneumonia Detection**
   - 26,684 images with annotations
   - Download: [Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)

### Data Preprocessing

Images are automatically preprocessed during training:

1. **Resize**: 256×256 pixels
2. **Grayscale conversion**
3. **Normalization**: [0, 1] range
4. **Degradation**: Simulated noise, blur, and contrast reduction
5. **Augmentation**: Flip, rotate, brightness/contrast adjustment

---

## Model Architecture

### Generator: Attention UNet

```python
AttentionUNet(
  in_channels=1,
  out_channels=1
)

# Encoder
Conv1: 1 → 64
Conv2: 64 → 128
Conv3: 128 → 256
Conv4: 256 → 512
Conv5: 512 → 1024

# Decoder with Attention
Up5 + Att5: 1024 + 512 → 512
Up4 + Att4: 512 + 256 → 256
Up3 + Att3: 256 + 128 → 128
Up2 + Att2: 128 + 64 → 64

# Output
Conv_1x1: 64 → 1
```

**Parameters**: ~34M

### Discriminator: PatchGAN

```python
PatchGANDiscriminator(
  in_channels=2,  # input + output
  features=64
)

# Architecture
C64 → C128 → C256 → C512 → 1-channel prediction
```

**Parameters**: ~2.7M

---

## API Documentation

### Endpoints

#### 1. **POST** `/enhance`

Enhance X-ray image.

**Request:**
- **Type**: `multipart/form-data`
- **Body**:
  - `file`: Image file (PNG, JPG, JPEG)

**Response:**
```json
{
  "enhanced_image": "base64_encoded_image",
  "metrics": {
    "psnr": 28.45,
    "ssim": 0.8532
  },
  "attention_maps": {
    "layer1": "base64_encoded_image",
    "layer2": "base64_encoded_image",
    ...
  }
}
```

#### 2. **POST** `/chatbot`

Healthcare chatbot interaction.

**Request:**
```json
{
  "message": "What is PSNR?",
  "conversation_history": []
}
```

**Response:**
```json
{
  "response": "PSNR (Peak Signal-to-Noise Ratio) measures...",
  "conversation_id": "abc123"
}
```

#### 3. **GET** `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

Full API documentation available at: `http://localhost:8000/docs`

---

## Deployment

### Google Cloud Run

```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/xray-backend

# Deploy
gcloud run deploy xray-backend \
  --image gcr.io/PROJECT_ID/xray-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com

docker build -t xray-backend ./backend
docker tag xray-backend:latest ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/xray-backend:latest
docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/xray-backend:latest

# Deploy to ECS (configure task definition)
```

### Frontend Deployment (Netlify/Vercel)

```bash
# Netlify
cd frontend
npm run build
netlify deploy --prod --dir=build

# Vercel
vercel --prod
```

---

## Project Structure

```
xray-healthcare-ai/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application
│   │   ├── models/
│   │   │   └── model_loader.py     # Model loading utilities
│   │   └── utils/
│   │       ├── image_processor.py  # Image processing
│   │       └── chatbot.py          # Chatbot implementation
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.js                  # Main app component
│   │   ├── components/
│   │   │   ├── ImageUpload.js
│   │   │   ├── ImageComparison.js
│   │   │   ├── MetricsDisplay.js
│   │   │   ├── ChatWidget.js
│   │   │   └── LoadingSpinner.js
│   │   └── index.css
│   ├── package.json
│   ├── Dockerfile
│   └── nginx.conf
├── models/
│   ├── attention_unet.py           # UNet with attention
│   └── gan.py                      # Pix2Pix GAN
├── training/
│   ├── dataset.py                  # Dataset and preprocessing
│   ├── train.py                    # Training script
│   └── metrics.py                  # Evaluation metrics
├── checkpoints/                    # Saved models
├── data/                           # Training data
│   ├── train/
│   └── val/
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Performance Metrics

Expected performance after 150 epochs of training:

| Metric | Value | Description |
|--------|-------|-------------|
| **PSNR** | 25-30 dB | Peak Signal-to-Noise Ratio |
| **SSIM** | 0.80-0.90 | Structural Similarity Index |
| **LPIPS** | 0.10-0.20 | Learned Perceptual Similarity |

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size in training config
config['batch_size'] = 4
```

**2. Model Not Loading**
```bash
# Check model path
export MODEL_PATH=./checkpoints/best_model.pth
```

**3. Frontend API Connection Error**
```javascript
// Update API URL in frontend
const API_URL = 'http://localhost:8000';
```

**4. Slow Inference**
- Ensure GPU is available: `torch.cuda.is_available()`
- Reduce image size in preprocessing
- Use quantized models for production

---

## Future Enhancements

- [ ] Support for DICOM format
- [ ] Multi-organ X-ray enhancement
- [ ] Real-time video enhancement
- [ ] Model quantization for mobile deployment
- [ ] Integration with PACS systems
- [ ] Multi-language chatbot support
- [ ] Advanced visualization tools

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **NIH Clinical Center** for ChestX-ray14 dataset
- **Kaggle** community for datasets and kernels
- **PyTorch** team for the deep learning framework
- **FastAPI** for the excellent web framework

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{xray_enhancement_ai,
  title={X-ray Enhancement AI: Advanced Medical Image Enhancement with Deep Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/xray-healthcare-ai}
}
```

---

## Contact

For questions, issues, or collaboration:

- **Email**: your.email@example.com
- **GitHub Issues**: [Create an issue](https://github.com/yourusername/xray-healthcare-ai/issues)
- **Twitter**: [@yourusername](https://twitter.com/yourusername)

---

## Disclaimer

⚠️ **Important**: This project is for **research and educational purposes only**. It is not intended for clinical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice and diagnosis.

---

**Built with ❤️ for advancing medical imaging technology**
