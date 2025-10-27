# 🎓 Complete Training Guide

You have **TWO OPTIONS** to train your X-ray Enhancement AI model:

---

## 🔵 Option 1: Google Colab (Recommended - Free GPU!)

### Why Colab?
- ✅ Free GPU access (Tesla T4)
- ✅ No local storage needed
- ✅ Training takes 3-5 hours instead of 10-20 hours
- ✅ No need to download large datasets locally

### Step-by-Step Instructions:

#### 1. Upload Notebook to Colab
1. Go to: https://colab.research.google.com/
2. Click **File → Upload notebook**
3. Upload this file: `notebooks/training_demo.ipynb` from your project

#### 2. Enable GPU
1. In Colab, click **Runtime → Change runtime type**
2. Select **GPU** from the Hardware accelerator dropdown
3. Click **Save**

#### 3. Follow the Notebook Cells

The notebook will guide you through:
- ✅ Installing dependencies
- ✅ Downloading the Kaggle dataset (or uploading your own)
- ✅ Training the model
- ✅ Monitoring with TensorBoard
- ✅ Downloading the trained model

#### 4. Download Trained Model

After training completes:
1. The notebook will download `best_model.pth` to your computer
2. Place it here: `xray-healthcare-ai/checkpoints/best_model.pth`
3. Run your local application with `start.bat`

**Total time: 3-5 hours (mostly waiting for training)**

---

## 💻 Option 2: Local Training with Sample Data (Quick Test)

### For Quick Testing Without Full Dataset

I can help you download a few sample X-ray images to test the training process locally.

### Steps:

#### 1. Get Sample Images

**Option A: Download from Open Medical Datasets**

Sample X-ray images from NIH (public domain):
```bash
# I'll help you download ~10-20 sample images
# This is enough to test training but won't give good results
```

**Option B: Use Any Chest X-ray Images**

You can use any chest X-ray images you have. Just place them in:
```
xray-healthcare-ai/
  data/
    train/    ← Put 10+ images here
    val/      ← Put 2-5 images here
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.dcm`

#### 2. Run Training

Once you have images in the data folders:

```bash
cd training
python train.py
```

#### 3. Monitor Training

In another terminal:
```bash
tensorboard --logdir=../logs
# Open: http://localhost:6006
```

**Note:** Training with only 10-20 images won't produce good enhancement results, but it will help you test the pipeline!

---

## 📊 Expected Training Times

| Hardware | Time (100 epochs) | Result Quality |
|----------|-------------------|----------------|
| **Google Colab (T4 GPU)** | 3-5 hours | ⭐⭐⭐⭐⭐ Excellent |
| **Local GPU (RTX 3060)** | 2-4 hours | ⭐⭐⭐⭐⭐ Excellent |
| **Local CPU** | 10-20 hours | ⭐⭐⭐⭐⭐ Excellent |
| **Sample Data (10 images)** | 10-30 minutes | ⭐ Poor (test only) |

---

## 🎯 Full Dataset Option (Best Results)

### Option A: Kaggle Chest X-ray Pneumonia (~2GB)

**If you have a Kaggle account:**

1. Get your API token:
   - Go to https://www.kaggle.com/
   - Click your profile → Settings → API → "Create New Token"
   - Download `kaggle.json`

2. Install and configure:
```bash
# Install Kaggle CLI
pip install kaggle

# Windows - Place kaggle.json in:
C:\Users\YOUR_USERNAME\.kaggle\kaggle.json

# Download dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

# Extract
unzip chest-xray-pneumonia.zip -d data/
```

3. Organize:
```bash
# Move images to train/val folders
# The dataset comes pre-organized, just move to our structure
```

### Option B: NIH ChestX-ray14 (~42GB)

1. Visit: https://nihcc.app.box.com/v/ChestXray-NIHCC
2. Download images (select subset or all)
3. Extract and organize into `data/train/` and `data/val/`

---

## 🔧 Training Configuration

Default settings (in `training/train.py`):

```python
config = {
    'batch_size': 8,           # Reduce if out of memory
    'num_epochs': 150,         # More epochs = better results
    'learning_rate': 0.0002,
    'lambda_L1': 100,          # Weight for pixel loss
    'img_size': 256,
    'num_workers': 4,
}
```

**Adjust for your hardware:**
- Less RAM/VRAM: `batch_size = 4`
- More GPU memory: `batch_size = 16`
- Quick test: `num_epochs = 10`

---

## 📈 What to Expect

### During Training:
- Loss values decreasing
- PSNR increasing (target: 25-30 dB)
- SSIM increasing (target: 0.80-0.90)

### After Training:
- Model saved to `checkpoints/best_model.pth`
- Logs saved to `logs/` (view with TensorBoard)
- Enhancement quality should be significantly improved

---

## 🆘 Troubleshooting

### Out of Memory Error
```python
# In training/train.py, reduce batch size:
config['batch_size'] = 4  # or even 2
```

### GPU Not Detected
```python
import torch
print(torch.cuda.is_available())  # Should be True

# If False, reinstall PyTorch with CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Training Too Slow
- Use Google Colab with GPU
- Or reduce `num_epochs` for testing

### Poor Results
- Need more training data (100+ images minimum)
- Train for more epochs (150-200)
- Use full dataset instead of samples

---

## ✅ After Training

Once training completes:

1. **Verify model exists:**
   ```bash
   dir checkpoints\best_model.pth
   ```

2. **Start the application:**
   ```bash
   start.bat
   ```

3. **Test enhancement:**
   - Upload an X-ray image
   - See improved enhancement quality
   - Check metrics (PSNR, SSIM)

---

## 🎓 My Recommendation

**For best experience:**

1. **Start with Google Colab** (Option 1)
   - Free GPU, faster training
   - Follow the notebook step-by-step
   - 3-5 hours to completion

2. **Download trained model**
   - Save `best_model.pth` to `checkpoints/`

3. **Run locally**
   - `start.bat`
   - Enjoy high-quality enhancement!

**For quick testing:**
- Use 10-20 sample images locally
- Just to see the training process
- Not for production-quality results

---

## 📞 Need Help?

Refer to these guides:
- `START_HERE.md` - Overview
- `NEXT_STEPS.md` - Detailed steps
- `README.md` - Full documentation

Happy training! 🚀
