# PythonAnywhere Deployment Guide

## Complete Guide to Deploy FastAPI Backend on PythonAnywhere

### Prerequisites
- PythonAnywhere account (free tier works, but paid tier recommended for ASGI)
- GitHub repository with your code

---

## Option 1: Automated Setup (Recommended)

### Step 1: Sign up for PythonAnywhere
1. Go to https://www.pythonanywhere.com
2. Sign up for a **free account** (or paid for better performance)
3. Verify your email and log in

### Step 2: Open Bash Console
1. From PythonAnywhere dashboard, go to **"Consoles"** tab
2. Click **"Bash"** to open a new Bash console

### Step 3: Run Setup Script
```bash
# Clone the repository
cd ~
git clone https://github.com/Mrinmoypatratint/Mantra_Health.git xray-healthcare-ai

# Run setup script
cd xray-healthcare-ai/backend
chmod +x setup_pythonanywhere.sh
bash setup_pythonanywhere.sh
```

### Step 4: Configure Web App
1. Go to **"Web"** tab in PythonAnywhere
2. Click **"Add a new web app"**
3. Choose **"Manual configuration"** (not "Web framework")
4. Select **Python 3.10**

### Step 5: Configure WSGI/ASGI File
1. In the Web tab, scroll to **"Code"** section
2. Click on the **WSGI configuration file** link
3. **Delete all contents** and replace with:

```python
import sys
import os

# Add your project directory to the sys.path
username = 'YOUR_USERNAME'  # Replace with your PythonAnywhere username
project_home = f'/home/{username}/xray-healthcare-ai'

if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Set environment variables
os.environ['MODEL_PATH'] = f'{project_home}/checkpoints/best_model.pth'
os.environ['LOG_LEVEL'] = 'INFO'

# Import FastAPI app
from backend.app.main import app

# Use ASGI to WSGI adapter for free tier
from a2wsgi import ASGIMiddleware
application = ASGIMiddleware(app)
```

**Important:** Replace `YOUR_USERNAME` with your actual PythonAnywhere username!

### Step 6: Configure Virtual Environment
1. In the Web tab, scroll to **"Virtualenv"** section
2. Enter the path: `/home/YOUR_USERNAME/xray-healthcare-ai/backend/venv`
   (Replace YOUR_USERNAME with your actual username)

### Step 7: Set Static Files (Optional)
This is not needed for the backend API, but if you want to serve files:
- URL: `/static/`
- Directory: `/home/YOUR_USERNAME/xray-healthcare-ai/backend/static/`

### Step 8: Enable HTTPS and CORS
1. Scroll to **"Security"** section
2. Enable **"Force HTTPS"** (recommended)

### Step 9: Reload Web App
1. Scroll to the top of the Web tab
2. Click the green **"Reload"** button
3. Wait 30-60 seconds for the app to restart

### Step 10: Test Your Deployment
Your API will be available at:
```
https://YOUR_USERNAME.pythonanywhere.com/
```

Test endpoints:
- Health check: `https://YOUR_USERNAME.pythonanywhere.com/health`
- API docs: `https://YOUR_USERNAME.pythonanywhere.com/docs`

---

## Option 2: Manual Setup (Step by Step)

### 1. Open Bash Console
From PythonAnywhere dashboard → Consoles → Bash

### 2. Clone Repository
```bash
cd ~
git clone https://github.com/Mrinmoypatratint/Mantra_Health.git xray-healthcare-ai
cd xray-healthcare-ai/backend
```

### 3. Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements-pythonanywhere.txt
```

If you encounter errors, install one by one:
```bash
pip install fastapi uvicorn python-multipart a2wsgi
pip install opencv-python-headless Pillow numpy
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu
```

### 5. Create Directories
```bash
mkdir -p ~/xray-healthcare-ai/checkpoints
mkdir -p ~/xray-healthcare-ai/logs
```

### 6. Follow Steps 4-10 from Option 1

---

## Important Notes

### Free Tier Limitations
- **512MB RAM** - May struggle with large PyTorch models
- **CPU only** - No GPU acceleration
- **100 second timeout** - Long-running requests will timeout
- **Daily limit** - Limited CPU seconds per day

### Paid Tier Benefits ($5/month)
- **1GB+ RAM** - Better for ML models
- **Longer timeouts** - Up to 5 minutes
- **More CPU time** - Better performance
- **Always-on tasks** - Can run background processes

### Model Considerations
Since the model checkpoint might be large:
1. **Without model:** App will work but use random weights (for testing)
2. **With model:** Upload via:
   ```bash
   # In PythonAnywhere Bash console
   cd ~/xray-healthcare-ai/checkpoints
   wget YOUR_MODEL_URL
   # Or upload via Files tab (max 100MB on free tier)
   ```

---

## Troubleshooting

### Error: "Could not import app"
- Check that the WSGI file has the correct username
- Verify virtualenv path is correct
- Check error logs in Web tab → Error log

### Error: "Module not found"
```bash
# Reinstall in virtual environment
source ~/xray-healthcare-ai/backend/venv/bin/activate
pip install -r requirements-pythonanywhere.txt
```

### Memory Errors
- Free tier has limited RAM
- Consider using a lighter PyTorch version
- Upgrade to paid tier for more memory

### Import Errors
```bash
# Check if modules are installed
source ~/xray-healthcare-ai/backend/venv/bin/activate
pip list | grep fastapi
pip list | grep torch
```

### CORS Issues
The backend has CORS enabled for all origins. To restrict:
1. Edit `backend/app/main.py`
2. Change `allow_origins=["*"]` to your frontend URL
3. Reload web app

---

## Environment Variables

Set in the WSGI configuration file:
```python
os.environ['MODEL_PATH'] = '/home/username/xray-healthcare-ai/checkpoints/best_model.pth'
os.environ['OPENAI_API_KEY'] = 'your-key-here'  # Optional
os.environ['LOG_LEVEL'] = 'INFO'
```

---

## Updating Your App

### Pull Latest Changes
```bash
cd ~/xray-healthcare-ai
git pull origin main
source backend/venv/bin/activate
pip install -r backend/requirements-pythonanywhere.txt --upgrade
```

### Reload Web App
Go to Web tab → Click "Reload"

---

## Check Logs

### Error Log
Web tab → Links section → Error log

### Server Log
Web tab → Links section → Server log

### Access Log
Web tab → Links section → Access log

---

## Next Steps

After deployment:
1. Get your backend URL: `https://YOUR_USERNAME.pythonanywhere.com`
2. Test it: Visit `/health` endpoint
3. Update your frontend with this URL
4. Redeploy frontend to Vercel

---

## Cost Comparison

| Plan | Price | RAM | Timeout | Best For |
|------|-------|-----|---------|----------|
| Free | $0 | 512MB | 100s | Testing, demos |
| Hacker | $5/mo | 1GB | 300s | Small projects |
| Web Dev | $12/mo | 2GB | 300s | Production apps |

**Recommendation:** Start with free tier for testing, upgrade to Hacker ($5/mo) if you need more resources.
