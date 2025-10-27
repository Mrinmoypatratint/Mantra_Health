# Railway Backend Deployment Guide

## Prerequisites
- GitHub account
- Railway account (sign up at https://railway.app)

## Step-by-Step Deployment

### 1. Push Your Code to GitHub

First, commit and push your backend changes:

```bash
cd xray-healthcare-ai
git add .
git commit -m "Add Railway deployment configuration"
git push origin master
```

### 2. Deploy on Railway

1. Go to https://railway.app and sign in with GitHub
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository: `xray-healthcare-ai`
5. Railway will auto-detect your configuration

### 3. Configure Environment Variables

In Railway dashboard, go to your project → Variables tab and add:

```
MODEL_PATH=/app/checkpoints/best_model.pth
OPENAI_API_KEY=your_openai_key_here (optional)
LOG_LEVEL=INFO
BACKEND_HOST=0.0.0.0
BACKEND_PORT=$PORT
```

### 4. Set Root Directory

1. In Settings → Service Settings
2. Set "Root Directory" to: `backend`
3. This tells Railway to deploy only the backend folder

### 5. Deploy

Railway will automatically:
- Install dependencies from `requirements-railway.txt`
- Build your backend
- Deploy to a public URL

### 6. Get Your Backend URL

After deployment completes:
- Your backend URL will be: `https://your-project.railway.app`
- Test it at: `https://your-project.up.railway.app/health`

### 7. Update Frontend

Copy your Railway backend URL and update the frontend:
- Go to Vercel dashboard
- Add environment variable: `REACT_APP_API_URL=<your-railway-url>`
- Redeploy frontend

## Important Notes

- Railway free tier includes $5/month credit
- Backend uses CPU-only PyTorch for better compatibility
- Model checkpoint is optional (uses random weights if missing)
- First deployment may take 5-10 minutes

## Troubleshooting

### Build Fails
- Check Railway build logs
- Verify `requirements-railway.txt` is in backend folder

### App Crashes
- Check Railway runtime logs
- Verify environment variables are set
- Ensure PORT is using $PORT variable

### Memory Issues
- Railway free tier has 512MB-1GB RAM
- Consider upgrading plan if needed for larger models
