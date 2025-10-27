"""
Vercel-optimized FastAPI app
Lightweight version without heavy ML dependencies
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="X-ray Enhancement AI API (Vercel)",
    description="Lightweight API for Vercel deployment",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@app.get("/api")
async def root():
    """Root endpoint"""
    return {
        "message": "X-ray Enhancement AI API (Vercel Deployment)",
        "version": "1.0.0",
        "status": "running",
        "note": "Running in lightweight mode on Vercel",
        "endpoints": {
            "health": "/api/health - GET - Health check",
            "enhance": "/api/enhance - POST - Enhance X-ray images (limited)",
            "docs": "/api/docs - GET - API documentation"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "platform": "Vercel Serverless",
        "mode": "lightweight",
        "model_loaded": False,
        "note": "Full ML model not available in serverless mode. Use PythonAnywhere for full functionality."
    }

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[list] = []

@app.post("/api/chatbot")
async def chatbot(request: ChatRequest):
    """Basic chatbot endpoint"""
    return {
        "response": "Chatbot functionality requires external ML service. Please deploy backend to PythonAnywhere for full AI features.",
        "status": "limited"
    }

@app.post("/api/enhance")
async def enhance_image(file: UploadFile = File(...)):
    """
    Basic image enhancement endpoint (limited functionality)
    For full ML enhancement, deploy to PythonAnywhere
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')

        # Apply basic enhancement (no ML model)
        # This is just histogram equalization as a placeholder
        import numpy as np
        img_array = np.array(image)

        # Simple contrast enhancement
        img_normalized = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        img_enhanced = (img_normalized * 255).astype(np.uint8)

        # Convert back to image
        enhanced_image = Image.fromarray(img_enhanced, mode='L')

        # Convert to base64
        buffered = BytesIO()
        enhanced_image.save(buffered, format="PNG")
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "success": True,
            "enhanced_image": f"data:image/png;base64,{enhanced_base64}",
            "note": "Basic enhancement applied. For AI-powered enhancement, deploy to PythonAnywhere.",
            "metrics": {
                "psnr": 0,
                "ssim": 0,
                "note": "Metrics not available in lightweight mode"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Vercel serverless handler
app_handler = app
