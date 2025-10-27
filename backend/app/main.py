"""
FastAPI Backend for X-ray Enhancement AI
=========================================
Main application file with API endpoints for image enhancement and chatbot.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import io
import base64
import numpy as np
from PIL import Image
import torch
import cv2
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.app.utils.image_processor import ImageProcessor
from backend.app.utils.chatbot import HealthcareChatbot
from backend.app.models.model_loader import ModelLoader

# Initialize FastAPI app
app = FastAPI(
    title="X-ray Enhancement AI API",
    description="API for X-ray image enhancement and healthcare chatbot",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
model_loader = None
image_processor = None
chatbot = None


@app.on_event("startup")
async def startup_event():
    """
    Initialize models and processors on startup.
    """
    global model_loader, image_processor, chatbot

    print("Initializing backend services...")

    # Load model
    model_loader = ModelLoader()
    await model_loader.load_model()

    # Initialize image processor
    image_processor = ImageProcessor(model_loader.model, model_loader.device)

    # Initialize chatbot
    chatbot = HealthcareChatbot()

    print("Backend services initialized successfully!")


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "X-ray Enhancement AI API",
        "version": "1.0.0",
        "endpoints": {
            "enhance": "/enhance - POST - Enhance X-ray images",
            "chatbot": "/chatbot - POST - Healthcare chatbot interaction",
            "health": "/health - GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "model_loaded": model_loader is not None and model_loader.model is not None,
        "device": str(model_loader.device) if model_loader else "unknown"
    }


class EnhanceRequest(BaseModel):
    """
    Request model for image enhancement.
    """
    image: str  # Base64 encoded image
    mask: Optional[str] = None  # Optional base64 encoded mask
    return_attention: bool = False  # Whether to return attention maps


class EnhanceResponse(BaseModel):
    """
    Response model for image enhancement.
    """
    enhanced_image: str  # Base64 encoded enhanced image
    metrics: Dict[str, float]
    attention_maps: Optional[Dict[str, str]] = None  # Base64 encoded attention maps


@app.post("/enhance", response_model=EnhanceResponse)
async def enhance_image(file: UploadFile = File(...)):
    """
    Enhance X-ray image using the trained model.

    Args:
        file: Uploaded image file

    Returns:
        Enhanced image with metrics and optional attention maps
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')

        # Process image
        result = await image_processor.enhance_image(
            image,
            return_attention=True
        )

        # Convert enhanced image to base64
        enhanced_img_array = result['enhanced']
        enhanced_pil = Image.fromarray((enhanced_img_array * 255).astype(np.uint8))
        buffered = io.BytesIO()
        enhanced_pil.save(buffered, format="PNG")
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Convert attention maps to base64
        attention_maps_base64 = {}
        if result.get('attention_maps'):
            for key, att_map in result['attention_maps'].items():
                # Normalize attention map to 0-255
                att_normalized = (att_map * 255).astype(np.uint8)
                att_pil = Image.fromarray(att_normalized)
                att_buffered = io.BytesIO()
                att_pil.save(att_buffered, format="PNG")
                attention_maps_base64[key] = base64.b64encode(att_buffered.getvalue()).decode()

        return EnhanceResponse(
            enhanced_image=enhanced_base64,
            metrics=result['metrics'],
            attention_maps=attention_maps_base64 if attention_maps_base64 else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/enhance-url")
async def enhance_image_url(request: EnhanceRequest):
    """
    Enhance X-ray image from base64 encoded data.

    Alternative endpoint for base64 encoded images.
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))

        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')

        # Process image
        result = await image_processor.enhance_image(
            image,
            return_attention=request.return_attention
        )

        # Convert enhanced image to base64
        enhanced_img_array = result['enhanced']
        enhanced_pil = Image.fromarray((enhanced_img_array * 255).astype(np.uint8))
        buffered = io.BytesIO()
        enhanced_pil.save(buffered, format="PNG")
        enhanced_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Convert attention maps to base64 if requested
        attention_maps_base64 = None
        if request.return_attention and result.get('attention_maps'):
            attention_maps_base64 = {}
            for key, att_map in result['attention_maps'].items():
                att_normalized = (att_map * 255).astype(np.uint8)
                att_pil = Image.fromarray(att_normalized)
                att_buffered = io.BytesIO()
                att_pil.save(att_buffered, format="PNG")
                attention_maps_base64[key] = base64.b64encode(att_buffered.getvalue()).decode()

        return {
            "enhanced_image": enhanced_base64,
            "metrics": result['metrics'],
            "attention_maps": attention_maps_base64
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


class ChatRequest(BaseModel):
    """
    Request model for chatbot interaction.
    """
    message: str
    conversation_history: Optional[list] = None


class ChatResponse(BaseModel):
    """
    Response model for chatbot interaction.
    """
    response: str
    conversation_id: Optional[str] = None


@app.post("/chatbot", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Healthcare chatbot endpoint.

    Handles medical questions, explains X-ray findings, and provides guidance.

    Args:
        request: ChatRequest with user message and optional conversation history

    Returns:
        Chatbot response
    """
    try:
        response = await chatbot.get_response(
            message=request.message,
            conversation_history=request.conversation_history
        )

        return ChatResponse(
            response=response['message'],
            conversation_id=response.get('conversation_id')
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chatbot: {str(e)}")


@app.post("/chatbot/explain-enhancement")
async def explain_enhancement(metrics: Dict[str, float]):
    """
    Specialized endpoint to explain enhancement results.

    Args:
        metrics: Dictionary of metric values (PSNR, SSIM, etc.)

    Returns:
        Natural language explanation of the enhancement
    """
    try:
        explanation = await chatbot.explain_metrics(metrics)
        return {"explanation": explanation}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error explaining metrics: {str(e)}")


@app.get("/model/info")
async def model_info():
    """
    Get information about the loaded model.
    """
    if model_loader and model_loader.model:
        num_params = sum(p.numel() for p in model_loader.model.generator.parameters())
        return {
            "model_type": "Pix2Pix GAN with Attention UNet",
            "parameters": num_params,
            "device": str(model_loader.device),
            "input_size": "256x256",
            "output_channels": 1
        }
    else:
        return {"error": "Model not loaded"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
