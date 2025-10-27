"""
Vercel Serverless Function Entry Point
This wraps the FastAPI app for Vercel deployment
"""
from backend.app.main import app

# Vercel expects a handler function
def handler(request, context):
    return app(request, context)
