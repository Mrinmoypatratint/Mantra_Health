"""
WSGI/ASGI wrapper for PythonAnywhere deployment
"""
import sys
import os

# Add your project directory to the sys.path
project_home = '/home/yourusername/xray-healthcare-ai'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Set environment variables
os.environ['MODEL_PATH'] = os.path.join(project_home, 'checkpoints', 'best_model.pth')

# Import FastAPI app
from backend.app.main import app

# For PythonAnywhere with ASGI support (paid plans)
application = app

# For PythonAnywhere free tier (WSGI only) - use a2wsgi adapter
try:
    from a2wsgi import ASGIMiddleware
    application = ASGIMiddleware(app)
except ImportError:
    # If a2wsgi not available, use the ASGI app directly
    # Note: This requires PythonAnywhere's ASGI support
    application = app
