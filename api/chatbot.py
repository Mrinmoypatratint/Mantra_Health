"""
Chatbot endpoint for Vercel
"""
from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Read request body
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode())

            message = data.get('message', '')

            # Generate a simple response (no AI model)
            responses = {
                'hello': 'Hello! I\'m here to help with X-ray analysis questions.',
                'help': 'I can help you understand X-ray enhancement results and medical imaging concepts.',
                'what': 'This platform uses AI to enhance X-ray images for better diagnostic clarity.',
                'how': 'The system analyzes your X-ray images and applies enhancement algorithms to improve visibility.',
            }

            # Simple keyword matching
            response_text = "I'm a basic chatbot running on Vercel. For advanced AI features, please deploy the full backend to PythonAnywhere with OpenAI integration."

            for keyword, reply in responses.items():
                if keyword in message.lower():
                    response_text = reply
                    break

            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            response = {
                "response": response_text,
                "status": "success"
            }

            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            error_response = {
                "response": "Sorry, I encountered an error. Please try again.",
                "status": "error",
                "error": str(e)
            }

            self.wfile.write(json.dumps(error_response).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
