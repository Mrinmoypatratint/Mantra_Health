"""
Healthcare Chatbot Module
==========================
Provides medical Q&A and explanation capabilities using LLMs.
"""

import os
from typing import Optional, List, Dict
import json


class HealthcareChatbot:
    """
    Healthcare assistant chatbot for medical questions and X-ray explanations.

    Supports:
        - Medical Q&A
        - X-ray enhancement result explanation
        - General healthcare guidance

    Can use either:
        - OpenAI API (GPT-4/GPT-3.5)
        - Local models (DistilGPT2, BioBERT, etc.)
    """
    def __init__(self, use_openai=True, api_key=None):
        self.use_openai = use_openai
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.conversation_history = {}

        # Initialize the appropriate backend
        if self.use_openai and self.api_key:
            self._init_openai()
        else:
            self._init_local_model()

    def _init_openai(self):
        """
        Initialize OpenAI API client.
        """
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=self.api_key)
            self.model_name = "gpt-3.5-turbo"
            print("Initialized OpenAI chatbot")
        except ImportError:
            print("OpenAI library not installed. Install with: pip install openai")
            self._init_local_model()
        except Exception as e:
            print(f"Error initializing OpenAI: {e}")
            self._init_local_model()

    def _init_local_model(self):
        """
        Initialize local transformer model.
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            print("Initializing local chatbot model...")
            model_name = "microsoft/DialoGPT-medium"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)

            print(f"Initialized local chatbot model: {model_name}")
            self.use_openai = False

        except Exception as e:
            print(f"Error initializing local model: {e}")
            print("Using fallback rule-based responses")
            self.model = None
            self.use_openai = False

    async def get_response(self, message: str, conversation_history: Optional[List] = None):
        """
        Get chatbot response to user message.

        Args:
            message: User message
            conversation_history: Optional conversation history

        Returns:
            Dictionary with response and conversation_id
        """
        if self.use_openai and hasattr(self, 'client'):
            return await self._get_openai_response(message, conversation_history)
        elif hasattr(self, 'model') and self.model is not None:
            return await self._get_local_response(message, conversation_history)
        else:
            return await self._get_fallback_response(message)

    async def _get_openai_response(self, message: str, conversation_history: Optional[List] = None):
        """
        Get response using OpenAI API.
        """
        try:
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful medical assistant specialized in radiology and X-ray imaging.
                    You provide accurate information about medical imaging, help explain X-ray findings, and answer
                    healthcare questions. Always remind users to consult healthcare professionals for medical advice.
                    Be clear, concise, and empathetic."""
                }
            ]

            # Add conversation history
            if conversation_history:
                messages.extend(conversation_history)

            # Add current message
            messages.append({"role": "user", "content": message})

            # Get response
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )

            assistant_message = response.choices[0].message.content

            return {
                "message": assistant_message,
                "conversation_id": None
            }

        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return await self._get_fallback_response(message)

    async def _get_local_response(self, message: str, conversation_history: Optional[List] = None):
        """
        Get response using local transformer model.
        """
        try:
            import torch

            # Encode input
            input_text = message
            if conversation_history:
                # Combine history
                history_text = " ".join([msg.get("content", "") for msg in conversation_history])
                input_text = history_text + " " + message

            input_ids = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')
            input_ids = input_ids.to(self.device)

            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + 100,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7
                )

            # Decode response
            response_text = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

            return {
                "message": response_text,
                "conversation_id": None
            }

        except Exception as e:
            print(f"Error with local model: {e}")
            return await self._get_fallback_response(message)

    async def _get_fallback_response(self, message: str):
        """
        Fallback rule-based responses.
        """
        message_lower = message.lower()

        # Define response templates
        responses = {
            "enhance": "The image enhancement process uses advanced AI to improve X-ray quality by reducing noise, "
                      "enhancing contrast, and preserving important details. Higher PSNR and SSIM values indicate "
                      "better enhancement quality.",

            "psnr": "PSNR (Peak Signal-to-Noise Ratio) measures image quality. Higher values (typically 25-40 dB) "
                   "indicate better quality with less noise and distortion.",

            "ssim": "SSIM (Structural Similarity Index) measures structural similarity between images. "
                   "Values range from 0 to 1, where 1 indicates perfect similarity.",

            "pneumonia": "Pneumonia appears on chest X-rays as areas of increased opacity (white areas) in the lungs. "
                        "Always consult a radiologist for professional diagnosis.",

            "xray": "X-rays are a form of electromagnetic radiation used to create images of internal body structures. "
                   "They're commonly used to detect bone fractures, lung infections, and other conditions.",

            "default": "I'm a healthcare AI assistant. I can help explain X-ray enhancement results, discuss medical "
                      "imaging concepts, and answer general health questions. However, please consult healthcare "
                      "professionals for medical advice. How can I help you today?"
        }

        # Check for keywords
        for keyword, response in responses.items():
            if keyword in message_lower and keyword != "default":
                return {"message": response, "conversation_id": None}

        # Default response
        return {"message": responses["default"], "conversation_id": None}

    async def explain_metrics(self, metrics: Dict[str, float]):
        """
        Explain enhancement metrics in natural language.

        Args:
            metrics: Dictionary with PSNR, SSIM, LPIPS values

        Returns:
            Natural language explanation
        """
        psnr = metrics.get('psnr', 0)
        ssim = metrics.get('ssim', 0)

        # Build explanation
        explanation = "Enhancement Results Analysis:\n\n"

        # PSNR interpretation
        if psnr > 30:
            psnr_quality = "excellent"
        elif psnr > 25:
            psnr_quality = "good"
        elif psnr > 20:
            psnr_quality = "fair"
        else:
            psnr_quality = "poor"

        explanation += f"Image Quality (PSNR): {psnr:.2f} dB - {psnr_quality.upper()}\n"
        explanation += f"The enhanced image shows {psnr_quality} quality with minimal noise and artifacts.\n\n"

        # SSIM interpretation
        if ssim > 0.9:
            ssim_quality = "excellent structural preservation"
        elif ssim > 0.8:
            ssim_quality = "good structural preservation"
        elif ssim > 0.7:
            ssim_quality = "fair structural preservation"
        else:
            ssim_quality = "structural details may be affected"

        explanation += f"Structural Similarity (SSIM): {ssim:.4f} - {ssim_quality.upper()}\n"
        explanation += f"The enhancement maintains {ssim_quality} of the original X-ray features.\n\n"

        # Overall assessment
        if psnr > 25 and ssim > 0.8:
            overall = "The enhancement was highly successful, producing a clear, detailed image suitable for diagnostic purposes."
        elif psnr > 20 and ssim > 0.7:
            overall = "The enhancement was successful, improving image quality while maintaining important structural details."
        else:
            overall = "The enhancement improved the image, though some fine details may require careful examination."

        explanation += f"Overall: {overall}\n\n"
        explanation += "Note: Always consult qualified radiologists for medical diagnosis and interpretation."

        return explanation


if __name__ == "__main__":
    # Test chatbot
    import asyncio

    async def test():
        print("Testing Healthcare Chatbot")
        print("=" * 50)

        # Initialize chatbot (will use fallback if no API key)
        chatbot = HealthcareChatbot(use_openai=False)

        # Test basic question
        print("\n1. Testing basic medical question:")
        response = await chatbot.get_response("What is pneumonia?")
        print(f"Response: {response['message']}")

        # Test metrics explanation
        print("\n2. Testing metrics explanation:")
        metrics = {"psnr": 28.5, "ssim": 0.85}
        explanation = await chatbot.explain_metrics(metrics)
        print(f"Explanation:\n{explanation}")

        print("\n" + "=" * 50)
        print("Test completed!")

    asyncio.run(test())
