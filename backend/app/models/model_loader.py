"""
Model Loader for X-ray Enhancement
===================================
Handles loading and managing the trained model.
"""

import os
import torch
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.gan import Pix2PixGAN


class ModelLoader:
    """
    Loads and manages the trained Pix2Pix GAN model.

    Args:
        model_path: Path to the trained model checkpoint
        device: Device to load model on (cuda/cpu)
    """
    def __init__(self, model_path=None, device=None):
        self.model_path = model_path or os.environ.get(
            'MODEL_PATH',
            './checkpoints/best_model.pth'
        )
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    async def load_model(self):
        """
        Load the trained model from checkpoint.
        """
        print(f"Loading model from: {self.model_path}")
        print(f"Device: {self.device}")

        try:
            # Initialize model
            self.model = Pix2PixGAN(in_channels=1, out_channels=1)

            # Check if checkpoint exists
            if os.path.exists(self.model_path):
                # Load checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Load generator state dict
                if 'generator_state_dict' in checkpoint:
                    self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
                    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                    print(f"Best PSNR: {checkpoint.get('best_psnr', 'unknown')} dB")
                else:
                    # Assume it's a direct state dict
                    self.model.generator.load_state_dict(checkpoint)
                    print("Loaded model state dict")

            else:
                print(f"Warning: Checkpoint not found at {self.model_path}")
                print("Using randomly initialized model (for testing only)")

            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def get_model(self):
        """
        Get the loaded model.
        """
        return self.model

    def get_device(self):
        """
        Get the device the model is on.
        """
        return self.device


if __name__ == "__main__":
    # Test model loader
    import asyncio

    async def test():
        loader = ModelLoader()
        await loader.load_model()
        model = loader.get_model()
        print(f"Model type: {type(model)}")
        print(f"Generator parameters: {sum(p.numel() for p in model.generator.parameters()):,}")

    asyncio.run(test())
