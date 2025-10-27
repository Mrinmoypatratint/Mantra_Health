"""
Image Processing Utilities
===========================
Handles image preprocessing, enhancement, and postprocessing.
"""

import numpy as np
import cv2
import torch
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from training.metrics import calculate_psnr, calculate_ssim, calculate_lpips


class ImageProcessor:
    """
    Processes images for enhancement using the trained model.

    Args:
        model: Trained Pix2Pix GAN model
        device: Device to run inference on
        target_size: Target image size (default: 256)
    """
    def __init__(self, model, device, target_size=256):
        self.model = model
        self.device = device
        self.target_size = target_size

    def preprocess_image(self, image):
        """
        Preprocess PIL image for model input.

        Args:
            image: PIL Image (grayscale)

        Returns:
            Preprocessed tensor and original size
        """
        # Store original size
        original_size = image.size  # (width, height)

        # Convert to numpy array
        img_array = np.array(image)

        # Resize to model input size
        img_resized = cv2.resize(img_array, (self.target_size, self.target_size))

        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0

        # Convert to tensor and add batch and channel dimensions
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)

        # Move to device
        img_tensor = img_tensor.to(self.device)

        return img_tensor, original_size

    def postprocess_image(self, tensor, original_size=None):
        """
        Postprocess model output to image array.

        Args:
            tensor: Model output tensor (B, C, H, W)
            original_size: Original image size (width, height) for resizing back

        Returns:
            Numpy array (H, W) in range [0, 1]
        """
        # Move to CPU and remove batch/channel dimensions
        img_array = tensor.cpu().detach().squeeze().numpy()

        # Clip to [0, 1]
        img_array = np.clip(img_array, 0, 1)

        # Resize back to original size if specified
        if original_size is not None:
            img_array = cv2.resize(img_array, original_size)

        return img_array

    async def enhance_image(self, image, mask=None, return_attention=False):
        """
        Enhance X-ray image using the model.

        Args:
            image: PIL Image (grayscale)
            mask: Optional mask (not used currently)
            return_attention: Whether to return attention maps

        Returns:
            Dictionary with enhanced image, metrics, and optional attention maps
        """
        # Preprocess
        input_tensor, original_size = self.preprocess_image(image)

        # Run inference
        with torch.no_grad():
            enhanced_tensor = self.model.generate(input_tensor)

        # Postprocess
        enhanced_array = self.postprocess_image(enhanced_tensor, original_size)

        # Calculate metrics (compare with input at same size)
        input_resized = cv2.resize(np.array(image), (self.target_size, self.target_size))
        input_normalized = input_resized.astype(np.float32) / 255.0

        metrics = {
            'psnr': float(calculate_psnr(
                enhanced_tensor.cpu(),
                torch.from_numpy(input_normalized).unsqueeze(0).unsqueeze(0)
            )),
            'ssim': float(calculate_ssim(
                enhanced_tensor.cpu(),
                torch.from_numpy(input_normalized).unsqueeze(0).unsqueeze(0)
            ))
        }

        # Prepare result
        result = {
            'enhanced': enhanced_array,
            'metrics': metrics
        }

        # Add attention maps if requested
        if return_attention:
            attention_maps = self.model.get_attention_maps()
            if attention_maps:
                # Convert attention maps to numpy arrays
                attention_arrays = {}
                for key, att_tensor in attention_maps.items():
                    # Average over channels and normalize
                    att_array = att_tensor.cpu().detach().squeeze().numpy()
                    if len(att_array.shape) == 3:
                        att_array = att_array.mean(axis=0)

                    # Normalize to [0, 1]
                    att_array = (att_array - att_array.min()) / (att_array.max() - att_array.min() + 1e-8)

                    # Resize to original size
                    att_array = cv2.resize(att_array, original_size)

                    attention_arrays[key] = att_array

                result['attention_maps'] = attention_arrays

        return result

    def degrade_image(self, image, degradation_level=0.5):
        """
        Simulate degraded X-ray image for testing.

        Args:
            image: PIL Image
            degradation_level: Level of degradation (0.0 to 1.0)

        Returns:
            Degraded PIL Image
        """
        img_array = np.array(image)

        # Add noise
        noise_level = 15 * degradation_level
        noise = np.random.normal(0, noise_level, img_array.shape)
        degraded = img_array + noise
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)

        # Reduce contrast
        mean = degraded.mean()
        contrast_factor = 1.0 - (0.4 * degradation_level)
        degraded = mean + contrast_factor * (degraded - mean)
        degraded = np.clip(degraded, 0, 255).astype(np.uint8)

        # Add blur
        kernel_size = int(3 + 4 * degradation_level)
        if kernel_size % 2 == 0:
            kernel_size += 1
        degraded = cv2.GaussianBlur(degraded, (kernel_size, kernel_size), 0)

        return Image.fromarray(degraded)


if __name__ == "__main__":
    # Test image processor
    print("Testing ImageProcessor")
    print("=" * 50)

    # Create dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (512, 512), dtype=np.uint8), mode='L')

    # Initialize processor (with dummy model for testing)
    from models.gan import Pix2PixGAN
    model = Pix2PixGAN()
    device = torch.device('cpu')
    processor = ImageProcessor(model, device)

    # Test preprocessing
    input_tensor, original_size = processor.preprocess_image(dummy_image)
    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Original size: {original_size}")

    # Test enhancement
    import asyncio
    async def test_enhance():
        result = await processor.enhance_image(dummy_image, return_attention=True)
        print(f"\nEnhanced image shape: {result['enhanced'].shape}")
        print(f"Metrics: {result['metrics']}")
        if result.get('attention_maps'):
            print(f"Attention maps: {list(result['attention_maps'].keys())}")

    asyncio.run(test_enhance())

    print("\n" + "=" * 50)
    print("Test completed!")
