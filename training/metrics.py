"""
Image Quality Metrics for X-ray Enhancement
============================================
This module provides functions to calculate PSNR, SSIM, and LPIPS metrics.
"""

import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1, img2, data_range=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1: First image tensor (B, C, H, W) or numpy array
        img2: Second image tensor (B, C, H, W) or numpy array
        data_range: Range of pixel values (default: 1.0 for normalized images)

    Returns:
        Average PSNR value in dB
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()

    # Handle batch dimension
    if len(img1.shape) == 4:
        batch_size = img1.shape[0]
        psnr_values = []
        for i in range(batch_size):
            # Convert (C, H, W) to (H, W, C) for scikit-image
            im1 = np.transpose(img1[i], (1, 2, 0))
            im2 = np.transpose(img2[i], (1, 2, 0))

            # Remove channel dimension if grayscale
            if im1.shape[2] == 1:
                im1 = im1.squeeze(2)
                im2 = im2.squeeze(2)

            psnr_val = psnr(im1, im2, data_range=data_range)
            psnr_values.append(psnr_val)

        return np.mean(psnr_values)
    else:
        # Single image
        im1 = np.transpose(img1, (1, 2, 0)) if len(img1.shape) == 3 else img1
        im2 = np.transpose(img2, (1, 2, 0)) if len(img2.shape) == 3 else img2

        if len(im1.shape) == 3 and im1.shape[2] == 1:
            im1 = im1.squeeze(2)
            im2 = im2.squeeze(2)

        return psnr(im1, im2, data_range=data_range)


def calculate_ssim(img1, img2, data_range=1.0):
    """
    Calculate Structural Similarity Index (SSIM) between two images.

    Args:
        img1: First image tensor (B, C, H, W) or numpy array
        img2: Second image tensor (B, C, H, W) or numpy array
        data_range: Range of pixel values (default: 1.0 for normalized images)

    Returns:
        Average SSIM value (0 to 1)
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().numpy()

    # Handle batch dimension
    if len(img1.shape) == 4:
        batch_size = img1.shape[0]
        ssim_values = []
        for i in range(batch_size):
            # Convert (C, H, W) to (H, W, C) for scikit-image
            im1 = np.transpose(img1[i], (1, 2, 0))
            im2 = np.transpose(img2[i], (1, 2, 0))

            # Remove channel dimension if grayscale
            if im1.shape[2] == 1:
                im1 = im1.squeeze(2)
                im2 = im2.squeeze(2)

            ssim_val = ssim(im1, im2, data_range=data_range)
            ssim_values.append(ssim_val)

        return np.mean(ssim_values)
    else:
        # Single image
        im1 = np.transpose(img1, (1, 2, 0)) if len(img1.shape) == 3 else img1
        im2 = np.transpose(img2, (1, 2, 0)) if len(img2.shape) == 3 else img2

        if len(im1.shape) == 3 and im1.shape[2] == 1:
            im1 = im1.squeeze(2)
            im2 = im2.squeeze(2)

        return ssim(im1, im2, data_range=data_range)


def calculate_lpips(img1, img2, device='cpu'):
    """
    Calculate Learned Perceptual Image Patch Similarity (LPIPS).

    Note: This requires the lpips library to be installed.
    Install with: pip install lpips

    Args:
        img1: First image tensor (B, C, H, W)
        img2: Second image tensor (B, C, H, W)
        device: Device to run computation on

    Returns:
        Average LPIPS value (lower is better)
    """
    try:
        import lpips
    except ImportError:
        print("Warning: lpips not installed. Install with: pip install lpips")
        return 0.0

    # Initialize LPIPS model (using VGG network)
    loss_fn = lpips.LPIPS(net='vgg').to(device)

    # Ensure images are on correct device
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).to(device)
    else:
        img1 = img1.to(device)

    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).to(device)
    else:
        img2 = img2.to(device)

    # LPIPS expects 3-channel images, so repeat grayscale to 3 channels
    if img1.shape[1] == 1:
        img1 = img1.repeat(1, 3, 1, 1)
        img2 = img2.repeat(1, 3, 1, 1)

    # Normalize to [-1, 1] range (LPIPS expects this)
    img1 = img1 * 2.0 - 1.0
    img2 = img2 * 2.0 - 1.0

    # Calculate LPIPS
    with torch.no_grad():
        lpips_value = loss_fn(img1, img2)

    return lpips_value.mean().item()


def calculate_all_metrics(img1, img2, device='cpu'):
    """
    Calculate all metrics (PSNR, SSIM, LPIPS) at once.

    Args:
        img1: First image tensor (B, C, H, W)
        img2: Second image tensor (B, C, H, W)
        device: Device for computation

    Returns:
        Dictionary with all metric values
    """
    metrics = {
        'psnr': calculate_psnr(img1, img2),
        'ssim': calculate_ssim(img1, img2),
        'lpips': calculate_lpips(img1, img2, device)
    }

    return metrics


def print_metrics(metrics):
    """
    Pretty print metrics dictionary.

    Args:
        metrics: Dictionary of metric names and values
    """
    print("Image Quality Metrics:")
    print("-" * 40)
    print(f"PSNR:  {metrics['psnr']:>8.2f} dB")
    print(f"SSIM:  {metrics['ssim']:>8.4f}")
    if 'lpips' in metrics:
        print(f"LPIPS: {metrics['lpips']:>8.4f}")
    print("-" * 40)


if __name__ == "__main__":
    # Test metrics
    print("Testing Image Quality Metrics")
    print("=" * 50)

    # Create test images
    batch_size = 4
    img1 = torch.rand(batch_size, 1, 256, 256)
    img2 = torch.rand(batch_size, 1, 256, 256)

    # Test identical images (should give perfect scores)
    print("\n1. Testing with identical images:")
    psnr_val = calculate_psnr(img1, img1)
    ssim_val = calculate_ssim(img1, img1)
    print(f"   PSNR: {psnr_val:.2f} dB (should be inf)")
    print(f"   SSIM: {ssim_val:.4f} (should be 1.0)")

    # Test different images
    print("\n2. Testing with different images:")
    psnr_val = calculate_psnr(img1, img2)
    ssim_val = calculate_ssim(img1, img2)
    print(f"   PSNR: {psnr_val:.2f} dB")
    print(f"   SSIM: {ssim_val:.4f}")

    # Test all metrics
    print("\n3. Testing all metrics function:")
    metrics = calculate_all_metrics(img1, img2)
    print_metrics(metrics)

    print("\n" + "=" * 50)
    print("All tests completed!")
