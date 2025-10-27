"""
X-ray Image Dataset and Preprocessing Module
=============================================
This module handles dataset loading, preprocessing, and augmentation for X-ray images.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


class XRayDataset(Dataset):
    """
    Custom Dataset for X-ray Image Enhancement.

    This dataset loads clean X-ray images and generates degraded versions
    by adding noise, reducing contrast, and applying blur.

    Args:
        image_dir: Directory containing clean X-ray images
        mask_dir: Optional directory containing segmentation masks
        transform: Albumentations transformation pipeline
        img_size: Target image size (default: 256)
        degradation_level: Level of degradation to apply (0.0 to 1.0)
    """
    def __init__(self, image_dir, mask_dir=None, transform=None, img_size=256, degradation_level=0.5):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_size = img_size
        self.degradation_level = degradation_level

        # Get list of image files
        self.image_files = [f for f in os.listdir(image_dir)
                           if f.endswith(('.png', '.jpg', '.jpeg', '.dcm'))]

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")

        print(f"Loaded {len(self.image_files)} images from {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def add_noise(self, image, noise_level):
        """
        Add Gaussian noise to image.

        Args:
            image: Input image (numpy array, 0-255)
            noise_level: Standard deviation of Gaussian noise

        Returns:
            Noisy image
        """
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_img = image + noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    def reduce_contrast(self, image, factor):
        """
        Reduce image contrast.

        Args:
            image: Input image (numpy array, 0-255)
            factor: Contrast reduction factor (0-1, lower = more reduction)

        Returns:
            Low contrast image
        """
        mean = image.mean()
        low_contrast = mean + factor * (image - mean)
        return np.clip(low_contrast, 0, 255).astype(np.uint8)

    def add_blur(self, image, kernel_size):
        """
        Add Gaussian blur to image.

        Args:
            image: Input image
            kernel_size: Size of Gaussian kernel (must be odd)

        Returns:
            Blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def degrade_image(self, image):
        """
        Generate degraded version of clean X-ray image.

        Applies combination of:
            - Gaussian noise
            - Contrast reduction
            - Blur

        Args:
            image: Clean X-ray image (numpy array, 0-255)

        Returns:
            Degraded X-ray image
        """
        degraded = image.copy()

        # Add noise (scaled by degradation level)
        noise_level = 15 * self.degradation_level
        degraded = self.add_noise(degraded, noise_level)

        # Reduce contrast
        contrast_factor = 1.0 - (0.4 * self.degradation_level)
        degraded = self.reduce_contrast(degraded, contrast_factor)

        # Add blur
        kernel_size = int(3 + 4 * self.degradation_level)
        degraded = self.add_blur(degraded, kernel_size)

        return degraded

    def __getitem__(self, idx):
        """
        Get a single item from dataset.

        Returns:
            Dictionary containing:
                - 'degraded': Degraded input image
                - 'clean': Ground truth clean image
                - 'mask': Optional segmentation mask
                - 'filename': Image filename
        """
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Read as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Could not read image: {img_path}")

        # Resize to target size
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Generate degraded version
        degraded = self.degrade_image(image)

        # Load mask if available
        mask = None
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, img_name)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (self.img_size, self.img_size))

        # Apply transformations
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, masks=[degraded, mask])
                image = augmented['image']
                degraded = augmented['masks'][0]
                mask = augmented['masks'][1]
            else:
                augmented = self.transform(image=image, masks=[degraded])
                image = augmented['image']
                degraded = augmented['masks'][0]

        # Convert to tensor and normalize to [0, 1]
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float() / 255.0
            degraded = torch.from_numpy(degraded).float() / 255.0
            if mask is not None:
                mask = torch.from_numpy(mask).float() / 255.0

        # Add channel dimension
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
            degraded = degraded.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        result = {
            'degraded': degraded,
            'clean': image,
            'filename': img_name
        }

        if mask is not None:
            result['mask'] = mask

        return result


def get_training_augmentation():
    """
    Returns training augmentation pipeline using Albumentations.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
    ])


def get_validation_augmentation():
    """
    Returns validation augmentation (no augmentation, just normalization).
    """
    return None


def create_data_loaders(train_dir, val_dir, batch_size=8, num_workers=4,
                       img_size=256, degradation_level=0.5, mask_dir=None):
    """
    Create training and validation data loaders.

    Args:
        train_dir: Directory with training images
        val_dir: Directory with validation images
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        img_size: Target image size
        degradation_level: Level of image degradation
        mask_dir: Optional directory with masks

    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = XRayDataset(
        image_dir=train_dir,
        mask_dir=mask_dir,
        transform=get_training_augmentation(),
        img_size=img_size,
        degradation_level=degradation_level
    )

    val_dataset = XRayDataset(
        image_dir=val_dir,
        mask_dir=mask_dir,
        transform=get_validation_augmentation(),
        img_size=img_size,
        degradation_level=degradation_level
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing XRayDataset")
    print("=" * 50)

    # Create dummy data directory for testing
    test_dir = "./data/test_images"
    os.makedirs(test_dir, exist_ok=True)

    # Create a dummy image
    dummy_img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    cv2.imwrite(os.path.join(test_dir, "test_xray.png"), dummy_img)

    # Test dataset
    dataset = XRayDataset(
        image_dir=test_dir,
        transform=get_training_augmentation(),
        img_size=256,
        degradation_level=0.5
    )

    print(f"Dataset length: {len(dataset)}")

    # Get first item
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Degraded shape: {sample['degraded'].shape}")
    print(f"Clean shape: {sample['clean'].shape}")
    print(f"Degraded range: [{sample['degraded'].min():.3f}, {sample['degraded'].max():.3f}]")
    print(f"Clean range: [{sample['clean'].min():.3f}, {sample['clean'].max():.3f}]")

    print("\n" + "=" * 50)
    print("Test completed successfully!")
