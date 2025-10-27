#!/usr/bin/env python3
"""
Download sample X-ray images for quick testing
This downloads a few public domain chest X-ray images from NIH
"""

import os
import urllib.request
from pathlib import Path

# Sample X-ray image URLs from public datasets
SAMPLE_URLS = [
    # These are sample URLs - in practice, you'd use actual public dataset URLs
    # For now, this is a template. Users should add their own URLs or use Kaggle
]

def download_file(url, dest_path):
    """Download a file from URL to destination path"""
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"  ✓ Saved to {dest_path}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

def main():
    """Main download function"""
    print("=" * 60)
    print("X-RAY SAMPLE IMAGE DOWNLOADER")
    print("=" * 60)
    print()

    # Create directories
    train_dir = Path("data/train")
    val_dir = Path("data/val")
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print("Directories created:")
    print(f"  - {train_dir}")
    print(f"  - {val_dir}")
    print()

    # Note to user
    print("IMPORTANT NOTE:")
    print()
    print("This script is a template. To download actual X-ray images:")
    print()
    print("OPTION 1: Use Kaggle (Recommended)")
    print("  1. Install: pip install kaggle")
    print("  2. Get API key from https://www.kaggle.com/")
    print("  3. Run: kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")
    print()
    print("OPTION 2: Use Google Colab")
    print("  1. Upload notebooks/training_demo.ipynb to Colab")
    print("  2. Follow the notebook to download and train")
    print()
    print("OPTION 3: Manual Download")
    print("  1. Visit: https://nihcc.app.box.com/v/ChestXray-NIHCC")
    print("  2. Download X-ray images")
    print("  3. Place JPG/PNG files in:")
    print(f"     - Training images: {train_dir.absolute()}")
    print(f"     - Validation images: {val_dir.absolute()}")
    print()
    print("OPTION 4: Use Your Own Images")
    print("  - Place any chest X-ray images (JPG/PNG) in the directories above")
    print("  - Minimum: 10 images for training, 2 for validation")
    print("  - Recommended: 100+ for good results")
    print()
    print("=" * 60)
    print()

    # Check if user has added images manually
    train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpeg"))
    val_images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png")) + list(val_dir.glob("*.jpeg"))

    if train_images or val_images:
        print(f"Found {len(train_images)} training images")
        print(f"Found {len(val_images)} validation images")
        print()
        if len(train_images) >= 10 and len(val_images) >= 2:
            print("SUCCESS: You have enough images to start training!")
            print()
            print("Run training with:")
            print("  cd training")
            print("  python train.py")
        else:
            print("WARNING: Need more images:")
            print(f"  - Training: {max(0, 10 - len(train_images))} more needed")
            print(f"  - Validation: {max(0, 2 - len(val_images))} more needed")
    else:
        print("No images found yet. Please add X-ray images using one of the options above.")

    print()
    print("=" * 60)

if __name__ == "__main__":
    main()
