"""
Pix2Pix GAN for X-ray Image Enhancement
========================================
This module implements the Pix2Pix GAN architecture combining UNet generator
with a PatchGAN discriminator for high-quality image-to-image translation.
"""

import torch
import torch.nn as nn
from .attention_unet import AttentionUNet


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for realistic texture discrimination.

    The discriminator classifies whether 70x70 overlapping patches
    are real or fake, encouraging sharp high-frequency details.

    Args:
        in_channels: Number of input channels (default: 2, for input+output images)
        features: Base number of feature maps (default: 64)
    """
    def __init__(self, in_channels=2, features=64):
        super(PatchGANDiscriminator, self).__init__()

        # C64-C128-C256-C512
        # No batch norm in first layer
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.down1 = self._block(features, features * 2, stride=2)      # 128
        self.down2 = self._block(features * 2, features * 4, stride=2)  # 256
        self.down3 = self._block(features * 4, features * 8, stride=1)  # 512

        # Final layer to produce 1-channel prediction map
        self.final = nn.Sequential(
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def _block(self, in_channels, out_channels, stride):
        """
        Creates a discriminator block with Conv2d, BatchNorm, and LeakyReLU.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x, y):
        """
        Forward pass through discriminator.

        Args:
            x: Input degraded X-ray image (batch_size, 1, H, W)
            y: Output enhanced or real X-ray image (batch_size, 1, H, W)

        Returns:
            Patch predictions (batch_size, 1, H/16, W/16)
        """
        # Concatenate input and output images
        combined = torch.cat([x, y], dim=1)

        x = self.initial(combined)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.final(x)

        return x


class Pix2PixGAN(nn.Module):
    """
    Complete Pix2Pix GAN model for X-ray enhancement.

    Combines:
        - Generator: UNet with Attention mechanism
        - Discriminator: PatchGAN discriminator

    This is a wrapper class for training and inference.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(Pix2PixGAN, self).__init__()

        # Generator: UNet with Attention
        self.generator = AttentionUNet(in_channels=in_channels, out_channels=out_channels)

        # Discriminator: PatchGAN (takes input+output, so 2 channels)
        self.discriminator = PatchGANDiscriminator(in_channels=in_channels + out_channels, features=64)

    def forward(self, x, mode='generate'):
        """
        Forward pass through the GAN.

        Args:
            x: Input degraded X-ray image
            mode: 'generate' for generator only, 'discriminate' for discriminator

        Returns:
            Enhanced image (if mode='generate')
        """
        if mode == 'generate':
            return self.generator(x)
        else:
            raise ValueError("Use separate forward calls for generator and discriminator during training")

    def generate(self, x):
        """
        Generate enhanced X-ray image.

        Args:
            x: Input degraded X-ray image (batch_size, 1, H, W)

        Returns:
            Enhanced X-ray image (batch_size, 1, H, W)
        """
        return self.generator(x)

    def discriminate(self, x, y):
        """
        Discriminate between real and fake image pairs.

        Args:
            x: Input degraded X-ray image
            y: Enhanced or ground truth image

        Returns:
            Patch predictions
        """
        return self.discriminator(x, y)

    def get_attention_maps(self):
        """
        Get attention maps from generator for visualization.
        """
        return self.generator.get_attention_maps()


class GANLoss(nn.Module):
    """
    GAN Loss with support for different loss functions.

    Supports:
        - Vanilla GAN (BCE loss)
        - LSGAN (MSE loss)
        - WGAN (no sigmoid)
    """
    def __init__(self, gan_mode='vanilla', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode

        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'wgan':
            self.loss = None
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')

    def get_target_tensor(self, prediction, target_is_real):
        """
        Create target tensor with the same size as prediction.
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def forward(self, prediction, target_is_real):
        """
        Calculate loss.

        Args:
            prediction: Discriminator predictions
            target_is_real: If the ground truth label is for real images or fake images

        Returns:
            Calculated loss
        """
        if self.gan_mode in ['vanilla', 'lsgan']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgan':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


if __name__ == "__main__":
    # Test the models
    print("Testing Pix2Pix GAN Architecture")
    print("=" * 50)

    # Initialize model
    model = Pix2PixGAN(in_channels=1, out_channels=1)

    # Test data
    batch_size = 4
    degraded_img = torch.randn(batch_size, 1, 256, 256)
    real_img = torch.randn(batch_size, 1, 256, 256)

    # Test Generator
    print("\n1. Testing Generator:")
    enhanced_img = model.generate(degraded_img)
    print(f"   Input shape: {degraded_img.shape}")
    print(f"   Output shape: {enhanced_img.shape}")
    print(f"   Generator parameters: {sum(p.numel() for p in model.generator.parameters()):,}")

    # Test Discriminator
    print("\n2. Testing Discriminator:")
    real_pred = model.discriminate(degraded_img, real_img)
    fake_pred = model.discriminate(degraded_img, enhanced_img.detach())
    print(f"   Real prediction shape: {real_pred.shape}")
    print(f"   Fake prediction shape: {fake_pred.shape}")
    print(f"   Discriminator parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")

    # Test GAN Loss
    print("\n3. Testing GAN Loss:")
    criterion = GANLoss(gan_mode='vanilla')
    loss_real = criterion(real_pred, True)
    loss_fake = criterion(fake_pred, False)
    print(f"   Loss on real: {loss_real.item():.4f}")
    print(f"   Loss on fake: {loss_fake.item():.4f}")

    # Test attention maps
    print("\n4. Testing Attention Maps:")
    att_maps = model.get_attention_maps()
    for key, value in att_maps.items():
        print(f"   {key}: {value.shape}")

    print("\n" + "=" * 50)
    print("All tests passed successfully!")
