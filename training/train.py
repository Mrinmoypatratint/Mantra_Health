"""
Training Script for X-ray Enhancement GAN
==========================================
This script trains the Pix2Pix GAN model with proper loss functions and metrics.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gan import Pix2PixGAN, GANLoss
from training.dataset import create_data_loaders
from training.metrics import calculate_psnr, calculate_ssim, calculate_lpips


class Trainer:
    """
    Trainer class for Pix2Pix GAN model.

    Args:
        model: Pix2PixGAN model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
        device: Device to train on (cuda/cpu)
    """
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizers
        self.optimizer_G = optim.Adam(
            self.model.generator.parameters(),
            lr=config['lr'],
            betas=(config['beta1'], 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.model.discriminator.parameters(),
            lr=config['lr'],
            betas=(config['beta1'], 0.999)
        )

        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.StepLR(
            self.optimizer_G,
            step_size=config['lr_decay_step'],
            gamma=config['lr_decay_gamma']
        )
        self.scheduler_D = optim.lr_scheduler.StepLR(
            self.optimizer_D,
            step_size=config['lr_decay_step'],
            gamma=config['lr_decay_gamma']
        )

        # Loss functions
        self.criterion_GAN = GANLoss(gan_mode=config['gan_mode']).to(device)
        self.criterion_L1 = nn.L1Loss()
        self.criterion_MSE = nn.MSELoss()

        # Loss weights
        self.lambda_L1 = config['lambda_L1']
        self.lambda_perceptual = config.get('lambda_perceptual', 0.1)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=config['log_dir'])

        # Tracking
        self.current_epoch = 0
        self.best_psnr = 0.0

    def train_epoch(self):
        """
        Train for one epoch.
        """
        self.model.train()
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            degraded = batch['degraded'].to(self.device)
            clean = batch['clean'].to(self.device)
            batch_size = degraded.size(0)

            # ==========================================
            # Train Discriminator
            # ==========================================
            self.optimizer_D.zero_grad()

            # Generate fake images
            fake_images = self.model.generate(degraded)

            # Real loss
            pred_real = self.model.discriminate(degraded, clean)
            loss_D_real = self.criterion_GAN(pred_real, True)

            # Fake loss
            pred_fake = self.model.discriminate(degraded, fake_images.detach())
            loss_D_fake = self.criterion_GAN(pred_fake, False)

            # Total discriminator loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            self.optimizer_D.step()

            # ==========================================
            # Train Generator
            # ==========================================
            self.optimizer_G.zero_grad()

            # Generate fake images (again, for gradient flow)
            fake_images = self.model.generate(degraded)

            # GAN loss
            pred_fake = self.model.discriminate(degraded, fake_images)
            loss_G_GAN = self.criterion_GAN(pred_fake, True)

            # L1 loss (pixel-wise)
            loss_G_L1 = self.criterion_L1(fake_images, clean) * self.lambda_L1

            # Total generator loss
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            self.optimizer_G.step()

            # Update metrics
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()

            # Update progress bar
            pbar.set_postfix({
                'G_loss': f'{loss_G.item():.4f}',
                'D_loss': f'{loss_D.item():.4f}',
                'G_GAN': f'{loss_G_GAN.item():.4f}',
                'G_L1': f'{loss_G_L1.item():.4f}'
            })

            # Log to TensorBoard
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss_G', loss_G.item(), global_step)
            self.writer.add_scalar('Train/Loss_D', loss_D.item(), global_step)
            self.writer.add_scalar('Train/Loss_G_GAN', loss_G_GAN.item(), global_step)
            self.writer.add_scalar('Train/Loss_G_L1', loss_G_L1.item(), global_step)

        avg_loss_G = epoch_loss_G / len(self.train_loader)
        avg_loss_D = epoch_loss_D / len(self.train_loader)

        return avg_loss_G, avg_loss_D

    @torch.no_grad()
    def validate(self):
        """
        Validate the model.
        """
        self.model.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc="Validating")

        for batch in pbar:
            degraded = batch['degraded'].to(self.device)
            clean = batch['clean'].to(self.device)

            # Generate enhanced images
            enhanced = self.model.generate(degraded)

            # Calculate metrics
            psnr = calculate_psnr(enhanced, clean)
            ssim = calculate_ssim(enhanced, clean)
            # lpips = calculate_lpips(enhanced, clean, self.device)

            total_psnr += psnr
            total_ssim += ssim
            # total_lpips += lpips
            num_batches += 1

            pbar.set_postfix({
                'PSNR': f'{psnr:.2f}',
                'SSIM': f'{ssim:.4f}',
                # 'LPIPS': f'{lpips:.4f}'
            })

        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches
        # avg_lpips = total_lpips / num_batches

        print(f"\nValidation Results:")
        print(f"  PSNR: {avg_psnr:.2f} dB")
        print(f"  SSIM: {avg_ssim:.4f}")
        # print(f"  LPIPS: {avg_lpips:.4f}")

        # Log to TensorBoard
        self.writer.add_scalar('Val/PSNR', avg_psnr, self.current_epoch)
        self.writer.add_scalar('Val/SSIM', avg_ssim, self.current_epoch)
        # self.writer.add_scalar('Val/LPIPS', avg_lpips, self.current_epoch)

        # Log sample images
        self.log_images(degraded[:4], clean[:4], enhanced[:4])

        return avg_psnr, avg_ssim  # , avg_lpips

    def log_images(self, degraded, clean, enhanced):
        """
        Log sample images to TensorBoard.
        """
        import torchvision

        # Create grid
        degraded_grid = torchvision.utils.make_grid(degraded, nrow=2, normalize=True)
        clean_grid = torchvision.utils.make_grid(clean, nrow=2, normalize=True)
        enhanced_grid = torchvision.utils.make_grid(enhanced, nrow=2, normalize=True)

        self.writer.add_image('Images/Degraded', degraded_grid, self.current_epoch)
        self.writer.add_image('Images/Clean', clean_grid, self.current_epoch)
        self.writer.add_image('Images/Enhanced', enhanced_grid, self.current_epoch)

    def save_checkpoint(self, filename):
        """
        Save model checkpoint.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state_dict': self.model.generator.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename):
        """
        Load model checkpoint.
        """
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_psnr = checkpoint['best_psnr']
        print(f"Checkpoint loaded: {filename} (Epoch {self.current_epoch})")

    def train(self, num_epochs):
        """
        Train the model for specified number of epochs.
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("=" * 60)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            avg_loss_G, avg_loss_D = self.train_epoch()
            print(f"\nEpoch {epoch}: G_loss={avg_loss_G:.4f}, D_loss={avg_loss_D:.4f}")

            # Validate
            if (epoch + 1) % self.config['val_interval'] == 0:
                avg_psnr, avg_ssim = self.validate()

                # Save best model
                if avg_psnr > self.best_psnr:
                    self.best_psnr = avg_psnr
                    best_model_path = os.path.join(self.config['checkpoint_dir'], 'best_model.pth')
                    self.save_checkpoint(best_model_path)
                    print(f"New best model saved! PSNR: {avg_psnr:.2f} dB")

            # Save checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                checkpoint_path = os.path.join(
                    self.config['checkpoint_dir'],
                    f'checkpoint_epoch_{epoch}.pth'
                )
                self.save_checkpoint(checkpoint_path)

            # Update learning rate
            self.scheduler_G.step()
            self.scheduler_D.step()

            # Log learning rates
            self.writer.add_scalar('LR/Generator', self.optimizer_G.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('LR/Discriminator', self.optimizer_D.param_groups[0]['lr'], epoch)

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best PSNR: {self.best_psnr:.2f} dB")

        self.writer.close()


def main():
    """
    Main training function.
    """
    # Configuration
    config = {
        # Data
        'train_dir': './data/train',
        'val_dir': './data/val',
        'img_size': 256,
        'batch_size': 8,
        'num_workers': 4,
        'degradation_level': 0.5,

        # Training
        'num_epochs': 150,
        'lr': 2e-4,
        'beta1': 0.5,
        'lr_decay_step': 50,
        'lr_decay_gamma': 0.5,

        # Loss weights
        'lambda_L1': 100.0,
        'lambda_perceptual': 0.1,
        'gan_mode': 'vanilla',  # 'vanilla', 'lsgan', 'wgan'

        # Checkpointing
        'checkpoint_dir': './checkpoints',
        'log_dir': './logs',
        'save_interval': 10,
        'val_interval': 5,
    }

    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = create_data_loaders(
        train_dir=config['train_dir'],
        val_dir=config['val_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['img_size'],
        degradation_level=config['degradation_level']
    )

    # Create model
    print("Creating model...")
    model = Pix2PixGAN(in_channels=1, out_channels=1)

    # Count parameters
    num_params_G = sum(p.numel() for p in model.generator.parameters())
    num_params_D = sum(p.numel() for p in model.discriminator.parameters())
    print(f"Generator parameters: {num_params_G:,}")
    print(f"Discriminator parameters: {num_params_D:,}")

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)

    # Train
    trainer.train(config['num_epochs'])


if __name__ == "__main__":
    main()
