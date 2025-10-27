"""
UNet with Attention Mechanism for X-ray Image Enhancement
===========================================================
This module implements a UNet architecture with attention gates for medical image enhancement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """
    Attention Gate block for focusing on relevant features.

    Args:
        F_g: Number of feature maps in gating signal
        F_l: Number of feature maps in skip connection
        F_int: Number of intermediate feature maps
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Forward pass through attention block.

        Args:
            g: Gating signal from decoder (coarser scale)
            x: Skip connection from encoder (finer scale)

        Returns:
            Attention-weighted feature map
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    """
    Convolutional block with two conv layers, batch norm, and ReLU activation.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """
    Upsampling block using transposed convolution.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class AttentionUNet(nn.Module):
    """
    UNet with Attention Gates for X-ray Image Enhancement.

    Architecture:
        - Encoder: 4 downsampling blocks (64, 128, 256, 512 channels)
        - Bottleneck: 1024 channels
        - Decoder: 4 upsampling blocks with attention gates
        - Output: 1 channel (grayscale X-ray image)

    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        out_channels: Number of output channels (default: 1 for grayscale)
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(AttentionUNet, self).__init__()

        # Encoder path
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(in_channels, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        # Decoder path with attention
        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = ConvBlock(128, 64)

        self.Conv_1x1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

        # Store attention maps for visualization
        self.attention_maps = {}

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, height, width)

        Returns:
            Enhanced X-ray image of shape (batch_size, 1, height, width)
        """
        # Encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # Decoding path with attention
        d5 = self.Up5(x5)
        x4_att = self.Att5(g=d5, x=x4)
        self.attention_maps['layer4'] = x4_att
        d5 = torch.cat((x4_att, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3_att = self.Att4(g=d4, x=x3)
        self.attention_maps['layer3'] = x3_att
        d4 = torch.cat((x3_att, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2_att = self.Att3(g=d3, x=x2)
        self.attention_maps['layer2'] = x2_att
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1_att = self.Att2(g=d2, x=x1)
        self.attention_maps['layer1'] = x1_att
        d2 = torch.cat((x1_att, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return torch.sigmoid(d1)

    def get_attention_maps(self):
        """
        Returns the attention maps from the last forward pass.
        Useful for visualization and interpretability.
        """
        return self.attention_maps


if __name__ == "__main__":
    # Test the model
    model = AttentionUNet(in_channels=1, out_channels=1)
    x = torch.randn(1, 1, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Test attention maps
    att_maps = model.get_attention_maps()
    print(f"\nAttention map keys: {att_maps.keys()}")
    for key, value in att_maps.items():
        print(f"{key} shape: {value.shape}")
