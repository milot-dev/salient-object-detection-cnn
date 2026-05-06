"""
Purpose:
--------
Defines the CNN models used for Salient Object Detection (SOD).

This file contains:
1. Baseline encoder-decoder CNN
2. Improved encoder-decoder CNN with BatchNorm and Dropout

Input:
------
RGB image tensor with shape [B, 3, 128, 128]

Output:
-------
Single-channel saliency mask with shape [B, 1, 128, 128]
Values are between 0 and 1 because of the final Sigmoid layer.
"""

import torch
import torch.nn as nn


# =========================================================
# Basic Convolution Block
# =========================================================
class ConvBlock(nn.Module):
    """
    A simple block: Conv2D -> ReLU

    Used in the baseline model.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# =========================================================
# Improved Convolution Block
# =========================================================
class ImprovedConvBlock(nn.Module):
    """
    Improved block: Conv2D -> BatchNorm -> ReLU -> Dropout

    Used in the improved model.
    BatchNorm helps stabilize training.
    Dropout helps reduce overfitting.
    """

    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.block(x)


# =========================================================
# Baseline Encoder-Decoder CNN
# =========================================================
class BaselineSODModel(nn.Module):
    """
    Baseline CNN for saliency prediction.

    Architecture:
    -------------
    Encoder:
        Conv -> Pool
        Conv -> Pool
        Conv -> Pool

    Decoder:
        Transposed Conv
        Transposed Conv
        Transposed Conv

    Final:
        1x1 Conv + Sigmoid
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(32, 32)

        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(16, 16)

        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x = self.pool1(x)

        x = self.enc2(x)
        x = self.pool2(x)

        x = self.enc3(x)
        x = self.pool3(x)

        # Decoder
        x = self.up1(x)
        x = self.dec1(x)

        x = self.up2(x)
        x = self.dec2(x)

        x = self.up3(x)
        x = self.dec3(x)

        return self.output(x)


# =========================================================
# Improved Encoder-Decoder CNN
# =========================================================
class ImprovedSODModel(nn.Module):
    """
    Improved CNN for saliency prediction.

    Improvements over baseline:
    ---------------------------
    1. Batch Normalization
    2. Dropout
    3. Slightly deeper encoder-decoder blocks
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            ImprovedConvBlock(3, 32),
            ImprovedConvBlock(32, 32)
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            ImprovedConvBlock(32, 64),
            ImprovedConvBlock(64, 64)
        )
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            ImprovedConvBlock(64, 128),
            ImprovedConvBlock(128, 128)
        )
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = nn.Sequential(
            ImprovedConvBlock(128, 256),
            ImprovedConvBlock(256, 256)
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = ImprovedConvBlock(128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ImprovedConvBlock(64, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = ImprovedConvBlock(32, 32)

        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x = self.pool1(x)

        x = self.enc2(x)
        x = self.pool2(x)

        x = self.enc3(x)
        x = self.pool3(x)

        x = self.enc4(x)

        # Decoder
        x = self.up1(x)
        x = self.dec1(x)

        x = self.up2(x)
        x = self.dec2(x)

        x = self.up3(x)
        x = self.dec3(x)

        return self.output(x)