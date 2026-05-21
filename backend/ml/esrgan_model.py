"""
ESRGAN Model Architecture Implementation
==========================================

Authored by: Machine Learning Engineering Team
Date: April 2026
Version: 1.0

Description:
    Complete implementation of Enhanced Super-Resolution Generative Adversarial Network (ESRGAN)
    following the architecture from "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
    (Wang et al., 2018). Includes RRDBNet generator with dense residual blocks and VGG-style discriminator.

Architecture Components:
    - RRDBNet: Generator with Residual-in-Residual Dense Blocks
    - ResidualDenseBlock: Dense connections with growth channel
    - VGGStyleDiscriminator: Multi-scale patch discriminator
    - Utilities: Weight initialization and architectural helpers

Key Features:
    - Configurable scaling factors (2x, 4x, 8x)
    - Residual scaling for stable training
    - Batch normalization-free design for artifact reduction
    - Support for both training and inference modes
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Dense Block Components
# ============================================================================

class DenseLayer(nn.Module):
    """
    Single dense layer with residual connection.
    
    Architecture:
        Input -> Conv(3x3) -> LeakyReLU -> Residual Connection
    
    Args:
        in_channels (int): Number of input channels
        growth_channels (int): Number of growth channels (typically 32)
    """
    
    def __init__(self, in_channels: int, growth_channels: int = 32):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            growth_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dense connection.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Concatenated output (B, C + growth, H, W)
        """
        out = self.activation(self.conv(x))
        return torch.cat([x, out], dim=1)


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB) with 5 dense layers and residual scaling.
    
    Architecture:
        Input -> [DenseLayer x 5] -> Conv(1x1) -> Residual Scaling -> Output
    
    The final 1x1 convolution reduces channels back to input dimension,
    followed by residual scaling (β) for training stability.
    
    Args:
        num_features (int): Number of input/output channels (typically 64)
        growth_channels (int): Growth rate for dense connections (typically 32)
        residual_scaling (float): Scaling factor for residual connection (typically 0.2)
    """
    
    def __init__(
        self,
        num_features: int = 64,
        growth_channels: int = 32,
        residual_scaling: float = 0.2
    ):
        super(ResidualDenseBlock, self).__init__()
        self.residual_scaling = residual_scaling
        
        # Five dense layers with progressive channel growth
        self.dense_layers = nn.ModuleList([
            DenseLayer(num_features + i * growth_channels, growth_channels)
            for i in range(5)
        ])
        
        # 1x1 convolution to reduce channels back to num_features
        self.bottleneck = nn.Conv2d(
            num_features + 5 * growth_channels,
            num_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual dense block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output with residual connection (B, C, H, W)
        """
        identity = x
        
        # Progressive dense connections
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        
        # Bottleneck and residual scaling
        x = self.bottleneck(x)
        x = x * self.residual_scaling
        
        return x + identity


class ResidualInResidualDenseBlock(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB) - core building block of ESRGAN.
    
    Architecture:
        Input -> [RDB x 3] -> Residual Scaling -> Output
    
    Stacks three RDB blocks with an outer residual connection, providing
    deeper and more expressive feature extraction.
    
    Args:
        num_features (int): Number of channels (typically 64)
        growth_channels (int): Growth rate for dense layers (typically 32)
        residual_scaling (float): Scaling factor for residual connections
    """
    
    def __init__(
        self,
        num_features: int = 64,
        growth_channels: int = 32,
        residual_scaling: float = 0.2
    ):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.residual_scaling = residual_scaling
        
        # Stack of three RDB blocks
        self.rdb_blocks = nn.ModuleList([
            ResidualDenseBlock(num_features, growth_channels, residual_scaling)
            for _ in range(3)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RRDB.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Output with nested residual connection (B, C, H, W)
        """
        identity = x
        
        # Pass through three RDB blocks
        for rdb in self.rdb_blocks:
            x = rdb(x)
        
        # Outer residual scaling
        x = x * self.residual_scaling
        
        return x + identity


# ============================================================================
# ESRGAN Generator (RRDBNet)
# ============================================================================

class RRDBNet(nn.Module):
    """
    ESRGAN Generator Network with RRDB architecture.
    
    Architecture:
        1. Initial feature extraction (Conv 3x3)
        2. RRDB trunk (23 RRDB blocks by default)
        3. Trunk fusion (Conv 3x3 + residual)
        4. Upsampling blocks (PixelShuffle)
        5. Final reconstruction (Conv 3x3 x 2)
    
    This is the complete generator from the ESRGAN paper, designed for
    photorealistic image super-resolution with artifact-free output.
    
    Args:
        in_channels (int): Number of input channels (3 for RGB)
        out_channels (int): Number of output channels (3 for RGB)
        num_features (int): Number of feature channels (typically 64)
        num_blocks (int): Number of RRDB blocks (typically 23)
        growth_channels (int): Growth rate for dense connections (typically 32)
        scale_factor (int): Upscaling factor (2, 4, or 8)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 23,
        growth_channels: int = 32,
        scale_factor: int = 4
    ):
        super(RRDBNet, self).__init__()
        self.scale_factor = scale_factor
        
        # Calculate number of upsampling blocks
        self.num_upsample = int(math.log(scale_factor, 2))
        
        # ====================================================================
        # 1. Initial Feature Extraction
        # ====================================================================
        self.conv_first = nn.Conv2d(
            in_channels,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        
        # ====================================================================
        # 2. RRDB Trunk
        # ====================================================================
        self.rrdb_trunk = nn.ModuleList([
            ResidualInResidualDenseBlock(num_features, growth_channels)
            for _ in range(num_blocks)
        ])
        
        # ====================================================================
        # 3. Trunk Fusion
        # ====================================================================
        self.trunk_conv = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        
        # ====================================================================
        # 4. Upsampling Blocks
        # ====================================================================
        self.upsampling = nn.ModuleList()
        for _ in range(self.num_upsample):
            self.upsampling.append(
                nn.Sequential(
                    nn.Conv2d(
                        num_features,
                        num_features * 4,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True
                    ),
                    nn.PixelShuffle(upscale_factor=2),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                )
            )
        
        # ====================================================================
        # 5. Final Reconstruction
        # ====================================================================
        self.conv_hr = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.conv_last = nn.Conv2d(
            num_features,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.activation_hr = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming initialization for Conv layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RRDBNet generator.
        
        Args:
            x (torch.Tensor): Input LR image (B, 3, H, W)
            
        Returns:
            torch.Tensor: Super-resolved HR image (B, 3, H*scale, W*scale)
        """
        # Initial feature extraction
        feat = self.conv_first(x)
        trunk = feat
        
        # Pass through RRDB blocks
        for rrdb_block in self.rrdb_trunk:
            trunk = rrdb_block(trunk)
        
        # Trunk fusion with residual connection
        trunk = self.trunk_conv(trunk)
        feat = feat + trunk
        
        # Upsampling
        for upsample_block in self.upsampling:
            feat = upsample_block(feat)
        
        # Final reconstruction
        out = self.conv_last(self.activation_hr(self.conv_hr(feat)))
        
        return out


# ============================================================================
# VGG-Style Discriminator
# ============================================================================

class VGGStyleDiscriminator(nn.Module):
    """
    VGG-style discriminator for adversarial training.
    
    Architecture follows VGG network design with:
        - 8 convolutional blocks with increasing channels
        - Batch normalization removed (as in ESRGAN paper)
        - LeakyReLU activation
        - Final classification layers
    
    Args:
        in_channels (int): Number of input channels (3 for RGB)
        num_features (int): Base number of features (typically 64)
    """
    
    def __init__(self, in_channels: int = 3, num_features: int = 64):
        super(VGGStyleDiscriminator, self).__init__()
        
        # Feature extraction blocks
        self.features = nn.Sequential(
            # Conv block 1: 64 channels
            nn.Conv2d(in_channels, num_features, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(num_features, num_features, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            # Conv block 2: 128 channels
            nn.Conv2d(num_features, num_features * 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(num_features * 2, num_features * 2, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            # Conv block 3: 256 channels
            nn.Conv2d(num_features * 2, num_features * 4, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(num_features * 4, num_features * 4, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            # Conv block 4: 512 channels
            nn.Conv2d(num_features * 4, num_features * 8, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(num_features * 8, num_features * 8, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_features * 8, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize discriminator weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.
        
        Args:
            x (torch.Tensor): Input image (B, 3, H, W)
            
        Returns:
            torch.Tensor: Realness score (B, 1)
        """
        features = self.features(x)
        out = self.classifier(features)
        return out


# ============================================================================
# Model Factory & Utilities
# ============================================================================

def create_esrgan_generator(
    scale_factor: int = 4,
    num_blocks: int = 23,
    pretrained: bool = False,
    weights_path: Optional[str] = None
) -> RRDBNet:
    """
    Factory function to create ESRGAN generator with optional pretrained weights.
    
    Args:
        scale_factor (int): Upscaling factor (2, 4, or 8)
        num_blocks (int): Number of RRDB blocks
        pretrained (bool): Whether to load pretrained weights
        weights_path (Optional[str]): Path to pretrained weights file
        
    Returns:
        RRDBNet: Initialized generator model
        
    Example:
        >>> generator = create_esrgan_generator(scale_factor=4, pretrained=True)
        >>> output = generator(low_res_image)
    """
    model = RRDBNet(
        in_channels=3,
        out_channels=3,
        num_features=64,
        num_blocks=num_blocks,
        growth_channels=32,
        scale_factor=scale_factor
    )
    
    if pretrained and weights_path:
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            # Handle different checkpoint formats
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'generator' in state_dict:
                state_dict = state_dict['generator']
            model.load_state_dict(state_dict, strict=True)
            print(f"✓ Loaded pretrained weights from {weights_path}")
        except Exception as e:
            print(f"⚠ Failed to load pretrained weights: {e}")
            print("  Continuing with randomly initialized weights")
    
    return model


def create_esrgan_discriminator() -> VGGStyleDiscriminator:
    """
    Factory function to create ESRGAN discriminator.
    
    Returns:
        VGGStyleDiscriminator: Initialized discriminator model
    """
    return VGGStyleDiscriminator(in_channels=3, num_features=64)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        Tuple[int, int]: (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================================
# Model Testing & Validation
# ============================================================================

if __name__ == "__main__":
    """
    Quick validation script to test model architecture.
    """
    print("=" * 80)
    print("ESRGAN Model Architecture Validation")
    print("=" * 80)
    
    # Create models
    print("\n[1] Creating Generator (RRDBNet)...")
    generator = create_esrgan_generator(scale_factor=4, num_blocks=23)
    total_g, trainable_g = count_parameters(generator)
    print(f"    ✓ Generator parameters: {total_g:,} (trainable: {trainable_g:,})")
    
    print("\n[2] Creating Discriminator...")
    discriminator = create_esrgan_discriminator()
    total_d, trainable_d = count_parameters(discriminator)
    print(f"    ✓ Discriminator parameters: {total_d:,} (trainable: {trainable_d:,})")
    
    # Test forward pass
    print("\n[3] Testing forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")
    
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Test input
    batch_size = 2
    lr_size = 128
    test_input = torch.randn(batch_size, 3, lr_size, lr_size).to(device)
    
    with torch.no_grad():
        # Generator
        sr_output = generator(test_input)
        print(f"    ✓ Generator: {test_input.shape} -> {sr_output.shape}")
        
        # Discriminator
        disc_output = discriminator(sr_output)
        print(f"    ✓ Discriminator: {sr_output.shape} -> {disc_output.shape}")
    
    print("\n[4] Model validation complete!")
    print("=" * 80)
