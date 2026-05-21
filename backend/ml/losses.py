"""
ESRGAN Losses Definition
========================

Authored by: Machine Learning Engineering Team
Date: April 2026
Version: 1.0

Description:
    Implements standard Wang et al. original paper loss objectives:
    - Pixel L1 Loss (for structural stability)
    - Relativistic Average GAN (RaGAN) Loss (for natural high-frequency textures)
    - VGG Perceptual Feature Loss (for perceptual quality)
"""

import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class VGGPerceptualLoss(nn.Module):
    """
    Computes perceptual similarity using feature maps extracted from pre-trained VGG-19.
    Matches original ESRGAN setup using features before the 35th activation layer (relu5_4).
    """
    def __init__(self, use_gpu: bool = True):
        super(VGGPerceptualLoss, self).__init__()
        
        # Load VGG-19 model weights
        weights = VGG19_Weights.DEFAULT
        vgg = vgg19(weights=weights)
        
        # We need features up to relu5_4 (index 34)
        self.features = nn.Sequential(*list(vgg.features.children())[:35]).eval()
        
        # Freeze parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
        if use_gpu and torch.cuda.is_available():
            self.features = self.features.cuda()
            
        # Normalization means and standard deviations for ImageNet
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.l1_loss = nn.L1Loss()
        
    def forward(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute VGG perceptual distance loss.
        """
        # Ensure device matching
        self.mean = self.mean.to(input_tensor.device)
        self.std = self.std.to(input_tensor.device)
        
        # Normalize inputs for ImageNet-trained VGG network
        input_norm = (input_tensor - self.mean) / self.std
        target_norm = (target_tensor - self.mean) / self.std
        
        input_features = self.features(input_norm)
        target_features = self.features(target_norm)
        
        return self.l1_loss(input_features, target_features)


class RelativisticAverageGANLoss(nn.Module):
    """
    Relativistic Average GAN (RaGAN) Loss implementation.
    Estimates whether a real image is more realistic than the average fake image,
    and vice-versa.
    """
    def __init__(self):
        super(RelativisticAverageGANLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def generator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        """Compute RaGAN loss for the generator."""
        # Mean of real/fake predictions
        real_mean = torch.mean(real_pred)
        fake_mean = torch.mean(fake_pred)
        
        # Relativistic predictions
        g_loss_real = self.bce_loss(real_pred - fake_mean, torch.zeros_like(real_pred))
        g_loss_fake = self.bce_loss(fake_pred - real_mean, torch.ones_like(fake_pred))
        
        return (g_loss_real + g_loss_fake) / 2.0
        
    def discriminator_loss(self, real_pred: torch.Tensor, fake_pred: torch.Tensor) -> torch.Tensor:
        """Compute RaGAN loss for the discriminator."""
        real_mean = torch.mean(real_pred)
        fake_mean = torch.mean(fake_pred)
        
        d_loss_real = self.bce_loss(real_pred - fake_mean, torch.ones_like(real_pred))
        d_loss_fake = self.bce_loss(fake_pred - real_mean, torch.zeros_like(fake_pred))
        
        return (d_loss_real + d_loss_fake) / 2.0
