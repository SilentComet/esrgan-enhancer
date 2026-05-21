"""
ML Model Configurations
=======================

Authored by: Machine Learning Engineering Team
Date: April 2026
Version: 1.0

Description:
    Declares hyperparameters for ESRGAN model training and inference.
    Synchronizes runtime parameters with app/core/config settings.
"""

from pathlib import Path
from app.core.config import settings

# Inference specific configurations
INFERENCE_DEVICE = settings.effective_device
INFERENCE_PRECISION = settings.MODEL_PRECISION
TILE_SIZE = settings.TILE_SIZE
TILE_OVERLAP = settings.TILE_OVERLAP

# Model Architecture configurations
SCALE_FACTOR = settings.SCALE_FACTOR
NUM_BLOCKS = 23  # Wang et al. original paper configuration
NUM_CHANNELS = 64
GROWTH_CHANNEL = 32

# Model weights
MODEL_WEIGHTS_DIR = Path(__file__).parent / "weights"
DEFAULT_WEIGHTS_PATH = MODEL_WEIGHTS_DIR / f"ESRGAN_x4.pth"

# ============================================================================
# Training Configuration Parameters
# ============================================================================

# Dataset settings
TRAIN_DATASET_DIR = Path("./data/train")
VAL_DATASET_DIR = Path("./data/val")
HR_PATCH_SIZE = 128
LR_PATCH_SIZE = HR_PATCH_SIZE // SCALE_FACTOR

# Optimizers & Loss weights
LR_G = 1e-4
LR_D = 1e-4
BETA1 = 0.9
BETA2 = 0.999

# Loss factors
LOSS_PIXEL_WEIGHT = 1e-2     # L1 loss coefficient
LOSS_ADVERSARIAL_WEIGHT = 5e-3 # GAN loss coefficient
LOSS_PERCEPTUAL_WEIGHT = 1.0  # VGG Perceptual loss coefficient

# Learning rate scheduling
DECAY_ITERATIONS = [50000, 100000, 200000, 300000]
LR_DECAY_FACTOR = 0.5

# Training details
NUM_EPOCHS = 100
BATCH_SIZE = 16
NUM_WORKERS = 4
CHECKPOINT_INTERVAL = 5000  # steps
