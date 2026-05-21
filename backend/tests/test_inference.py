"""
ESRGAN Inference Unit Tests
===========================

Authored by: Machine Learning Engineering Team
Date: April 2026
Version: 1.0

Description:
    Runs PyTorch level unit tests on preprocessors, postprocessors, and the main
    ESRGANInference engine using dynamic synthetic image arrays.
"""

import sys
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# Adjust path for import resolutions
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.inference import ESRGANInference, InferenceConfig, ImagePreprocessor, ImagePostprocessor


def test_image_preprocessor_padding():
    """Verify that padding utility pads dimensions to multiples of 4 correctly."""
    preprocessor = ImagePreprocessor()
    
    # Create an arbitrary un-divisible image size (e.g. 15x13)
    dummy_img = np.zeros((15, 13, 3), dtype=np.uint8)
    padded, original_size = preprocessor.pad_to_multiple(dummy_img, multiple=4)
    
    assert padded.shape[0] == 16
    assert padded.shape[1] == 16
    assert original_size == (15, 13)


def test_image_postprocessor_crop():
    """Verify that cropping utility restores dimensions to correct shapes post-upscaling."""
    postprocessor = ImagePostprocessor()
    
    # Padded and scaled up image (e.g. padded to 16, scaled by 4 = 64)
    dummy_scaled = np.zeros((64, 64, 3), dtype=np.uint8)
    cropped = postprocessor.crop_to_original(dummy_scaled, (15, 13), scale=4)
    
    # 15 * 4 = 60, 13 * 4 = 52
    assert cropped.shape[0] == 60
    assert cropped.shape[1] == 52


def test_inference_pipeline_execution():
    """
    Test that running upscaling inference on a dummy PIL Image runs without throwing
    errors, and upscales it precisely by the target scale factor.
    """
    # Use CPU mode for quick dry testing
    config = InferenceConfig(
        device="cpu",
        precision="fp32",
        scale_factor=4
    )
    
    # Create dummy generator model without loading pth weights
    enhancer = ESRGANInference(model_path=None, config=config)
    
    # Generate tiny PIL image (16x16 pixels)
    img = Image.fromarray(np.uint8(np.random.rand(16, 16, 3) * 255))
    
    # Run upscaling
    sr_output = enhancer.enhance_image(img, return_array=True)
    
    assert sr_output is not None
    # Verify dimensions (16 * 4 = 64)
    assert sr_output.shape[0] == 64
    assert sr_output.shape[1] == 64
    assert sr_output.shape[2] == 3  # RGB channels
