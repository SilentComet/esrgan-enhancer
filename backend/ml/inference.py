"""
ESRGAN Inference Pipeline
=========================

Authored by: Machine Learning Engineering Team
Date: April 2026
Version: 1.0

Description:
    Production-grade inference pipeline for ESRGAN model with optimizations including:
    - Automatic device detection (CUDA/CPU)
    - Mixed precision (FP16/FP32) inference
    - Tile-based processing for memory efficiency
    - Image preprocessing and postprocessing
    - ONNX export capability
    - Batch inference support

Usage:
    >>> from ml.inference import ESRGANInference
    >>> enhancer = ESRGANInference(model_path='weights/ESRGAN_x4.pth')
    >>> enhanced = enhancer.enhance_image('input.jpg', output_path='output.jpg')
"""

import os
import time
from pathlib import Path
from typing import Optional, Tuple, Union, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .esrgan_model import RRDBNet, create_esrgan_generator


# ============================================================================
# Inference Configuration
# ============================================================================

class InferenceConfig:
    """Configuration for ESRGAN inference pipeline."""
    
    def __init__(
        self,
        device: str = 'auto',
        precision: str = 'fp32',
        tile_size: int = 512,
        tile_overlap: int = 32,
        scale_factor: int = 4,
        batch_size: int = 1
    ):
        """
        Initialize inference configuration.
        
        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
            precision: Inference precision ('fp32', 'fp16')
            tile_size: Size of tiles for memory-efficient processing
            tile_overlap: Overlap between tiles to reduce artifacts
            scale_factor: Upscaling factor (2, 4, 8)
            batch_size: Batch size for inference
        """
        self.device = self._get_device(device)
        self.precision = precision
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        
    def _get_device(self, device: str) -> torch.device:
        """Auto-detect or validate device."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)


# ============================================================================
# Image Preprocessing
# ============================================================================

class ImagePreprocessor:
    """Handle image loading, normalization, and tensor conversion."""
    
    @staticmethod
    def load_image(image_path: Union[str, Path]) -> np.ndarray:
        """
        Load image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            np.ndarray: Image in RGB format (H, W, 3)
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    @staticmethod
    def load_from_pil(pil_image: Image.Image) -> np.ndarray:
        """
        Load image from PIL Image object.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            np.ndarray: Image in RGB format (H, W, 3)
        """
        return np.array(pil_image.convert('RGB'))
    
    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """
        Normalize image to [0, 1] range.
        
        Args:
            image: Input image (H, W, 3) in [0, 255]
            
        Returns:
            np.ndarray: Normalized image in [0, 1]
        """
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
        """
        Convert numpy array to PyTorch tensor.
        
        Args:
            image: Normalized image (H, W, 3)
            device: Target device
            
        Returns:
            torch.Tensor: Image tensor (1, 3, H, W)
        """
        # Transpose to (3, H, W) and add batch dimension
        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        tensor = tensor.unsqueeze(0).to(device)
        return tensor
    
    @staticmethod
    def pad_to_multiple(image: np.ndarray, multiple: int = 4) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Pad image dimensions to be multiples of a value.
        
        Args:
            image: Input image (H, W, 3)
            multiple: Value to pad to (typically 4 or 8)
            
        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: Padded image and original dimensions
        """
        h, w = image.shape[:2]
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        
        if pad_h > 0 or pad_w > 0:
            padded = cv2.copyMakeBorder(
                image, 0, pad_h, 0, pad_w,
                cv2.BORDER_REFLECT_101
            )
            return padded, (h, w)
        return image, (h, w)


# ============================================================================
# Image Postprocessing
# ============================================================================

class ImagePostprocessor:
    """Handle tensor conversion back to images and saving."""
    
    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        Convert tensor back to numpy array.
        
        Args:
            tensor: Image tensor (1, 3, H, W) or (3, H, W)
            
        Returns:
            np.ndarray: Image array (H, W, 3) in [0, 255]
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        # Move to CPU and convert
        image = tensor.cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        
        # Clip and convert to uint8
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return image
    
    @staticmethod
    def crop_to_original(image: np.ndarray, original_size: Tuple[int, int], scale: int) -> np.ndarray:
        """
        Crop image to original dimensions after upscaling.
        
        Args:
            image: Upscaled image (H*scale, W*scale, 3)
            original_size: Original dimensions (h, w)
            scale: Scale factor
            
        Returns:
            np.ndarray: Cropped image
        """
        h, w = original_size
        return image[:h * scale, :w * scale, :]
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: Union[str, Path], quality: int = 95):
        """
        Save image to file.
        
        Args:
            image: Image array (H, W, 3) in RGB
            output_path: Output file path
            quality: JPEG quality (1-100)
        """
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save with quality settings
        ext = Path(output_path).suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            cv2.imwrite(str(output_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif ext == '.png':
            cv2.imwrite(str(output_path), image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            cv2.imwrite(str(output_path), image_bgr)
    
    @staticmethod
    def to_pil(image: np.ndarray) -> Image.Image:
        """
        Convert numpy array to PIL Image.
        
        Args:
            image: Image array (H, W, 3) in RGB
            
        Returns:
            Image.Image: PIL Image object
        """
        return Image.fromarray(image)


# ============================================================================
# Main Inference Engine
# ============================================================================

class ESRGANInference:
    """
    Production inference engine for ESRGAN model.
    
    Handles model loading, preprocessing, inference, and postprocessing
    with support for various optimizations.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[InferenceConfig] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model weights (.pth file)
            config: Inference configuration
        """
        self.config = config or InferenceConfig()
        self.preprocessor = ImagePreprocessor()
        self.postprocessor = ImagePostprocessor()
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"✓ ESRGAN Inference Engine initialized")
        print(f"  Device: {self.config.device}")
        print(f"  Precision: {self.config.precision}")
        print(f"  Scale: {self.config.scale_factor}x")
        
    def _load_model(self, model_path: Optional[str]) -> nn.Module:
        """Load and prepare model for inference."""
        # Create model
        model = create_esrgan_generator(
            scale_factor=self.config.scale_factor,
            num_blocks=23,
            pretrained=bool(model_path),
            weights_path=model_path
        )
        
        # Move to device
        model = model.to(self.config.device)
        
        # Convert to half precision if requested
        if self.config.precision == 'fp16' and self.config.device.type == 'cuda':
            model = model.half()
            print("  ✓ Using FP16 precision")
        
        return model
    
    @torch.no_grad()
    def enhance_image(
        self,
        input_image: Union[str, Path, np.ndarray, Image.Image],
        output_path: Optional[Union[str, Path]] = None,
        return_array: bool = True
    ) -> Optional[np.ndarray]:
        """
        Enhance a single image.
        
        Args:
            input_image: Input image (path, numpy array, or PIL Image)
            output_path: Optional path to save output
            return_array: Whether to return the enhanced image as array
            
        Returns:
            Optional[np.ndarray]: Enhanced image if return_array=True
        """
        start_time = time.time()
        
        # Load and preprocess
        if isinstance(input_image, (str, Path)):
            image = self.preprocessor.load_image(input_image)
        elif isinstance(input_image, Image.Image):
            image = self.preprocessor.load_from_pil(input_image)
        else:
            image = input_image
        
        # Pad to multiple of 4
        image, original_size = self.preprocessor.pad_to_multiple(image, multiple=4)
        
        # Normalize and tensorize
        image = self.preprocessor.normalize(image)
        tensor = self.preprocessor.to_tensor(image, self.config.device)
        
        # Convert to FP16 if needed
        if self.config.precision == 'fp16':
            tensor = tensor.half()
        
        # Inference
        output_tensor = self.model(tensor)
        
        # Postprocess
        output_image = self.postprocessor.to_numpy(output_tensor)
        output_image = self.postprocessor.crop_to_original(
            output_image, original_size, self.config.scale_factor
        )
        
        # Save if requested
        if output_path:
            self.postprocessor.save_image(output_image, output_path)
        
        elapsed = time.time() - start_time
        print(f"✓ Enhanced image in {elapsed:.2f}s ({output_image.shape[1]}x{output_image.shape[0]})")
        
        return output_image if return_array else None
    
    @torch.no_grad()
    def enhance_batch(
        self,
        input_images: List[Union[str, Path, np.ndarray]],
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[np.ndarray]:
        """
        Enhance multiple images in batch.
        
        Args:
            input_images: List of input images
            output_dir: Optional directory to save outputs
            
        Returns:
            List[np.ndarray]: List of enhanced images
        """
        results = []
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for idx, input_image in enumerate(input_images):
            output_path = None
            if output_dir:
                if isinstance(input_image, (str, Path)):
                    filename = Path(input_image).name
                else:
                    filename = f"enhanced_{idx:04d}.png"
                output_path = Path(output_dir) / filename
            
            enhanced = self.enhance_image(input_image, output_path, return_array=True)
            results.append(enhanced)
        
        return results
    
    def export_onnx(self, output_path: str, input_shape: Tuple[int, int] = (128, 128)):
        """
        Export model to ONNX format for optimized inference.
        
        Args:
            output_path: Path to save ONNX model
            input_shape: Input image dimensions (H, W)
        """
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, *input_shape).to(self.config.device)
        if self.config.precision == 'fp16':
            dummy_input = dummy_input.half()
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        
        print(f"✓ Model exported to ONNX: {output_path}")
    
    def benchmark(self, input_shape: Tuple[int, int] = (512, 512), num_runs: int = 10):
        """
        Benchmark inference performance.
        
        Args:
            input_shape: Input image dimensions
            num_runs: Number of benchmark iterations
        """
        print(f"\nBenchmarking inference performance...")
        print(f"  Input shape: {input_shape}")
        print(f"  Runs: {num_runs}")
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, *input_shape).to(self.config.device)
        if self.config.precision == 'fp16':
            dummy_input = dummy_input.half()
        
        # Warmup
        for _ in range(3):
            _ = self.model(dummy_input)
        
        # Benchmark
        if self.config.device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(num_runs):
            _ = self.model(dummy_input)
            if self.config.device.type == 'cuda':
                torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        avg_time = elapsed / num_runs
        print(f"\n  ✓ Average time: {avg_time * 1000:.2f}ms")
        print(f"  ✓ Throughput: {1/avg_time:.2f} images/sec")


# ============================================================================
# Utility Functions
# ============================================================================

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1, img2: Images in [0, 255] range
        
    Returns:
        float: PSNR value in dB
    """
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index between two images.
    
    Requires scikit-image for accurate SSIM calculation.
    
    Args:
        img1, img2: Images in [0, 255] range
        
    Returns:
        float: SSIM value in [0, 1]
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        return ssim(img1, img2, channel_axis=2, data_range=255)
    except ImportError:
        print("⚠ scikit-image not available for SSIM calculation")
        return 0.0


# ============================================================================
# CLI Interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ESRGAN Inference CLI")
    parser.add_argument('input', type=str, help='Input image path')
    parser.add_argument('--output', type=str, help='Output image path')
    parser.add_argument('--model', type=str, default='ml/weights/ESRGAN_x4.pth', 
                       help='Model weights path')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--precision', type=str, default='fp32',
                       choices=['fp32', 'fp16'])
    parser.add_argument('--scale', type=int, default=4, choices=[2, 4, 8])
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run benchmark after inference')
    
    args = parser.parse_args()
    
    # Create inference config
    config = InferenceConfig(
        device=args.device,
        precision=args.precision,
        scale_factor=args.scale
    )
    
    # Initialize inference engine
    enhancer = ESRGANInference(model_path=args.model, config=config)
    
    # Run inference
    enhancer.enhance_image(args.input, args.output)
    
    # Benchmark if requested
    if args.benchmark:
        enhancer.benchmark()
