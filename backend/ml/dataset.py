"""
ESRGAN PyTorch Dataset Loader
=============================

Authored by: Machine Learning Engineering Team
Date: April 2026
Version: 1.0

Description:
    Implements a custom PyTorch dataset module to load pairs of Low-Resolution (LR)
    and High-Resolution (HR) images for model training. Includes real-time image
    patching, normalizations, and data augmentations (flips, rotations).
"""

import os
import random
import numpy as np
from pathlib import Path
from typing import Tuple, List, Union
import cv2
import torch
from torch.utils.data import Dataset

from ml.config import HR_PATCH_SIZE, LR_PATCH_SIZE, SCALE_FACTOR


class ESRGANDataset(Dataset):
    """
    Custom PyTorch Dataset for paired Low-Resolution and High-Resolution images.
    """
    def __init__(
        self,
        hr_dir: Union[str, Path],
        lr_dir: Optional[Union[str, Path]] = None,
        is_train: bool = True
    ):
        """
        Initialize paired dataset.
        
        Args:
            hr_dir: Directory containing High-Resolution ground truth images
            lr_dir: Optional directory containing Low-Resolution inputs. If None,
                    LR images will be downsampled dynamically from HR images.
            is_train: If True, applies random patches and flips for augmentation
        """
        self.hr_dir = Path(hr_dir)
        self.lr_dir = Path(lr_dir) if lr_dir else None
        self.is_train = is_train
        
        # Collect image file paths
        self.hr_paths = self._collect_image_paths(self.hr_dir)
        
        if self.lr_dir:
            self.lr_paths = self._collect_image_paths(self.lr_dir)
            assert len(self.hr_paths) == len(self.lr_paths), "HR and LR image counts must match!"
            self.hr_paths.sort()
            self.lr_paths.sort()
        else:
            self.lr_paths = None
            
        if len(self.hr_paths) == 0:
            raise ValueError(f"No image files found in {hr_dir}")
            
    def _collect_image_paths(self, folder: Path) -> List[Path]:
        """Utility to scan directories for supported image files."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        return [
            p for p in folder.glob("**/*")
            if p.suffix.lower() in valid_extensions
        ]
        
    def __len__(self) -> int:
        return len(self.hr_paths)
        
    def _augment(self, img_hr: np.ndarray, img_lr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random flips and rotations for robust data augmentation."""
        hflip = random.random() < 0.5
        vflip = random.random() < 0.5
        rot90 = random.random() < 0.5
        
        if hflip:
            img_hr = cv2.flip(img_hr, 1)
            img_lr = cv2.flip(img_lr, 1)
        if vflip:
            img_hr = cv2.flip(img_hr, 0)
            img_lr = cv2.flip(img_lr, 0)
        if rot90:
            # Rotate 90 degrees counter-clockwise
            img_hr = np.rot90(img_hr)
            img_lr = np.rot90(img_lr)
            
        return np.ascontiguousarray(img_hr), np.ascontiguousarray(img_lr)
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get paired item (LR, HR) for training or validation.
        
        Returns:
            Tuple[Tensor, Tensor]: Padded/Normalized LR (3, LR_H, LR_W) and HR (3, HR_H, HR_W) PyTorch Tensors
        """
        # 1. Read HR Image
        hr_path = self.hr_paths[index]
        img_hr = cv2.imread(str(hr_path), cv2.IMREAD_COLOR)
        if img_hr is None:
            raise ValueError(f"Failed to read image: {hr_path}")
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
        
        # 2. Read or Generate LR Image
        if self.lr_paths:
            lr_path = self.lr_paths[index]
            img_lr = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
            if img_lr is None:
                raise ValueError(f"Failed to read image: {lr_path}")
            img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
        else:
            # Dynamic downsampling fallback using bicubic interpolation
            h, w = img_hr.shape[:2]
            # Ensure dimensions are divisible by scale factor
            h_new, w_new = (h // SCALE_FACTOR) * SCALE_FACTOR, (w // SCALE_FACTOR) * SCALE_FACTOR
            if h_new != h or w_new != w:
                img_hr = cv2.resize(img_hr, (w_new, h_new), interpolation=cv2.INTER_LANCZOS4)
                
            img_lr = cv2.resize(
                img_hr,
                (img_hr.shape[1] // SCALE_FACTOR, img_hr.shape[0] // SCALE_FACTOR),
                interpolation=cv2.INTER_CUBIC
            )
            
        # 3. Patching in training mode
        if self.is_train:
            h_hr, w_hr = img_hr.shape[:2]
            
            # Ensure image is large enough for patching
            if h_hr < HR_PATCH_SIZE or w_hr < HR_PATCH_SIZE:
                # Resize if too small
                img_hr = cv2.resize(img_hr, (HR_PATCH_SIZE, HR_PATCH_SIZE), interpolation=cv2.INTER_LANCZOS4)
                img_lr = cv2.resize(img_lr, (LR_PATCH_SIZE, LR_PATCH_SIZE), interpolation=cv2.INTER_CUBIC)
                h_hr, w_hr = HR_PATCH_SIZE, HR_PATCH_SIZE
                
            # Pick random top-left coordinate for patch
            x_hr = random.randint(0, w_hr - HR_PATCH_SIZE)
            y_hr = random.randint(0, h_hr - HR_PATCH_SIZE)
            
            # Map coordinates to LR patch
            x_lr = x_hr // SCALE_FACTOR
            y_lr = y_hr // SCALE_FACTOR
            
            # Slice patch
            img_hr = img_hr[y_hr : y_hr + HR_PATCH_SIZE, x_hr : x_hr + HR_PATCH_SIZE, :]
            img_lr = img_lr[y_lr : y_lr + LR_PATCH_SIZE, x_lr : x_lr + LR_PATCH_SIZE, :]
            
            # Apply Augmentations
            img_hr, img_lr = self._augment(img_hr, img_lr)
            
        # 4. Normalize to [0, 1] range and transpose to (C, H, W)
        tensor_hr = torch.from_numpy(np.transpose(img_hr.astype(np.float32) / 255.0, (2, 0, 1)))
        tensor_lr = torch.from_numpy(np.transpose(img_lr.astype(np.float32) / 255.0, (2, 0, 1)))
        
        return tensor_lr, tensor_hr
