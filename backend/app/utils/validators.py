"""
Upload Validation Utilities
===========================

Authored by: Backend Engineering Team
Date: April 2026
Version: 1.0

Description:
    Validates user file uploads strictly based on file sizes, file extensions,
    and actual image headers (magic bytes) to prevent file injection attacks.
"""

from fastapi import UploadFile
from typing import List
from pathlib import Path
from PIL import Image
import io

from app.utils.logger import get_logger

logger = get_logger(__name__)


def validate_image_file(
    file: UploadFile,
    max_file_size: int,
    allowed_extensions: List[str]
) -> None:
    """
    Perform deep validation of an uploaded file.
    
    Checks:
        1. File extension
        2. File size
        3. Real image validation via PIL (magic numbers & dimensions)
        
    Args:
        file: The uploaded file from FastAPI
        max_file_size: Maximum allowed size in bytes
        allowed_extensions: List of valid lowercase extensions (e.g. ['png', 'jpg'])
        
    Raises:
        ValueError: If validation fails
    """
    # 1. Check extension
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower().lstrip(".")
    
    if suffix not in allowed_extensions:
        raise ValueError(
            f"Unsupported file format '.{suffix}'. Allowed formats: {', '.join(allowed_extensions)}"
        )
        
    # 2. Check file size
    # FastAPI/Starlette caches small files in SpooledTemporaryFile or BytesIO.
    # We should read the size using file.size or by seeking.
    file_size = getattr(file, "size", None)
    
    if file_size is None:
        # Fallback manual check by reading and seeking back
        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)
        
    if file_size > max_file_size:
        raise ValueError(
            f"File exceeds maximum size limit of {max_file_size / 1024 / 1024:.1f}MB. "
            f"Uploaded file is {file_size / 1024 / 1024:.1f}MB."
        )
        
    # 3. Verify magic bytes and image integrity
    try:
        # Read first 1MB to verify image headers safely
        header = file.file.read(1024 * 1024)
        file.file.seek(0)  # Reset pointer
        
        img = Image.open(io.BytesIO(header))
        img.verify()  # Verifies image integrity without loading actual pixels
        
        # Check dimensions are reasonable
        width, height = img.size
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be greater than zero")
        if width > 8000 or height > 8000:
            raise ValueError("Image dimensions exceed 8000x8000 pixel maximum threshold")
            
    except Exception as e:
        logger.error(f"Failed deep image verification for {filename}: {str(e)}")
        raise ValueError("Corrupted or invalid image file. Header analysis failed.")
