"""
API Routes - Image Enhancement Endpoints
========================================

Authored by: Backend Engineering Team
Date: April 2026
Version: 1.0

Description:
    RESTful API endpoints for ESRGAN image enhancement with async processing,
    file upload handling, and task status tracking.

Endpoints:
    POST /api/v1/enhance - Enhance single image
    POST /api/v1/enhance/batch - Enhance multiple images
    GET /api/v1/task/{task_id} - Get task status
    GET /api/v1/result/{task_id} - Download enhanced image
"""

import asyncio
import uuid
from pathlib import Path
from typing import Optional, List

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    status,
    BackgroundTasks
)
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.schemas.enhancement import (
    EnhancementRequest,
    EnhancementResponse,
    TaskStatusResponse,
    BatchEnhancementResponse
)
from app.services.enhancement_service import EnhancementService
from app.services.file_service import FileService
from app.utils.validators import validate_image_file
from app.utils.logger import get_logger


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(tags=["Enhancement"])
limiter = Limiter(key_func=get_remote_address)
logger = get_logger(__name__)

# Initialize services
file_service = FileService()
enhancement_service = EnhancementService()


# ============================================================================
# Enhancement Endpoints
# ============================================================================

@router.post(
    "/enhance",
    response_model=EnhancementResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enhance Single Image",
    description="Upload a single image for 4× super-resolution enhancement using ESRGAN"
)
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def enhance_image(
    file: UploadFile = File(..., description="Image file to enhance (JPG, PNG, WEBP)"),
    scale_factor: int = Form(default=4, ge=2, le=8, description="Upscaling factor"),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> EnhancementResponse:
    """
    Upload and enhance a single image.
    
    Process:
        1. Validate file (type, size, format)
        2. Save to temporary storage
        3. Queue async enhancement task
        4. Return task ID for status tracking
    
    Args:
        file: Image file upload
        scale_factor: Desired upscaling factor (2, 4, or 8)
        background_tasks: FastAPI background tasks
        
    Returns:
        EnhancementResponse: Task ID and estimated processing time
        
    Raises:
        HTTPException: If validation fails or processing error occurs
    """
    task_id = str(uuid.uuid4())
    
    try:
        # Validate file
        validate_image_file(file, settings.MAX_FILE_SIZE, settings.ALLOWED_EXTENSIONS)
        logger.info(f"Task {task_id}: Received file {file.filename} ({file.size} bytes)")
        
        # Save uploaded file
        input_path = await file_service.save_upload(file, task_id)
        logger.info(f"Task {task_id}: Saved to {input_path}")
        
        # Get image dimensions for time estimation
        from PIL import Image
        with Image.open(input_path) as img:
            width, height = img.size
        
        # Estimate processing time (rough heuristic)
        pixels = width * height
        estimated_time = enhancement_service.estimate_processing_time(pixels, scale_factor)
        
        # Queue enhancement task
        task = enhancement_service.queue_enhancement(
            task_id=task_id,
            input_path=str(input_path),
            scale_factor=scale_factor
        )
        
        logger.info(f"Task {task_id}: Queued for processing (ETA: {estimated_time}s)")
        
        # Schedule cleanup in background
        background_tasks.add_task(
            file_service.cleanup_old_files,
            max_age_hours=24
        )
        
        return EnhancementResponse(
            task_id=task_id,
            status="queued",
            message="Image queued for enhancement",
            estimated_time_seconds=estimated_time,
            input_filename=file.filename
        )
        
    except ValueError as e:
        logger.error(f"Task {task_id}: Validation error - {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Task {task_id}: Enhancement failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enhancement failed: {str(e)}"
        )


@router.post(
    "/enhance/batch",
    response_model=BatchEnhancementResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enhance Multiple Images",
    description="Upload multiple images for batch enhancement"
)
@limiter.limit(f"{settings.RATE_LIMIT_PER_HOUR}/hour")
async def enhance_batch(
    files: List[UploadFile] = File(..., description="Image files to enhance (max 10)"),
    scale_factor: int = Form(default=4, ge=2, le=8),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> BatchEnhancementResponse:
    """
    Upload and enhance multiple images in batch.
    
    Args:
        files: List of image files (max 10)
        scale_factor: Upscaling factor
        background_tasks: Background tasks
        
    Returns:
        BatchEnhancementResponse: List of task IDs
    """
    # Validate batch size
    if len(files) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 images per batch"
        )
    
    batch_id = str(uuid.uuid4())
    task_ids = []
    
    logger.info(f"Batch {batch_id}: Processing {len(files)} files")
    
    for idx, file in enumerate(files):
        try:
            task_id = f"{batch_id}_{idx}"
            
            # Validate and save
            validate_image_file(file, settings.MAX_FILE_SIZE, settings.ALLOWED_EXTENSIONS)
            input_path = await file_service.save_upload(file, task_id)
            
            # Queue task
            enhancement_service.queue_enhancement(
                task_id=task_id,
                input_path=str(input_path),
                scale_factor=scale_factor
            )
            
            task_ids.append(task_id)
            logger.info(f"Batch {batch_id}: Queued task {task_id} ({file.filename})")
            
        except Exception as e:
            logger.error(f"Batch {batch_id}: Failed to queue {file.filename} - {str(e)}")
            # Continue with remaining files
            continue
    
    if not task_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid images in batch"
        )
    
    return BatchEnhancementResponse(
        batch_id=batch_id,
        task_ids=task_ids,
        total_images=len(task_ids),
        message=f"Batch queued: {len(task_ids)} images"
    )


# ============================================================================
# Task Status Endpoints
# ============================================================================

@router.get(
    "/task/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get Task Status",
    description="Check the status of an enhancement task"
)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    Get the current status of an enhancement task.
    
    Args:
        task_id: Unique task identifier
        
    Returns:
        TaskStatusResponse: Current task status and details
    """
    try:
        status_info = enhancement_service.get_task_status(task_id)
        
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        return TaskStatusResponse(**status_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get status for task {task_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve task status"
        )


# ============================================================================
# Result Download Endpoints
# ============================================================================

@router.get(
    "/result/{task_id}",
    response_class=FileResponse,
    summary="Download Enhanced Image",
    description="Download the enhanced image result"
)
async def get_result(task_id: str) -> FileResponse:
    """
    Download the enhanced image for a completed task.
    
    Args:
        task_id: Unique task identifier
        
    Returns:
        FileResponse: Enhanced image file
    """
    try:
        # Check task status
        status_info = enhancement_service.get_task_status(task_id)
        
        if not status_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found"
            )
        
        if status_info['status'] != 'completed':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Task not completed (status: {status_info['status']})"
            )
        
        # Get result file path
        result_path = file_service.get_result_path(task_id)
        
        if not result_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Result file not found"
            )
        
        # Determine filename
        original_filename = status_info.get('input_filename', 'enhanced.png')
        filename_parts = Path(original_filename).stem
        output_filename = f"{filename_parts}_enhanced.png"
        
        logger.info(f"Task {task_id}: Serving result {output_filename}")
        
        return FileResponse(
            path=result_path,
            media_type="image/png",
            filename=output_filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to serve result for task {task_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve result"
        )


# ============================================================================
# Utility Endpoints
# ============================================================================

@router.get(
    "/models/info",
    summary="Get Model Information",
    description="Get information about available models and configurations"
)
async def get_model_info() -> dict:
    """
    Get information about available ESRGAN models.
    
    Returns:
        dict: Model configuration and capabilities
    """
    import torch
    
    return {
        "model": "ESRGAN (RRDBNet)",
        "scale_factors": [2, 4, 8],
        "max_file_size_mb": settings.MAX_FILE_SIZE / 1024 / 1024,
        "supported_formats": settings.ALLOWED_EXTENSIONS,
        "device": settings.effective_device,
        "cuda_available": torch.cuda.is_available(),
        "precision": settings.MODEL_PRECISION,
        "version": "1.0.0"
    }


@router.delete(
    "/task/{task_id}",
    summary="Cancel Task",
    description="Cancel a queued or running enhancement task"
)
async def cancel_task(task_id: str) -> dict:
    """
    Cancel an enhancement task.
    
    Args:
        task_id: Task to cancel
        
    Returns:
        dict: Cancellation confirmation
    """
    try:
        result = enhancement_service.cancel_task(task_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task {task_id} not found or already completed"
            )
        
        logger.info(f"Task {task_id}: Cancelled by user")
        
        return {
            "message": "Task cancelled successfully",
            "task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to cancel task {task_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel task"
        )
