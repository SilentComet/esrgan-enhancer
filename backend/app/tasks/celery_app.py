"""
Celery Task Queue Configuration and Workers
============================================

Authored by: DevOps & Backend Engineering Team
Date: April 2026
Version: 1.0

Description:
    Configures the Celery application instance, registers the async
    enhancement worker task, and manages active/failed task reporting
    to the Redis status backend.
"""

import asyncio
from celery import Celery
from pathlib import Path
from datetime import datetime

from app.core.config import settings
from app.utils.logger import get_logger
from app.models.database import update_db_task

logger = get_logger(__name__)

# Initialize Celery app instance
celery_app = Celery(
    "esrgan_tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Apply settings
celery_app.conf.update(
    task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
    task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True
)


@celery_app.task(name="app.tasks.enhance_image_task", bind=True)
def enhance_image_task(self, task_id: str, input_path: str, scale_factor: int) -> dict:
    """
    Celery background worker task for running image enhancement through PyTorch/CUDA.
    
    Args:
        task_id: UUID of the enhancement task
        input_path: Local path to the raw uploaded image file
        scale_factor: Upscaling multiplier factor
        
    Returns:
        dict: Final task completion statistics
    """
    logger.info(f"Task {task_id}: Celery worker started processing")
    
    # Establish local loop context to run async database updates from within synchronous worker thread
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    def sync_update_status(status=None, progress=None, error_message=None, completed=False):
        """Helper to run async database updates from the synchronous worker thread."""
        coro = update_db_task(task_id, status, progress, error_message, completed)
        loop.run_until_complete(coro)
        
    try:
        sync_update_status(status="processing", progress=15.0)
        
        # Determine output file path
        output_path = str(Path(input_path).parent / f"{task_id}_output.png")
        
        # Ensure weights are present
        weights_path = settings.MODEL_WEIGHTS_PATH
        if not weights_path.exists():
            logger.warning(f"Task {task_id}: Weights not found at {weights_path}. Running downloader.")
            from ml.weights.download_weights import download_official_weights
            download_official_weights(weights_path)
            
        sync_update_status(progress=40.0)
        
        # Load models and process
        from ml.inference import ESRGANInference, InferenceConfig
        
        config = InferenceConfig(
            device=settings.effective_device,
            precision=settings.MODEL_PRECISION,
            scale_factor=scale_factor
        )
        
        # Initialize inference pipeline
        enhancer = ESRGANInference(model_path=str(weights_path), config=config)
        sync_update_status(progress=50.0)
        
        # Run ESRGAN upscaler
        logger.info(f"Task {task_id}: Starting model inference...")
        enhancer.enhance_image(
            input_image=input_path,
            output_path=output_path,
            return_array=False
        )
        
        # Update database with success
        logger.info(f"Task {task_id}: Celery task completed successfully")
        sync_update_status(status="completed", progress=100.0, completed=True)
        
        return {
            "task_id": task_id,
            "status": "completed",
            "progress": 100.0,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.exception(f"Task {task_id}: Celery task failed")
        sync_update_status(status="failed", error_message=str(e), completed=True)
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e)
        }
