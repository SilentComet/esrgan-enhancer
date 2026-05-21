"""
Image Enhancement Service
=========================

Authored by: Backend Engineering Team
Date: April 2026
Version: 1.0

Description:
    Coordinates image super-resolution tasks. Provides dual processing modes:
    1. Distributed async task execution using Celery + Redis.
    2. Local multi-threaded task runner using PyTorch on a thread pool
       as a zero-dependency fallback for simplified local execution.
"""

import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from app.core.config import settings
from app.utils.logger import get_logger
from app.models.database import (
    create_db_task,
    get_db_task,
    update_db_task,
    get_active_tasks
)

logger = get_logger(__name__)

# Lazy loaded cached local model instance
_local_enhancer = None
_thread_pool = ThreadPoolExecutor(max_workers=2)


class EnhancementService:
    """
    Orchestration service for queueing, executing, and tracking ESRGAN enhancement tasks.
    """
    
    def estimate_processing_time(self, pixels: int, scale_factor: int) -> float:
        """
        Estimate enhancement execution time based on input pixels and scale.
        
        Args:
            pixels: Total pixels (width * height)
            scale_factor: Scale factor (2, 4, or 8)
            
        Returns:
            float: Estimated seconds
        """
        # Heuristic: 512x512 (262k pixels) takes ~1s on GPU, ~12s on CPU
        device = settings.effective_device
        
        if device == "cuda":
            base_time = 0.5
            pixel_factor = 2.0e-6
        else:
            base_time = 3.0
            pixel_factor = 4.0e-5
            
        scale_mult = {2: 0.5, 4: 1.0, 8: 3.0}.get(scale_factor, 1.0)
        return round(max(0.5, (base_time + pixels * pixel_factor) * scale_mult), 1)
        
    def queue_enhancement(self, task_id: str, input_path: str, scale_factor: int) -> Dict[str, Any]:
        """
        Queue a task for async processing.
        First attempts to route through Celery. If connection fails or URL not set,
        spawns a local ThreadPool task to process in the background.
        
        Args:
            task_id: UUID of the task
            input_path: Local path to raw image upload
            scale_factor: Target scale factor
            
        Returns:
            dict: Initial task status dictionary
        """
        filename = Path(input_path).name
        
        # 1. Initialize task record in SQLite (async database call)
        # We run it synchronously by scheduling on the running event loop or using a future
        loop = asyncio.get_event_loop()
        if loop.is_running():
            db_coro = create_db_task(task_id, filename, scale_factor)
            task_info = asyncio.run_coroutine_threadsafe(db_coro, loop).result()
        else:
            task_info = loop.run_until_complete(create_db_task(task_id, filename, scale_factor))
            
        # 2. Choose backend execution runner
        use_celery = False
        try:
            import redis
            # Check if redis is reachable
            r = redis.Redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
            r.ping()
            use_celery = True
        except Exception:
            logger.warning("Redis is unreachable. Falling back to local ThreadPool enhancement runner.")
            
        if use_celery:
            try:
                # Dispatch to Celery worker
                from app.tasks.celery_app import enhance_image_task
                enhance_image_task.delay(task_id, input_path, scale_factor)
                logger.info(f"Task {task_id}: Dispatched successfully to Celery")
            except Exception as ce:
                logger.error(f"Failed to queue to Celery: {str(ce)}. Falling back to local runner.")
                use_celery = False
                
        if not use_celery:
            # Fallback to local thread pool
            _thread_pool.submit(
                self._run_local_enhancement,
                task_id,
                input_path,
                scale_factor
            )
            
        return task_info
        
    def _run_local_enhancement(self, task_id: str, input_path: str, scale_factor: int) -> None:
        """
        Synchronous thread pool runner for performing ML inference locally.
        Updates task status iteratively in SQLite.
        """
        global _local_enhancer
        loop = asyncio.get_event_loop()
        
        def update_status(status=None, progress=None, error_message=None, completed=False):
            """Sync helper to trigger async DB updates safely from worker thread."""
            coro = update_db_task(task_id, status, progress, error_message, completed)
            asyncio.run_coroutine_threadsafe(coro, loop).result()
            
        try:
            logger.info(f"Task {task_id}: Local worker thread started")
            update_status(status="processing", progress=15.0)
            
            # Create outputs folder
            output_path = str(Path(input_path).parent / f"{task_id}_output.png")
            
            # Lazy initialize model
            if _local_enhancer is None:
                logger.info("Initializing cached local ESRGAN Inference Engine")
                from ml.inference import ESRGANInference, InferenceConfig
                
                # Setup model weight download path
                weights_path = settings.MODEL_WEIGHTS_PATH
                if not weights_path.exists():
                    logger.warning(f"Weights not found at {weights_path}. Running downloader.")
                    from ml.weights.download_weights import download_official_weights
                    download_official_weights(weights_path)
                    
                config = InferenceConfig(
                    device=settings.effective_device,
                    precision=settings.MODEL_PRECISION,
                    scale_factor=scale_factor
                )
                _local_enhancer = ESRGANInference(model_path=str(weights_path), config=config)
                
            update_status(progress=40.0)
            
            # Execute inference
            logger.info(f"Task {task_id}: Executing upscaling inference...")
            _local_enhancer.enhance_image(
                input_image=input_path,
                output_path=output_path,
                return_array=False
            )
            
            logger.info(f"Task {task_id}: Upscaling completed successfully")
            update_status(status="completed", progress=100.0, completed=True)
            
        except Exception as e:
            logger.exception(f"Task {task_id}: Local processing thread failed")
            update_status(status="failed", error_message=str(e), completed=True)
            
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a task status from the SQLite database.
        
        Args:
            task_id: UUID of the task
            
        Returns:
            dict: Task status info or None if not found
        """
        loop = asyncio.get_event_loop()
        if loop.is_running():
            db_coro = get_db_task(task_id)
            task_info = asyncio.run_coroutine_threadsafe(db_coro, loop).result()
        else:
            task_info = loop.run_until_complete(get_db_task(task_id))
            
        if task_info and task_info["status"] in ["queued", "processing"]:
            # Populate a dynamic remaining time estimation
            from PIL import Image
            input_path = settings.UPLOAD_TEMP_DIR / f"{task_id}_input.png"
            if input_path.exists():
                try:
                    with Image.open(input_path) as img:
                        pixels = img.width * img.height
                    total_est = self.estimate_processing_time(pixels, task_info["scale_factor"])
                    # Subtract elapsed time since creation
                    created_dt = datetime.fromisoformat(task_info["created_at"])
                    elapsed = (datetime.utcnow() - created_dt).total_seconds()
                    task_info["estimated_time_remaining"] = max(0.5, total_est - elapsed)
                except Exception:
                    task_info["estimated_time_remaining"] = 5.0
            else:
                task_info["estimated_time_remaining"] = 5.0
                
        return task_info
        
    def cancel_task(self, task_id: str) -> bool:
        """
        Mark an active task as cancelled in database.
        
        Args:
            task_id: UUID of the task to cancel
            
        Returns:
            bool: True if task found and cancelled, else False
        """
        status_info = self.get_task_status(task_id)
        if not status_info or status_info["status"] in ["completed", "failed", "cancelled"]:
            return False
            
        loop = asyncio.get_event_loop()
        coro = update_db_task(task_id, status="cancelled", completed=True)
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, loop).result()
        else:
            loop.run_until_complete(coro)
            
        # Clean up temporary uploaded files
        from app.services.file_service import FileService
        FileService().cleanup_task_files(task_id)
        
        return True
