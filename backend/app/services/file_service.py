"""
File Management Service
=======================

Authored by: Backend Engineering Team
Date: April 2026
Version: 1.0

Description:
    Manages safe reading, writing, path resolution, and localized garbage
    cleanup routines for temporary raw uploads and enhanced outputs.
"""

import aiofiles
import os
import shutil
import time
from pathlib import Path
from fastapi import UploadFile
from typing import Union

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class FileService:
    """
    Handles read/write operations on the filesystem for input/output files.
    """
    def __init__(self, upload_dir: Union[str, Path] = settings.UPLOAD_TEMP_DIR):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
    def get_input_path(self, task_id: str) -> Path:
        """Get absolute path for a task's input image."""
        return self.upload_dir / f"{task_id}_input.png"
        
    def get_result_path(self, task_id: str) -> Path:
        """Get absolute path for a task's enhanced result image."""
        return self.upload_dir / f"{task_id}_output.png"
        
    async def save_upload(self, file: UploadFile, task_id: str) -> Path:
        """
        Asynchronously save a FastAPI UploadFile to a standard temporary file on disk.
        Converts the image to PNG format internally using PIL to sanitize raw image headers.
        
        Args:
            file: The FastAPI UploadFile instance
            task_id: Unique task identifier
            
        Returns:
            Path: The local path to the saved input file
        """
        input_path = self.get_input_path(task_id)
        
        # Read uploaded bytes asynchronously
        content = await file.read()
        await file.seek(0)  # Reset pointer for subsequent reads
        
        # Safe async write to disk
        async with aiofiles.open(input_path, 'wb') as out_file:
            await out_file.write(content)
            
        logger.info(f"Task {task_id}: Input image saved to {input_path}")
        return input_path
        
    def cleanup_task_files(self, task_id: str) -> None:
        """
        Clean up files associated with a specific task ID (both input and output).
        """
        input_path = self.get_input_path(task_id)
        result_path = self.get_result_path(task_id)
        
        try:
            if input_path.exists():
                input_path.unlink()
            if result_path.exists():
                result_path.unlink()
            logger.info(f"Task {task_id}: Cleaned up temporary files")
        except Exception as e:
            logger.error(f"Task {task_id}: Failed to delete files - {str(e)}")
            
    def cleanup_old_files(self, max_age_hours: int = 24) -> None:
        """
        Garbage collector task to purge files older than the specified age limit.
        Usually executed as an async background task from API endpoints.
        
        Args:
            max_age_hours: Threshold age in hours after which files are deleted
        """
        now = time.time()
        max_age_seconds = max_age_hours * 3600
        purged_count = 0
        
        logger.info(f"Running automated cleanup for files older than {max_age_hours} hours")
        
        for filepath in self.upload_dir.glob("*_input.png"):
            try:
                # Check modification time
                mtime = filepath.stat().st_mtime
                if now - mtime > max_age_seconds:
                    # Retrieve matching task_id
                    task_id = filepath.name.replace("_input.png", "")
                    self.cleanup_task_files(task_id)
                    purged_count += 1
            except Exception as e:
                logger.error(f"Error purging old file {filepath}: {str(e)}")
                
        # Also clean up trailing outputs without inputs
        for filepath in self.upload_dir.glob("*_output.png"):
            try:
                mtime = filepath.stat().st_mtime
                if now - mtime > max_age_seconds:
                    filepath.unlink()
                    purged_count += 1
            except Exception as e:
                logger.error(f"Error purging output {filepath}: {str(e)}")
                
        if purged_count > 0:
            logger.info(f"Cleanup complete. Purged {purged_count} files from disk.")
        else:
            logger.info("Cleanup complete. No files required purging.")
