"""
Enhancement API Schemas
=======================

Authored by: Backend Engineering Team
Date: April 2026
Version: 1.0

Description:
    Pydantic schemas for validating request payloads and formatting response models
    for single, batch, and task status tracking endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class EnhancementRequest(BaseModel):
    """
    Schema for manual non-multipart request parameters (if applicable).
    """
    scale_factor: int = Field(default=4, ge=2, le=8, description="Upscaling multiplier factor")


class EnhancementResponse(BaseModel):
    """
    Response schema returned immediately after an enhancement task is successfully queued.
    """
    task_id: str = Field(..., description="Unique UUID identifying the enhancement task")
    status: str = Field(default="queued", description="Initial queue state of the task")
    message: str = Field(..., description="Status message detailing queuing state")
    estimated_time_seconds: float = Field(..., description="Estimated inference execution duration")
    input_filename: str = Field(..., description="Original name of the uploaded image file")


class TaskStatusResponse(BaseModel):
    """
    Response schema for check status requests containing real-time processing statistics.
    """
    task_id: str = Field(..., description="Unique task identification code")
    status: str = Field(..., description="Task execution status: queued | processing | completed | failed | cancelled")
    progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Task progress percentage (0.0 to 100.0)")
    estimated_time_remaining: float = Field(default=0.0, description="Remaining seconds estimated until completion")
    input_filename: str = Field(..., description="Original name of the input image file")
    scale_factor: int = Field(..., description="The scale factor being applied (e.g. 4)")
    error_message: Optional[str] = Field(default=None, description="Detailed error information if task has failed")
    created_at: str = Field(..., description="UTC timestamp showing task creation time")
    completed_at: Optional[str] = Field(default=None, description="UTC timestamp showing task completion time")


class BatchEnhancementResponse(BaseModel):
    """
    Response schema returned immediately after a batch image upload request is queued.
    """
    batch_id: str = Field(..., description="Unique UUID identifying the batch request")
    task_ids: List[str] = Field(..., description="List of individual task IDs associated with this batch")
    total_images: int = Field(..., description="Count of successfully validated and queued images")
    message: str = Field(..., description="Progress status summary message")
