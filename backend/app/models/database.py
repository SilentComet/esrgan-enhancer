"""
Database ORM Models and Session Pool
====================================

Authored by: Backend Engineering Team
Date: April 2026
Version: 1.0

Description:
    Configures SQLAlchemy ORM with async SQLite (aiosqlite) support
    to manage persisting and querying task histories.
"""

import sys
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import Column, String, Integer, Float, Text, DateTime
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from sqlalchemy.future import select

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# SQLAlchemy declarative base
Base = declarative_base()


class TaskModel(Base):
    """
    SQLAlchemy table model representing individual image enhancement tasks.
    """
    __tablename__ = "enhancement_tasks"
    
    id = Column(String(36), primary_key=True, index=True)
    status = Column(String(20), default="queued", nullable=False)
    progress = Column(Float, default=0.0, nullable=False)
    input_filename = Column(String(255), nullable=False)
    scale_factor = Column(Integer, default=4, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert SQLAlchemy object to dictionary representation."""
        return {
            "task_id": self.id,
            "status": self.status,
            "progress": self.progress,
            "input_filename": self.input_filename,
            "scale_factor": self.scale_factor,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else "",
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "estimated_time_remaining": 0.0 # Will be populated dynamically in service
        }


# Async database connection configuration
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DB_ECHO,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def init_db() -> None:
    """
    Initialize database schema by creating all required tables if they don't exist.
    """
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✓ Database tables initialized successfully")
    except Exception as e:
        logger.exception("Failed to initialize database schema")
        sys.exit(1)


# Database transaction functions
async def create_db_task(
    task_id: str,
    input_filename: str,
    scale_factor: int
) -> Dict[str, Any]:
    """Create a new task record in the database."""
    async with AsyncSessionLocal() as session:
        async with session.begin():
            new_task = TaskModel(
                id=task_id,
                status="queued",
                progress=0.0,
                input_filename=input_filename,
                scale_factor=scale_factor,
                created_at=datetime.utcnow()
            )
            session.add(new_task)
        await session.refresh(new_task)
        return new_task.to_dict()


async def get_db_task(task_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a specific task by its UUID from the database."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(TaskModel).where(TaskModel.id == task_id))
        task = result.scalars().first()
        return task.to_dict() if task else None


async def update_db_task(
    task_id: str,
    status: Optional[str] = None,
    progress: Optional[float] = None,
    error_message: Optional[str] = None,
    completed: bool = False
) -> Optional[Dict[str, Any]]:
    """Update an existing task record in the database."""
    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(select(TaskModel).where(TaskModel.id == task_id))
            task = result.scalars().first()
            if not task:
                return None
                
            if status is not None:
                task.status = status
            if progress is not None:
                task.progress = progress
            if error_message is not None:
                task.error_message = error_message
            if completed:
                task.completed_at = datetime.utcnow()
                task.progress = 100.0
                
        return task.to_dict()


async def get_active_tasks() -> List[Dict[str, Any]]:
    """Retrieve a list of currently active or queued tasks."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(TaskModel).where(TaskModel.status.in_(["queued", "processing"]))
        )
        tasks = result.scalars().all()
        return [task.to_dict() for task in tasks]
