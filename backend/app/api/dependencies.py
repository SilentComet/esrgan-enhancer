"""
API Route Dependencies
======================

Authored by: Backend Engineering Team
Date: April 2026
Version: 1.0

Description:
    Declares FastAPI dependencies such as asynchronous database sessions,
    API key validation checks, and security access controls.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import AsyncSessionLocal
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Asynchronous database session generator dependency.
    Yields a session object and automatically closes it upon request termination.
    
    Yields:
        AsyncSession: Active SQLAlchemy async session database handler
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session encountered transaction error: {str(e)}")
            await session.rollback()
            raise
        finally:
            await session.close()
