"""
FastAPI Backend Application - Main Entry Point
=============================================

Authored by: Backend Engineering Team
Date: April 2026
Version: 1.0

Description:
    Production-ready FastAPI application serving ESRGAN image enhancement API.
    Includes comprehensive middleware stack, CORS configuration, rate limiting,
    health checks, and async request handling.

Features:
    - RESTful API with OpenAPI documentation
    - Async file handling and processing
    - Redis-backed task queue with Celery
    - Request validation and error handling
    - Prometheus metrics endpoint
    - Structured logging

Usage:
    Development:
        uvicorn app.main:app --reload --port 8000
    
    Production:
        uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.security import setup_security
from app.utils.logger import setup_logger, get_logger
from app.api.routes import router as api_router


# ============================================================================
# Application Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle with startup and shutdown events.
    
    Handles:
        - Model loading and caching
        - Database connections
        - Redis connection pool
        - Cleanup on shutdown
    """
    logger = get_logger(__name__)
    
    # Startup
    logger.info("=" * 80)
    logger.info("Starting ESRGAN Enhancement API Server")
    logger.info("=" * 80)
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug Mode: {settings.DEBUG}")
    logger.info(f"CORS Origins: {settings.ALLOWED_ORIGINS}")
    logger.info(f"CUDA Enabled: {settings.ENABLE_CUDA}")
    logger.info(f"Model Precision: {settings.MODEL_PRECISION}")
    
    # Create necessary directories
    settings.UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Upload directory: {settings.UPLOAD_TEMP_DIR}")
    
    # Initialize database
    from app.models.database import init_db
    await init_db()
    
    # Initialize model (lazy loading in worker)
    logger.info("Model initialization deferred to first request")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ESRGAN Enhancement API Server")
    logger.info("Cleaning up resources...")
    logger.info("=" * 80)


# ============================================================================
# Application Factory
# ============================================================================

def create_application() -> FastAPI:
    """
    Create and configure FastAPI application instance.
    
    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title="ESRGAN Image Enhancement API",
        description="Production-grade API for image super-resolution using ESRGAN neural network",
        version="1.0.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan
    )
    
    # Setup logging
    setup_logger()
    logger = get_logger(__name__)
    
    # ========================================================================
    # Middleware Stack
    # ========================================================================
    
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        max_age=settings.CORS_MAX_AGE,
    )
    logger.info("✓ CORS middleware configured")
    
    # GZip Compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    logger.info("✓ GZip compression enabled")
    
    # Request Timing Middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        """Add X-Process-Time header to responses."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        return response
    
    # ========================================================================
    # Rate Limiting
    # ========================================================================
    
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    logger.info("✓ Rate limiting configured")
    
    # ========================================================================
    # Security Headers
    # ========================================================================
    
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Add security headers to all responses."""
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response
    
    # ========================================================================
    # Exception Handlers
    # ========================================================================
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle validation errors with detailed response."""
        logger.warning(f"Validation error on {request.url.path}: {exc.errors()}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "detail": exc.errors(),
                "body": exc.body,
                "message": "Request validation failed"
            }
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle value errors."""
        logger.error(f"ValueError on {request.url.path}: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc), "message": "Invalid request parameters"}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected errors."""
        logger.exception(f"Unhandled exception on {request.url.path}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error" if not settings.DEBUG else str(exc),
                "message": "An unexpected error occurred"
            }
        )
    
    # ========================================================================
    # Route Registration
    # ========================================================================
    
    # Health check endpoint (no rate limiting)
    @app.get("/health", tags=["System"])
    async def health_check() -> Dict[str, Any]:
        """
        Health check endpoint for monitoring and load balancers.
        
        Returns:
            dict: System health status
        """
        import torch
        return {
            "status": "healthy",
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    
    # Root endpoint
    @app.get("/", tags=["System"])
    async def root():
        """API root endpoint."""
        return {
            "message": "ESRGAN Image Enhancement API",
            "version": "1.0.0",
            "docs": "/docs" if settings.DEBUG else "Documentation disabled in production",
            "health": "/health"
        }
    
    # API routes
    app.include_router(api_router, prefix="/api/v1")
    logger.info("✓ API routes registered")
    
    return app


# ============================================================================
# Application Instance
# ============================================================================

app = create_application()


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    """Run application with uvicorn."""
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS if not settings.RELOAD else 1,
        log_level=settings.LOG_LEVEL.lower()
    )
