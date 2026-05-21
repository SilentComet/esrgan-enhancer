"""
Application Configuration
========================

Authored by: Backend Engineering Team
Date: April 2026
Version: 1.0

Description:
    Centralized configuration management using Pydantic Settings.
    All environment variables and application constants defined here.
    
Environment Variables:
    See .env.example for full documentation
"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration loaded from environment variables.
    
    Uses Pydantic Settings for type validation and automatic loading
    from .env files.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # ========================================================================
    # Application Settings
    # ========================================================================
    
    APP_NAME: str = "ESRGAN Enhancement API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", pattern="^(development|staging|production)$")
    DEBUG: bool = Field(default=True)
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000, ge=1024, le=65535)
    WORKERS: int = Field(default=2, ge=1, le=16)
    RELOAD: bool = Field(default=False)
    
    # ========================================================================
    # Security Settings
    # ========================================================================
    
    SECRET_KEY: str = Field(
        default="change-this-to-a-secure-random-key-in-production",
        min_length=32
    )
    API_KEY_HEADER: str = Field(default="X-API-Key")
    
    # CORS Configuration
    ALLOWED_ORIGINS_STR: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="ALLOWED_ORIGINS"
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_MAX_AGE: int = Field(default=600, ge=0)
    
    @property
    def ALLOWED_ORIGINS(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS_STR.split(",")]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=30, ge=1)
    RATE_LIMIT_PER_HOUR: int = Field(default=500, ge=1)
    
    # ========================================================================
    # File Upload Settings
    # ========================================================================
    
    MAX_FILE_SIZE: int = Field(default=10_485_760, ge=1024)  # 10MB
    ALLOWED_EXTENSIONS: List[str] = Field(
        default=["jpg", "jpeg", "png", "webp"]
    )
    UPLOAD_TEMP_DIR: Path = Field(default=Path("./tmp/uploads"))
    
    @field_validator("UPLOAD_TEMP_DIR", mode="before")
    @classmethod
    def create_upload_dir(cls, v):
        """Ensure upload directory exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # ========================================================================
    # ML Model Configuration
    # ========================================================================
    
    # Device Configuration
    ENABLE_CUDA: bool = Field(default=True)
    DEVICE: str = Field(default="auto", pattern="^(auto|cuda|cpu)$")
    MODEL_PRECISION: str = Field(default="fp32", pattern="^(fp32|fp16)$")
    
    # Model Paths
    MODEL_WEIGHTS_PATH: Path = Field(default=Path("./ml/weights/ESRGAN_x4.pth"))
    MODEL_ONNX_PATH: Path = Field(default=Path("./ml/weights/ESRGAN_x4.onnx"))
    
    # Inference Settings
    SCALE_FACTOR: int = Field(default=4, ge=2, le=8)
    TILE_SIZE: int = Field(default=512, ge=128, le=2048)
    TILE_OVERLAP: int = Field(default=32, ge=0, le=128)
    BATCH_SIZE: int = Field(default=1, ge=1, le=16)
    
    # ========================================================================
    # Redis & Celery Configuration
    # ========================================================================
    
    REDIS_URL: str = Field(default="redis://localhost:6379/0")
    REDIS_MAX_CONNECTIONS: int = Field(default=50, ge=10)
    
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/1")
    CELERY_TASK_TIME_LIMIT: int = Field(default=300, ge=60)  # 5 minutes
    CELERY_TASK_SOFT_TIME_LIMIT: int = Field(default=240, ge=30)  # 4 minutes
    
    # ========================================================================
    # Database Configuration
    # ========================================================================
    
    DATABASE_URL: str = Field(default="sqlite+aiosqlite:///./esrgan.db")
    DB_ECHO: bool = Field(default=False)
    DB_POOL_SIZE: int = Field(default=5, ge=1)
    DB_MAX_OVERFLOW: int = Field(default=10, ge=0)
    
    # ========================================================================
    # Logging Configuration
    # ========================================================================
    
    LOG_LEVEL: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    LOG_FORMAT: str = Field(default="json", pattern="^(json|text)$")
    LOG_FILE: Optional[Path] = Field(default=None)
    
    @field_validator("LOG_FILE", mode="before")
    @classmethod
    def create_log_dir(cls, v):
        """Ensure log directory exists if log file specified."""
        if v:
            path = Path(v)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return None
    
    # ========================================================================
    # Monitoring Configuration
    # ========================================================================
    
    ENABLE_METRICS: bool = Field(default=True)
    METRICS_PORT: int = Field(default=9090, ge=1024, le=65535)
    
    # ========================================================================
    # Computed Properties
    # ========================================================================
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == "production"
    
    @property
    def cuda_enabled(self) -> bool:
        """Check if CUDA should be used (if available)."""
        import torch
        return self.ENABLE_CUDA and torch.cuda.is_available()
    
    @property
    def effective_device(self) -> str:
        """Get actual device to use based on configuration and availability."""
        if self.DEVICE == "auto":
            import torch
            return "cuda" if self.cuda_enabled else "cpu"
        return self.DEVICE
    
    # ========================================================================
    # Validation
    # ========================================================================
    
    def validate_configuration(self) -> None:
        """
        Validate configuration settings and log warnings.
        """
        import warnings
        
        # Security checks
        if self.is_production:
            if self.SECRET_KEY == "change-this-to-a-secure-random-key-in-production":
                warnings.warn("SECRET_KEY not changed from default - SECURITY RISK")
            
            if self.DEBUG:
                warnings.warn("DEBUG mode enabled in production - SECURITY RISK")
        
        # Model path checks
        if not self.MODEL_WEIGHTS_PATH.exists():
            warnings.warn(f"Model weights not found at {self.MODEL_WEIGHTS_PATH}")
        
        # CUDA checks
        if self.ENABLE_CUDA:
            import torch
            if not torch.cuda.is_available():
                warnings.warn("CUDA enabled but not available - falling back to CPU")
    
    def print_config(self) -> None:
        """Print non-sensitive configuration for debugging."""
        config_info = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    ESRGAN API Configuration                               ║
╚═══════════════════════════════════════════════════════════════════════════╝

  Application:
    ├─ Name:        {self.APP_NAME}
    ├─ Version:     {self.VERSION}
    ├─ Environment: {self.ENVIRONMENT}
    └─ Debug:       {self.DEBUG}

  Server:
    ├─ Host:        {self.HOST}
    ├─ Port:        {self.PORT}
    ├─ Workers:     {self.WORKERS}
    └─ Reload:      {self.RELOAD}

  ML Model:
    ├─ Device:      {self.effective_device}
    ├─ Precision:   {self.MODEL_PRECISION}
    ├─ Scale:       {self.SCALE_FACTOR}x
    └─ Weights:     {self.MODEL_WEIGHTS_PATH}

  Files:
    ├─ Max Size:    {self.MAX_FILE_SIZE / 1024 / 1024:.1f} MB
    ├─ Extensions:  {', '.join(self.ALLOWED_EXTENSIONS)}
    └─ Temp Dir:    {self.UPLOAD_TEMP_DIR}

  Redis:
    └─ URL:         {self.REDIS_URL.split('@')[-1]}

  Database:
    └─ URL:         {self.DATABASE_URL.split('@')[-1]}

  Security:
    ├─ CORS Origins: {len(self.ALLOWED_ORIGINS)} configured
    └─ Rate Limit:   {self.RATE_LIMIT_PER_MINUTE}/min, {self.RATE_LIMIT_PER_HOUR}/hour

╚═══════════════════════════════════════════════════════════════════════════╝
"""
        print(config_info)


# ============================================================================
# Global Settings Instance
# ============================================================================

settings = Settings()

# Validate on import
settings.validate_configuration()


# ============================================================================
# Testing & Debugging
# ============================================================================

if __name__ == "__main__":
    """Print configuration for debugging."""
    settings.print_config()
    
    # Test device detection
    import torch
    print("\n🔍 Device Detection:")
    print(f"   ├─ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ├─ CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"   └─ CUDA Version: {torch.version.cuda}")
    else:
        print(f"   └─ Using CPU")
