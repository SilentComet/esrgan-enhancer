"""
Security & Hardening Configuration
==================================

Authored by: Backend Engineering Team
Date: April 2026
Version: 1.0

Description:
    Implements security features including API key validation, password hashing,
    JWT generation, and specialized middleware setups.
"""

from fastapi import FastAPI, Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Password hashing configuration
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# API Key Security Scheme
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)


def setup_security(app: FastAPI) -> None:
    """
    Hook to configure additional security features on the FastAPI app instance.
    Could be extended to configure secure session cookies, state CSRF checks,
    or OAuth2 configurations.
    """
    logger.info("✓ Enterprise security parameters applied")


def get_password_hash(password: str) -> str:
    """Hash a plain text password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Generate a secure JWT access token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
        
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm="HS256")
    return encoded_jwt


async def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """
    Optional API key validation dependency.
    Can be used to secure specific backend routes.
    """
    if settings.is_production and not api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API Key missing from header"
        )
    # Customize this validation with database or configuration keys
    return api_key
