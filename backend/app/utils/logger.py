"""
Logger Configuration Utility
============================

Authored by: Backend Engineering Team
Date: April 2026
Version: 1.0

Description:
    Provides structured and standardized logging setup across the application.
    Supports both JSON formatting (for production log aggregation) and
    traditional colorized text formatting (for development readability).
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from pythonjsonlogger import jsonlogger

from app.core.config import settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter to add extra context fields like timestamp,
    log level, and module name in standard formats.
    """
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = record.created
        log_record['level'] = record.levelname
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno


def setup_logger() -> None:
    """
    Initialize and configure the global logging system based on application settings.
    """
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    root_logger = logging.getLogger()
    
    # Reset existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    root_logger.setLevel(log_level)
    
    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    if settings.LOG_FORMAT.lower() == "json":
        formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(module)s %(function)s %(line)d %(message)s'
        )
    else:
        # Development readable colorized/clean text format
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Configure file handler if specified
    if settings.LOG_FILE:
        try:
            # Ensure path directory exists (handled by validator, but double check)
            settings.LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(settings.LOG_FILE, encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(
                CustomJsonFormatter(
                    '%(timestamp)s %(level)s %(module)s %(function)s %(line)d %(message)s'
                ) if settings.LOG_FORMAT.lower() == "json" else logging.Formatter(
                    fmt="%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s"
                )
            )
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.error(f"Failed to initialize file logger at {settings.LOG_FILE}: {str(e)}")
            
    # Quiet verbose libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    get_logger(__name__).info(f"Logging initialized at level {settings.LOG_LEVEL} ({settings.LOG_FORMAT} format)")


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a named logger instance.
    
    Args:
        name: Name of the module requesting the logger
        
    Returns:
        logging.Logger: Named logger instance
    """
    return logging.getLogger(name)
