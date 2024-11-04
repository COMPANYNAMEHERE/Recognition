# src/utils/__init__.py
"""
Video Recognition Utilities Package

This package contains utility modules for video processing, model management,
and application configuration.
"""

from .config import load_config, save_config, DEFAULT_CONFIG
from .video_converter import VideoConverter, convert_video_format
from .model_manager import ModelManager

__all__ = [
    'load_config',
    'save_config',
    'DEFAULT_CONFIG',
    'VideoConverter',
    'convert_video_format',
    'ModelManager'
]

# Package version
__version__ = '1.0.0'

# Configure package-wide logging
import logging
logger = logging.getLogger('VideoRecognition.utils')
logger.addHandler(logging.NullHandler())  # Prevents "No handler found" warning

# Feature flags
ENABLE_GPU = True
ENABLE_THREADING = True
ENABLE_OPTIMIZATION = True

# Default paths
from pathlib import Path
DEFAULT_MODEL_DIR = Path('models')
DEFAULT_CACHE_DIR = Path('cache')
DEFAULT_OUTPUT_DIR = Path('output')
DEFAULT_LOG_DIR = Path('logs')

# Ensure critical directories exist
for directory in [DEFAULT_MODEL_DIR, DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_LOG_DIR]:
    directory.mkdir(exist_ok=True)