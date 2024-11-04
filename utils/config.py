# src/utils/config.py
import json
import os
import logging
from typing import Dict, Any, Callable, Optional
from pathlib import Path

logger = logging.getLogger('VideoRecognition.utils')

DEFAULT_CONFIG = {
    'confidence_threshold': 0.5,
    'use_gpu': True,
    'model': 'yolov8n',
    'save_path': 'detections',
    'max_image_size': 1280,
    'batch_size': 4,
    'enable_optimization': True,
    'cache_results': True,
    # Face detection settings
    'enable_face_detection': True,
    'face_confidence_threshold': 0.5,
    'emotion_detection_interval': 5,  # Process every N frames
    'emotion_min_face_size': 64,  # Minimum face size in pixels
}

def load_config() -> Dict[str, Any]:
    config_path = Path('config.json')
    try:
        if config_path.exists():
            with config_path.open('r') as f:
                config = json.load(f)
                logger.info("Configuration loaded successfully")
                # Update with any missing default values
                return {**DEFAULT_CONFIG, **config}
        else:
            logger.info("No config file found, creating with defaults")
            save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}", exc_info=True)
        return DEFAULT_CONFIG

def save_config(config: Dict[str, Any]) -> None:
    try:
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=4)
        logger.info("Configuration saved successfully")
    except Exception as e:
        logger.error(f"Error saving config: {str(e)}", exc_info=True)