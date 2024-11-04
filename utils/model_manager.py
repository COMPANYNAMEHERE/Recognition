# src/utils/model_manager.py
import torch
import logging
from pathlib import Path
from ultralytics import YOLO
from PyQt6.QtCore import QObject, pyqtSignal
import shutil
import time
from typing import Optional, Callable

logger = logging.getLogger('VideoRecognition.utils')

class ModelManager(QObject):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_name = 'yolov8n'
        self.model_dir = Path('models')
        self.model_path = self.model_dir / f'{self.model_name}.pt'
        self.device = self._get_optimal_device()
        self._cancel_flag = False
        
    def _get_optimal_device(self) -> torch.device:
        """Determine the best available device for inference"""
        if torch.cuda.is_available():
            # Get GPU with most free memory
            gpu_memory = []
            for i in range(torch.cuda.device_count()):
                total_mem = torch.cuda.get_device_properties(i).total_memory
                used_mem = torch.cuda.memory_allocated(i)
                gpu_memory.append((i, total_mem - used_mem))
            
            if gpu_memory:
                best_gpu = max(gpu_memory, key=lambda x: x[1])[0]
                logger.info(f"Using GPU {best_gpu}")
                return torch.device(f'cuda:{best_gpu}')
        
        logger.info("Using CPU")
        return torch.device('cpu')
        
    def download_model(self, 
                      progress_callback: Optional[Callable[[int], None]] = None,
                      status_callback: Optional[Callable[[str], None]] = None,
                      cancel_check: Optional[Callable[[], bool]] = None) -> None:
        """
        Download and initialize the YOLO model with progress tracking
        """
        try:
            start_time = time.time()
            if progress_callback:
                progress_callback(0)
            if status_callback:
                status_callback("Initializing model setup...")
            
            # Create model directory
            self.model_dir.mkdir(exist_ok=True)
            
            # Clean up any existing corrupted models
            self._cleanup_existing_models()
            
            if cancel_check and cancel_check():
                return
            
            # Download and initialize model with specific configuration
            if status_callback:
                status_callback("Downloading YOLO model...")
                
            # Initialize model with float32 precision
            self.model = YOLO(f"{self.model_name}.pt")
            
            # Move model to device first
            self.model.to(self.device)
            
            if progress_callback:
                progress_callback(50)
            
            if cancel_check and cancel_check():
                self._cleanup_existing_models()
                return
            
            # Save model locally
            if status_callback:
                status_callback("Saving model...")
            self.model.save(str(self.model_path))
            
            # Optional: Configure model for inference
            if self.device.type == 'cuda':
                if status_callback:
                    status_callback("Optimizing model for GPU...")
                self.model.model.float()  # Ensure model is in float32
                
            elapsed_time = time.time() - start_time
            logger.info(f"Model setup completed in {elapsed_time:.2f} seconds")
            
            if progress_callback:
                progress_callback(100)
            if status_callback:
                status_callback("Model ready")
            
        except Exception as e:
            logger.error(f"Error in model setup: {str(e)}", exc_info=True)
            self._cleanup_existing_models()
            raise
            
    def _cleanup_existing_models(self) -> None:
        """Clean up any existing model files"""
        try:
            if self.model_path.exists():
                self.model_path.unlink()
                logger.info(f"Removed existing model: {self.model_path}")
        except Exception as e:
            logger.error(f"Error cleaning up models: {str(e)}", exc_info=True)