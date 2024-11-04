# src/utils/video_converter.py
import cv2
import os
import logging
from pathlib import Path
from typing import Optional, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger('VideoRecognition.utils')

def convert_video_format(input_path: str, 
                        max_size: int = 1280,
                        progress_callback: Optional[Callable[[int], None]] = None,
                        status_callback: Optional[Callable[[str], None]] = None,
                        cancel_check: Optional[Callable[[], bool]] = None) -> Optional[str]:
    """
    Convert video to a compatible format with size optimization and progress tracking
    
    Args:
        input_path: Path to input video file
        max_size: Maximum dimension for video
        progress_callback: Function to call with progress (0-100)
        status_callback: Function to call with status messages
        cancel_check: Function to check if operation should be cancelled
    """
    converter = VideoConverter()
    return converter.convert_video_format(
        input_path=input_path,
        max_size=max_size,
        progress_callback=progress_callback,
        status_callback=status_callback,
        cancel_check=cancel_check
    )

class VideoConverter:
    def __init__(self):
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv'}
        self.output_format = 'mp4'
        self.output_codec = 'mp4v'
        self._cancel_flag = threading.Event()
        
    def convert_video_format(self, 
                           input_path: str, 
                           max_size: int = 1280,
                           progress_callback: Optional[Callable[[int], None]] = None,
                           status_callback: Optional[Callable[[str], None]] = None,
                           cancel_check: Optional[Callable[[], bool]] = None) -> Optional[str]:
        """
        Convert video to a compatible format with size optimization
        """
        try:
            input_path = Path(input_path)
            if not input_path.exists():
                raise FileNotFoundError(f"Input video not found: {input_path}")
            
            if status_callback:
                status_callback("Initializing video conversion...")
            
            output_dir = input_path.parent / 'processed'
            output_dir.mkdir(exist_ok=True)
            
            output_path = output_dir / f"{input_path.stem}_converted.{self.output_format}"
            
            # Check if already converted
            if output_path.exists():
                if status_callback:
                    status_callback("Using existing converted video")
                if progress_callback:
                    progress_callback(100)
                logger.info(f"Using existing converted video: {output_path}")
                return str(output_path)
            
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise RuntimeError("Failed to open input video")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if status_callback:
                status_callback(f"Processing video: {width}x{height} @ {fps}fps")
            
            # Calculate new dimensions if needed
            if width > max_size or height > max_size:
                scale = max_size / max(width, height)
                width = int(width * scale)
                height = int(height * scale)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*self.output_codec)
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            def process_frame(frame):
                if cancel_check and cancel_check():
                    return None
                if width != frame.shape[1] or height != frame.shape[0]:
                    frame = cv2.resize(frame, (width, height))
                return frame
            
            frames_processed = 0
            
            # Process frames with thread pool
            with ThreadPoolExecutor() as executor:
                while cap.isOpened():
                    if cancel_check and cancel_check():
                        logger.info("Video conversion cancelled")
                        break
                        
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    processed_frame = executor.submit(process_frame, frame).result()
                    if processed_frame is None:  # Cancelled
                        break
                        
                    out.write(processed_frame)
                    
                    frames_processed += 1
                    if progress_callback:
                        progress = min(99, int((frames_processed / total_frames) * 100))
                        progress_callback(progress)
                        
                    if status_callback and frames_processed % 30 == 0:
                        status_callback(f"Processing frame {frames_processed}/{total_frames}")
            
            cap.release()
            out.release()
            
            if cancel_check and cancel_check():
                if output_path.exists():
                    output_path.unlink()
                return None
            
            if progress_callback:
                progress_callback(100)
            if status_callback:
                status_callback("Video conversion completed")
                
            logger.info(f"Video converted successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error converting video: {str(e)}", exc_info=True)
            if 'out' in locals():
                out.release()
            if 'cap' in locals():
                cap.release()
            if 'output_path' in locals() and output_path.exists():
                output_path.unlink()
            raise