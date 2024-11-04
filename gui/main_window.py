# src/gui/main_window.py
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QFileDialog, QProgressBar,
                           QMessageBox, QStatusBar, QFrame)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from utils.video_converter import VideoConverter
from utils.model_manager import ModelManager
from utils.video_processor import VideoProcessor
from gui.settings_dialog import SettingsDialog
from gui.base_classes import WorkerThread, AnalysisWorker, DetectionSummary, EmotionSummary
import logging
from pathlib import Path
import psutil
import cv2
import time
import sys
import os
import subprocess

logger = logging.getLogger('VideoRecognition.gui')

class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_manager = ModelManager()
        self.video_converter = VideoConverter()
        self.current_worker = None
        self.current_video_path = None
        self.analysis_worker = None
        self.video_processor = None
        self.current_results = []
        
        # Basic window setup
        self.setWindowTitle("Video Recognition")
        self.resize(1200, 800)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #2b2b2b; color: white; }
            QPushButton { 
                background-color: #2196F3; 
                color: white; 
                border: none; 
                padding: 8px 16px; 
                border-radius: 4px; 
                min-width: 80px; 
            }
            QPushButton:disabled { background-color: #424242; }
            QProgressBar { 
                border: 2px solid #424242; 
                border-radius: 5px; 
                text-align: center; 
            }
            QProgressBar::chunk { background-color: #2196F3; }
            QFrame { 
                border: 2px solid #424242; 
                border-radius: 4px; 
                background-color: #1b1b1b; 
            }
        """)
        
        self.setup_ui()
        self.setup_monitoring()
        self.load_model()

    # [Keep all the remaining methods from the previous MainWindow class]
    # [From setup_ui() to closeEvent()]

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Video frame
        self.video_frame = QLabel()
        self.video_frame.setMinimumSize(640, 480)
        self.video_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_frame.setFrameStyle(QFrame.Shape.Box)
        layout.addWidget(self.video_frame)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Video")
        self.analyze_button = QPushButton("Analyze")
        self.save_button = QPushButton("Save Results")
        self.settings_button = QPushButton("Settings")
        
        for btn in [self.load_button, self.analyze_button, self.save_button, self.settings_button]:
            btn_layout.addWidget(btn)
            
        self.analyze_button.setEnabled(False)
        self.save_button.setEnabled(False)
        
        self.load_button.clicked.connect(self.load_video)
        self.analyze_button.clicked.connect(self.analyze_video)
        self.save_button.clicked.connect(self.save_results)
        self.settings_button.clicked.connect(self.show_settings)
        
        layout.addLayout(btn_layout)
        
        # Progress section
        self.progress_status = QLabel()
        layout.addWidget(self.progress_status)
        
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.cancel_button = QPushButton("Cancel")
        
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self.cancel_operation)
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.cancel_button)
        layout.addLayout(progress_layout)
        
        # Status bar
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self.cpu_label = QLabel("CPU: ---%")
        self.memory_label = QLabel("Memory: ---MB")
        self.gpu_label = QLabel("GPU: ---")
        
        status_bar.addPermanentWidget(self.cpu_label)
        status_bar.addPermanentWidget(self.memory_label)
        status_bar.addPermanentWidget(self.gpu_label)

    def setup_monitoring(self):
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(2000)

    def update_status(self):
        try:
            self.cpu_label.setText(f"CPU: {psutil.cpu_percent()}%")
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.memory_label.setText(f"Memory: {memory_mb:.1f}MB")
            
            if hasattr(self.model_manager, 'device'):
                self.gpu_label.setText(
                    "GPU: Active" if self.model_manager.device.type == 'cuda' 
                    else "GPU: N/A"
                )
        except Exception as e:
            logger.error(f"Status update error: {e}")

    def load_model(self):
        self.start_task(self.model_manager.download_model, on_complete=self.model_loaded)

    def model_loaded(self):
        self.load_button.setEnabled(True)
        self.status_label.setText("Model loaded successfully")

    def load_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if file_path:
            self.current_video_path = file_path
            self.status_label.setText(f"Loading: {Path(file_path).name}")
            self.start_task(
                self.video_converter.convert_video_format,
                file_path,
                self.config['max_image_size'],
                on_complete=self.video_loaded
            )

    def video_loaded(self):
        self.analyze_button.setEnabled(True)
        self.status_label.setText("Video loaded successfully")
        
        # Display first frame
        cap = cv2.VideoCapture(self.current_video_path)
        ret, frame = cap.read()
        if ret:
            self.update_video_frame((frame, None))
        cap.release()

    def analyze_video(self):
        if not self.current_video_path:
            return
            
        try:
            # Initialize video processor with current model
            self.video_processor = VideoProcessor(self.model_manager.model)
            
            # Create and configure worker thread
            self.analysis_worker = AnalysisWorker(
                self.video_processor,
                self.current_video_path
            )
            
            # Connect signals
            self.analysis_worker.progress.connect(self.progress_bar.setValue)
            self.analysis_worker.frame_processed.connect(self.update_video_frame)
            self.analysis_worker.error.connect(self.handle_error)
            self.analysis_worker.finished.connect(self.analysis_completed)
            
            # Update UI
            self.status_label.setText("Analyzing video...")
            self.progress_bar.setVisible(True)
            self.cancel_button.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Disable buttons during analysis
            self.analyze_button.setEnabled(False)
            self.load_button.setEnabled(False)
            self.save_button.setEnabled(False)
            self.settings_button.setEnabled(False)
            
            # Connect video processor signals
            self.video_processor.progress.connect(self.progress_bar.setValue)
            self.video_processor.frame_processed.connect(self.update_video_frame)
            
            # Start analysis
            self.analysis_worker.start()
            
            logger.info(f"Started video analysis: {self.current_video_path}")
            
        except Exception as e:
            logger.error(f"Error starting analysis: {str(e)}", exc_info=True)
            self.handle_error(f"Failed to start analysis: {str(e)}")

    def update_video_frame(self, frame_data):
        try:
            frame, results = frame_data
            if frame is not None:
                # Convert frame for display
                height, width = frame.shape[:2]
                bytes_per_line = 3 * width
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create QImage
                image = QImage(frame_rgb.data, width, height, 
                             bytes_per_line, QImage.Format.Format_RGB888)
                
                # Update video frame
                self.video_frame.setPixmap(QPixmap.fromImage(image).scaled(
                    self.video_frame.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))
                
                # Store results for saving
                if results is not None:
                    self.current_results.append(results)
                
        except Exception as e:
            logger.error(f"Error updating video frame: {str(e)}", exc_info=True)

    def analysis_completed(self):
        try:
            self.status_label.setText("Analysis completed")
            self.progress_bar.setVisible(False)
            self.cancel_button.setVisible(False)
            
            # Re-enable buttons
            self.load_button.setEnabled(True)
            self.analyze_button.setEnabled(True)
            self.save_button.setEnabled(True)
            self.settings_button.setEnabled(True)
            
            self.analysis_worker = None
            logger.info("Video analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in analysis completion: {str(e)}", exc_info=True)

    def save_results(self):
        if not hasattr(self, 'current_results') or not self.current_results:
            QMessageBox.warning(self, "No Results", "No analysis results available to save.")
            return
            
        try:
            # Create save directory if it doesn't exist
            save_dir = Path(self.config.get('save_path', 'detections'))
            save_dir.mkdir(exist_ok=True)
            
            # Generate output filename
            input_path = Path(self.current_video_path)
            output_path = save_dir / f"{input_path.stem}_detections.txt"
            
            # Save results
            with open(output_path, 'w') as f:
                for result in self.current_results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        conf = box.conf[0]
                        cls = box.cls[0]
                        f.write(f"Frame: {result.frame_idx}, "
                               f"Class: {result.names[int(cls)]}, "
                               f"Confidence: {conf:.2f}, "
                               f"Box: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})\n")
            
            self.status_label.setText(f"Results saved to {output_path}")
            logger.info(f"Analysis results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}", exc_info=True)
            self.handle_error(f"Failed to save results: {str(e)}")
            
    def show_settings(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            # Reload model if model settings changed
            if (self.config['model'] != dialog.model_combo.currentText() or
                self.config['use_gpu'] != dialog.gpu_check.isChecked()):
                self.load_model()
                
    def start_task(self, task_func, *args, on_complete=None, **kwargs):
        if self.current_worker and self.current_worker.isRunning():
            return
            
        self.current_worker = WorkerThread(task_func, *args, **kwargs)
        self.current_worker.progress.connect(self.progress_bar.setValue)
        self.current_worker.status.connect(self.status_label.setText)
        self.current_worker.error.connect(self.handle_error)
        
        if on_complete:
            self.current_worker.finished.connect(on_complete)
        self.current_worker.finished.connect(self.task_completed)
        
        self.progress_bar.setVisible(True)
        self.cancel_button.setVisible(True)
        self.progress_bar.setValue(0)
        
        for btn in [self.load_button, self.analyze_button, self.save_button, self.settings_button]:
            btn.setEnabled(False)
            
        self.current_worker.start()
        
    def cancel_operation(self):
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.cancel()
            self.status_label.setText("Cancelling...")
            self.cancel_button.setEnabled(False)
            
    def handle_error(self, error_message):
        logger.error(f"Task error: {error_message}")
        self.reset_ui()
        QMessageBox.critical(self, "Error", f"Operation failed: {error_message}")
        
    def task_completed(self):
        logger.info("Task completed successfully")
        self.reset_ui()
        
    def reset_ui(self):
        self.progress_bar.setVisible(False)
        self.cancel_button.setVisible(False)
        self.load_button.setEnabled(True)
        self.current_worker = None
        
    def closeEvent(self, event):
        logger.info("Application closing")
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.cancel()
        event.accept()