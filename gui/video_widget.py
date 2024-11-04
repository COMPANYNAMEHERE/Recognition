# src/gui/video_widget.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
import cv2
from utils.video_converter import convert_video_format

class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.video_label)
        
    def load_video(self, path):
        self.video_path = path
        # Convert video if needed
        self.video_path = convert_video_format(path)
        # Load first frame
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        if ret:
            self.update_frame(frame)
        cap.release()
        
    def update_frame(self, frame):
        if isinstance(frame, tuple):
            frame, _ = frame  # Handle case where frame comes with detections
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        bytes_per_line = 3 * width
        image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))