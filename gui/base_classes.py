# src/gui/base_classes.py
from PyQt6.QtCore import QThread, pyqtSignal
import logging

logger = logging.getLogger('VideoRecognition.gui')

class WorkerThread(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()
    
    def __init__(self, task_func, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs
        self._is_cancelled = False
        
    def run(self):
        try:
            self.task_func(
                *self.args,
                progress_callback=self.progress.emit,
                status_callback=self.status.emit,
                cancel_check=lambda: self._is_cancelled,
                **self.kwargs
            )
            if not self._is_cancelled:
                self.finished.emit()
        except Exception as e:
            logger.error(f"Worker thread error: {str(e)}", exc_info=True)
            self.error.emit(str(e))
            
    def cancel(self):
        self._is_cancelled = True

class AnalysisWorker(QThread):
    progress = pyqtSignal(int)
    frame_processed = pyqtSignal(tuple)
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, video_processor, video_path):
        super().__init__()
        self.video_processor = video_processor
        self.video_path = video_path
        self._is_cancelled = False
        
    def run(self):
        try:
            self.video_processor.start(self.video_path)
            if not self._is_cancelled:
                self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
            
    def cancel(self):
        self._is_cancelled = True

class DetectionSummary:
    def __init__(self):
        self.total_frames = 0
        self.first_seen = None
        self.last_seen = None
        self.confidence_sum = 0.0
        self.count = 0
        
    @property
    def average_confidence(self):
        return self.confidence_sum / self.count if self.count > 0 else 0.0
        
    @property
    def duration_frames(self):
        if self.first_seen is None or self.last_seen is None:
            return 0
        return self.last_seen - self.first_seen + 1

class EmotionSummary:
    def __init__(self):
        self.emotions = {
            'happy': 0, 'sad': 0, 'angry': 0,
            'surprised': 0, 'neutral': 0,
            'fear': 0, 'disgust': 0
        }
        self.total_detections = 0
        
    def add_emotion(self, emotion):
        if emotion.lower() in self.emotions:
            self.emotions[emotion.lower()] += 1
            self.total_detections += 1
            
    @property
    def dominant_emotion(self):
        if self.total_detections == 0:
            return "No emotions detected"
        return max(self.emotions.items(), key=lambda x: x[1])[0]
        
    def get_emotion_percentage(self, emotion):
        if self.total_detections == 0:
            return 0.0
        return (self.emotions[emotion] / self.total_detections) * 100