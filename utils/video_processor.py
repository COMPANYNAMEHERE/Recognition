# src/utils/video_processor.py
from PyQt6.QtCore import QObject, pyqtSignal
import cv2
import torch
import numpy as np
from deepface import DeepFace
import mediapipe as mp
import logging
from typing import Tuple, List, Dict
import time

logger = logging.getLogger('VideoRecognition.utils')

class VideoProcessor(QObject):
    progress = pyqtSignal(int)
    frame_processed = pyqtSignal(tuple)
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # Initialize face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for close range, 1 for far range
            min_detection_confidence=0.5
        )
        
        # Cache for emotion detection to improve performance
        self.emotion_cache = {}
        self.cache_ttl = 5  # frames
        
    def detect_emotions(self, frame: np.ndarray, face_location: Dict) -> Tuple[str, float]:
        """
        Detect emotions in a face region
        Returns emotion label and confidence
        """
        try:
            # Extract face region
            x, y, w, h = face_location['box']
            face_region = frame[y:y+h, x:x+w]
            
            # Skip if face region is too small
            if face_region.size == 0 or w < 64 or h < 64:
                return None, 0.0
            
            # Analyze emotions
            result = DeepFace.analyze(
                face_region,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list):
                result = result[0]
                
            emotion = result['dominant_emotion']
            confidence = result['emotion'][emotion] / 100.0
            
            return emotion, confidence
            
        except Exception as e:
            logger.debug(f"Emotion detection error: {str(e)}")
            return None, 0.0
            
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Process a single frame with both object and face detection
        """
        # Object detection
        results = self.model(frame)
        
        # Face detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = self.face_detection.process(frame_rgb)
        
        # Draw object detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]
                
                cv2.rectangle(frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                cv2.putText(frame, 
                           f"{result.names[int(cls)]}: {conf:.2f}",
                           (int(x1), int(y1) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
        
        # Process and draw face detections
        if face_results.detections:
            for detection in face_results.detections:
                # Get face location
                box = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(box.xmin * iw)
                y = int(box.ymin * ih)
                w = int(box.width * iw)
                h = int(box.height * ih)
                
                # Ensure coordinates are within frame
                x = max(0, x)
                y = max(0, y)
                w = min(w, iw - x)
                h = min(h, ih - y)
                
                face_location = {'box': (x, y, w, h)}
                
                # Get cached emotion or detect new one
                cache_key = f"{x}_{y}_{w}_{h}"
                if cache_key in self.emotion_cache:
                    emotion, confidence, ttl = self.emotion_cache[cache_key]
                    if ttl > 0:
                        self.emotion_cache[cache_key] = (emotion, confidence, ttl - 1)
                    else:
                        del self.emotion_cache[cache_key]
                        emotion, confidence = self.detect_emotions(frame, face_location)
                        if emotion:
                            self.emotion_cache[cache_key] = (emotion, confidence, self.cache_ttl)
                else:
                    emotion, confidence = self.detect_emotions(frame, face_location)
                    if emotion:
                        self.emotion_cache[cache_key] = (emotion, confidence, self.cache_ttl)
                
                # Draw face detection
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Draw emotion if detected
                if emotion:
                    label = f"{emotion}: {confidence:.2f}"
                    cv2.putText(frame, label,
                              (x, y - 25),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.7, (255, 0, 0), 2)
        
        return frame, results
        
    def start(self, video_path: str) -> None:
        """
        Process video with object and face detection
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            processed_frames = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Emit progress and processed frame
                processed_frames += 1
                progress = int((processed_frames / total_frames) * 100)
                self.progress.emit(progress)
                self.frame_processed.emit((processed_frame, results))
                
            cap.release()
            
        except Exception as e:
            logger.error(f"Video processing error: {str(e)}", exc_info=True)
            raise
        finally:
            if 'cap' in locals():
                cap.release()