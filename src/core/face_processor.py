import cv2
import numpy as np
from typing import List, Dict
from .detector import FaceDetector
from .enhancer import ImageEnhancer
from .recognizer import FaceRecognizer
from ..config import DETECTION_CONFIDENCE, FACE_IMAGE_SIZE
from datetime import datetime

class FaceProcessor:
    def __init__(self, detection_confidence=0.5):
        self.detector = FaceDetector(detection_confidence)
        self.recognizer = FaceRecognizer()
        self.last_detection = {}  # To track when we last saw each person
        self.detection_cooldown = 3  # Only print every 3 seconds
        
    def process_frame(self, frame):
        faces = self.detector.detect_faces(frame)
        results = []
        
        for face_coords in faces:
            face_img = self.detector.extract_face(frame, face_coords)
            name, distance = self.recognizer.recognize_face(face_img, threshold=0.9)
            
            current_time = datetime.now()
            
            # Only print if we haven't seen this person recently
            if name != "Unknown":
                if (name not in self.last_detection or 
                    (current_time - self.last_detection[name]).seconds >= self.detection_cooldown):
                    print(f"[{current_time.strftime('%H:%M:%S')}] Detected: {name}")
                    self.last_detection[name] = current_time
            
            results.append((face_coords['box'], name, distance))
            
        return results