import cv2
import numpy as np
from mtcnn import MTCNN
from typing import List, Dict, Optional

class FaceDetector:
    def __init__(self, min_confidence: float = 0.95):
        self.detector = MTCNN()
        self.min_confidence = min_confidence

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image using MTCNN
        """
        # Convert BGR to RGB (MTCNN expects RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = self.detector.detect_faces(rgb_image)
        
        # Filter by confidence
        return [
            det for det in detections 
            if det['confidence'] >= self.min_confidence
        ]

    def extract_face(self, frame, face_coords):
        # Check if face_coords is a dictionary (from live detection)
        if isinstance(face_coords, dict):
            x, y, w, h = face_coords['box']
        # Check if face_coords is a list/tuple (direct coordinates)
        elif len(face_coords) == 4:
            x, y, w, h = face_coords
        else:
            raise ValueError(f"Unexpected face coordinate format: {face_coords}")
            
        # Extract and process the face
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))  # FaceNet expects 160x160 images
        return face