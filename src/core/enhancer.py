import cv2
import numpy as np
from typing import Optional

class ImageEnhancer:
    def __init__(self):
        pass
        
    def enhance_face(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Enhance face image quality using OpenCV
        """
        try:
            # Basic image enhancement
            # 1. Denoise
            denoised = cv2.fastNlMeansDenoisingColored(face_image)
            
            # 2. Enhance contrast
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except Exception:
            return face_image  # Return original image if enhancement fails