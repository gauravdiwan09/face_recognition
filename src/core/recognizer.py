from keras_facenet import FaceNet
import numpy as np

class FaceRecognizer:
    def __init__(self):
        self.model = FaceNet()
        self.known_face_embeddings = {}
        self.known_face_names = []
        self.distance_threshold = 0.8  # Adjusted threshold
        
    def get_embeddings(self, face_image):
        # Ensure image is float32 and correct shape
        face_image = face_image.astype('float32')
        if len(face_image.shape) == 3:
            face_image = np.expand_dims(face_image, axis=0)
        return self.model.embeddings(face_image)
    
    def add_known_face(self, person_name, face_image):
        # Get embedding for the face
        embedding = self.get_embeddings(face_image)[0]
        
        # Store the embedding and name
        if person_name not in self.known_face_embeddings:
            self.known_face_embeddings[person_name] = []
            self.known_face_names.append(person_name)
        
        self.known_face_embeddings[person_name].append(embedding)
        
    def recognize_face(self, face_image, threshold=None):
        if not self.known_face_embeddings:
            return "Unknown", float('inf')
        
        if threshold is None:
            threshold = self.distance_threshold
            
        # Get embedding for the test face
        test_embedding = self.get_embeddings(face_image)[0]
        
        # Find the closest match
        min_distance = float('inf')
        best_match = None
        
        for name, embeddings in self.known_face_embeddings.items():
            for embedding in embeddings:
                distance = np.linalg.norm(embedding - test_embedding)
                if distance < min_distance:
                    min_distance = distance
                    best_match = name
        
        if min_distance < threshold:
            return best_match, min_distance
        return "Unknown", min_distance