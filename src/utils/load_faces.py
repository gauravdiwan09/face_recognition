import cv2
from pathlib import Path
from ..core.face_processor import FaceProcessor
from ..config import KNOWN_FACES_DIR

def load_known_faces(processor):
    print("\nLoading known faces...")
    loaded_faces = 0
    
    # Process all jpg/jpeg/png files in the known_faces directory
    for image_path in KNOWN_FACES_DIR.glob("*.[jJ][pP][gG]"):  # This will match .jpg, .JPG
        # Get person name from filename (remove extension)
        person_name = image_path.stem.lower().replace(" ", "_")  # Convert to lowercase and replace spaces
        print(f"Loading: {person_name}")
        
        # Load and process image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        detections = processor.detector.detect_faces(image)
        if not detections:
            print(f"No face detected in {image_path}")
            continue
        
        # Get the face with highest confidence
        face_data = detections[0]
        face = processor.detector.extract_face(image, face_data)
        
        # Preprocess face
        face = cv2.resize(face, (160, 160))  # FaceNet requires 160x160 images
        face = face.astype('float32')  # Convert to float32
        
        # Add to known faces
        processor.recognizer.add_known_face(person_name, face)
        loaded_faces += 1
    
    print(f"Successfully loaded {loaded_faces} faces")
    if loaded_faces > 0:
        print("Known persons:", ", ".join(processor.recognizer.known_face_names))
    else:
        print("No faces were loaded! Please add images to the known_faces directory.")