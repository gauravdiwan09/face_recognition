import cv2
from src.core.face_processor import FaceProcessor
from src.config import DETECTION_CONFIDENCE
from src.utils.load_faces import load_known_faces

def main():
    processor = FaceProcessor(detection_confidence=DETECTION_CONFIDENCE)
    
    # Load known faces
    load_known_faces(processor)
    
    print("\nStarting face recognition...")
    print("Press 'q' to quit")
    print("Press 'r' to reload known faces")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        results = processor.process_frame(frame)
        
        # Draw results
        for (x, y, w, h), name, confidence in results:
            if name != "Unknown":
                # Known face - green box
                color = (0, 255, 0)
                conf_text = f"{name} ({confidence:.2%})"
            else:
                # Unknown face - red box
                color = (0, 0, 255)
                conf_text = "Unknown"
                
            # Draw box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, conf_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show frame
        cv2.imshow('Face Recognition', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("\nReloading known faces...")
            load_known_faces(processor)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()