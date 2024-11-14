from pathlib import Path

PROJECT_ROOT = Path(r"E:\face_recognition")
DATA_DIR = PROJECT_ROOT / "data"
KNOWN_FACES_DIR = DATA_DIR / "known_faces"
UNKNOWN_FACES_DIR = DATA_DIR / "unknown_faces"

# Other configurations
DETECTION_CONFIDENCE = 0.5
FACE_IMAGE_SIZE = (160, 160)

# Create directories if they don't exist
KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
UNKNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)

# Add at the bottom of config.py
if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Known Faces Directory: {KNOWN_FACES_DIR}")
    print(f"Unknown Faces Directory: {UNKNOWN_FACES_DIR}")