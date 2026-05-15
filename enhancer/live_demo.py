import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
import time

# --- USER SETTINGS ---
# Set to 0 for Webcam
# Set to "filename.mp4" for a video file (must be in enhancer folder)
VIDEO_SOURCE = "test2.mp4"  
# ---------------------

# Setup Paths
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, "EnlightenGAN-inference"))

try:
    from enlighten_inference import EnlightenOnnxModel
except ImportError:
    print("❌ ERROR: Run this from the 'enhancer' folder.")
    sys.exit()

def run_live():
    # 1. Load Models
    print("⏳ Loading Night Vision (EnlightenGAN)...")
    enhancer = EnlightenOnnxModel()
    
    print("⏳ Loading YOLOv8...")
    model_path = os.path.abspath(os.path.join(current_dir, "..", "runs", "detect", "heavy_night_model2", "weights", "best.pt"))
    detector = YOLO(model_path)

    # 2. Open Video Source
    print(f"🎥 Opening video source: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("❌ Error: Could not open video source.")
        return

    # Get video size
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # We will resize large videos to speed things up (Max width 640)
    process_width = 640
    scale = process_width / width if width > process_width else 1.0
    new_height = int(height * scale)

    print("🚀 STARTED! Press 'q' to quit.")

    while True:
        start_time = time.time()
        
        # Read Frame
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Resize for speed
        if scale != 1.0:
            frame = cv2.resize(frame, (process_width, new_height))

        # A. Night Vision (The heavy part)
        enhanced_frame = enhancer.predict(frame)
        
        # Fix memory layout for YOLO
        enhanced_frame = np.ascontiguousarray(enhanced_frame)

        # B. Detection
        results = detector(enhanced_frame, verbose=False)
        annotated_frame = results[0].plot()

        # C. Display
        # Stack images: Top = Original, Bottom = Result
        # (Vertical stacking is better for laptop screens)
        combined = np.vstack((frame, annotated_frame))
        
        cv2.imshow("Night Vision System (Press 'q' to exit)", combined)

        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live()