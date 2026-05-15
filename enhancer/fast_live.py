import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
import time

# --- SPEED SETTINGS ---
VIDEO_SOURCE = "test1.mp4"          # 0 for Webcam, or "filename.mp4"
PROCESS_WIDTH = 320       # Low res for high speed
SKIP_FRAMES = 2           # Skip 2 frames for every 1 processed
# ----------------------

# Setup Paths
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, "EnlightenGAN-inference"))

try:
    from enlighten_inference import EnlightenOnnxModel
except ImportError:
    print("❌ ERROR: Run this from the 'enhancer' folder.")
    sys.exit()

def run_fast_live():
    print("⏳ Loading Models (Optimized Mode)...")
    enhancer = EnlightenOnnxModel()
    model_path = os.path.abspath(os.path.join(current_dir, "..", "runs", "detect", "heavy_night_model2", "weights", "best.pt"))
    detector = YOLO(model_path)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    # Get original size
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate new size
    scale = PROCESS_WIDTH / width
    new_height = int(height * scale)
    
    print(f"🚀 RUNNING! Resolution: {PROCESS_WIDTH}x{new_height}")
    print("Press 'q' to quit.")

    frame_count = 0
    last_annotated_frame = None

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret: break

        # 1. Resize immediately
        small_frame = cv2.resize(frame, (PROCESS_WIDTH, new_height))

        # 2. Smart Skipping
        if frame_count % (SKIP_FRAMES + 1) == 0:
            # A. Night Vision
            enhanced_frame = enhancer.predict(small_frame)
            enhanced_frame = np.ascontiguousarray(enhanced_frame)

            # B. Detection
            results = detector(enhanced_frame, verbose=False)
            last_annotated_frame = results[0].plot()
        
        # Use previous frame if skipping (Visual smoothing)
        if last_annotated_frame is None:
            display_image = small_frame
        else:
            display_image = last_annotated_frame

        # 3. Stack and Show
        combined = np.vstack((small_frame, display_image))
        display_scale = 2
        final_display = cv2.resize(combined, (0, 0), fx=display_scale, fy=display_scale)

        cv2.imshow("Fast Night Vision (Press 'q')", final_display)

        frame_count += 1
        
        # --- THE FIX IS HERE ---
        elapsed = time.time() - start_time
        # Add a tiny number (1e-6) to prevent division by zero
        fps = 1.0 / (elapsed + 0.000001) 
        print(f"FPS: {fps:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_fast_live()