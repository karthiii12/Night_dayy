import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
import winsound

# --- ADAS SETTINGS ---
VIDEO_SOURCE = "test3.mp4"  # Use a driving video or 0 for webcam
PROCESS_WIDTH = 320         # Keep it fast
WARNING_DISTANCE_THRESHOLD = 0.25  # If object takes up > 25% of screen height, it's "CLOSE"
# ---------------------

# --- PROJECT PATHS (RELATIVE, SAFE FROM ANY WORKING DIR) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENHANCER_DIR = os.path.join(BASE_DIR, "enhancer", "EnlightenGAN-inference")
WEIGHTS_PATH = os.path.join(
    BASE_DIR, "runs", "detect", "heavy_night_model2", "weights", "best.pt"
)

if ENHANCER_DIR not in sys.path:
    sys.path.append(ENHANCER_DIR)

try:
    from enlighten_inference import EnlightenOnnxModel
except ImportError:
    print(f"❌ ERROR: Cannot import EnlightenGAN from: {ENHANCER_DIR}")
    sys.exit(1)

def get_distance_level(box_height, img_height):
    """
    Estimate distance based on how tall the object is in the frame.
    Returns: 'FAR', 'MEDIUM', or 'CLOSE'
    """
    ratio = box_height / img_height
    if ratio > WARNING_DISTANCE_THRESHOLD:
        return "CLOSE", (0, 0, 255) # Red
    elif ratio > 0.1:
        return "MEDIUM", (0, 255, 255) # Yellow
    else:
        return "FAR", (0, 255, 0) # Green

def run_adas():
    print("🚗 STARTING ADAS NIGHT VISION SYSTEM...")
    
    # 1. Load Models
    enhancer = EnlightenOnnxModel()
    detector = YOLO(WEIGHTS_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    # Speed setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = PROCESS_WIDTH / width
    new_height = int(height * scale)

    # Define the "Driving Lane" (Center of screen)
    lane_center_x = PROCESS_WIDTH // 2
    lane_width = PROCESS_WIDTH // 3  # Approximate lane width

    print("✅ ADAS ACTIVE. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Enhance (Night Vision)
        small_frame = cv2.resize(frame, (PROCESS_WIDTH, new_height))
        enhanced_frame = enhancer.predict(small_frame)
        enhanced_frame = np.ascontiguousarray(enhanced_frame)

        # 2. Detect
        results = detector(enhanced_frame, verbose=False)
        
        # 3. ADAS Logic Layer
        display_frame = enhanced_frame.copy()
        
        # Draw "Virtual Lane" markers (just for visuals)
        cv2.line(display_frame, (lane_center_x - lane_width//2, new_height), (lane_center_x - 50, new_height//2), (255, 255, 255), 1)
        cv2.line(display_frame, (lane_center_x + lane_width//2, new_height), (lane_center_x + 50, new_height//2), (255, 255, 255), 1)

        danger_detected = False

        for box in results[0].boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            
            # Calculate Center of Object
            obj_center_x = x1 + (w // 2)

            # Check 1: Is it in my lane?
            in_lane = (lane_center_x - lane_width//2) < obj_center_x < (lane_center_x + lane_width//2)
            
            # Check 2: How close is it?
            dist_status, color = get_distance_level(h, new_height)

            # LOGIC: Only alert if it's CLOSE and IN LANE (or very close generally)
            if dist_status == "CLOSE" and in_lane:
                danger_detected = True
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(display_frame, "⚠️ COLLISION WARNING", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                # Just draw normal boxes for awareness
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)

        # 4. HUD Warning Overlay
        if danger_detected:
            cv2.putText(display_frame, "BRAKE!", (PROCESS_WIDTH//2 - 40, new_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            # Async sound so video doesn't lag
            winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)

        # 5. Display (Scale up x2 for easier viewing)
        final_view = cv2.resize(display_frame, (0, 0), fx=2, fy=2)
        cv2.imshow("ADAS Night Vision HUD", final_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_adas()