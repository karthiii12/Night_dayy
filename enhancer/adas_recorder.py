import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
import time

# --- SETTINGS ---
INPUT_FILE = "test1.mp4"       # Your input video
OUTPUT_FILE = "adas_final_demo.mp4" # The smooth result
PROCESS_WIDTH = 320            # Keep it 320 for consistency
WARNING_DISTANCE_THRESHOLD = 0.25 
# ----------------

current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, "EnlightenGAN-inference"))

try:
    from enlighten_inference import EnlightenOnnxModel
except ImportError:
    print("❌ ERROR: Run this from the 'enhancer' folder.")
    sys.exit()

def get_distance_level(box_height, img_height):
    ratio = box_height / img_height
    if ratio > WARNING_DISTANCE_THRESHOLD:
        return "CLOSE", (0, 0, 255)
    elif ratio > 0.1:
        return "MEDIUM", (0, 255, 255)
    else:
        return "FAR", (0, 255, 0)

def run_recorder():
    print("🎥 INITIALIZING VIDEO RENDERER...")
    
    # 1. Load Models
    enhancer = EnlightenOnnxModel()
    model_path = os.path.abspath(os.path.join(current_dir, "..", "runs", "detect", "heavy_night_model2", "weights", "best.pt"))
    detector = YOLO(model_path)

    # 2. Input Video
    cap = cv2.VideoCapture(INPUT_FILE)
    if not cap.isOpened():
        print(f"❌ Error: Could not open {INPUT_FILE}")
        return

    # Get video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate new dimensions
    scale = PROCESS_WIDTH / orig_width
    new_height = int(orig_height * scale)
    
    # We will display/save at 2x scale so it's not tiny
    display_scale = 2
    final_w = PROCESS_WIDTH * display_scale
    final_h = new_height * display_scale

    # 3. Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30.0, (final_w, final_h))

    lane_center_x = PROCESS_WIDTH // 2
    lane_width = PROCESS_WIDTH // 3

    print(f"🚀 RENDERING STARTED! Total Frames: {total_frames}")
    print("   (This will run slower than real-time, but the output will be smooth)")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        # A. Night Vision
        small_frame = cv2.resize(frame, (PROCESS_WIDTH, new_height))
        enhanced_frame = enhancer.predict(small_frame)
        enhanced_frame = np.ascontiguousarray(enhanced_frame)

        # B. Detection
        results = detector(enhanced_frame, verbose=False)
        
        # C. Draw Logic
        display_frame = enhanced_frame.copy()
        
        # Virtual Lane
        cv2.line(display_frame, (lane_center_x - lane_width//2, new_height), (lane_center_x - 50, new_height//2), (255, 255, 255), 1)
        cv2.line(display_frame, (lane_center_x + lane_width//2, new_height), (lane_center_x + 50, new_height//2), (255, 255, 255), 1)

        danger_detected = False

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            obj_center_x = x1 + (w // 2)

            in_lane = (lane_center_x - lane_width//2) < obj_center_x < (lane_center_x + lane_width//2)
            dist_status, color = get_distance_level(h, new_height)

            if dist_status == "CLOSE" and in_lane:
                danger_detected = True
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(display_frame, "WARNING", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)

        if danger_detected:
            cv2.putText(display_frame, "BRAKE!", (PROCESS_WIDTH//2 - 40, new_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # D. Save Frame
        # Scale up x2 for the final video
        final_view = cv2.resize(display_frame, (0, 0), fx=display_scale, fy=display_scale)
        out.write(final_view)

        # Optional: Show progress on screen (press q to stop early)
        cv2.imshow("Rendering...", final_view)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"   Processed: {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ DONE! Saved smooth video to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_recorder()