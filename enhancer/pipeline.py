import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO

# --- SETUP PATHS ---
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, "EnlightenGAN-inference"))

try:
    from enlighten_inference import EnlightenOnnxModel
except ImportError:
    print("❌ ERROR: Could not import EnlightenGAN. Run this from the 'enhancer' folder.")
    sys.exit()

def run_pipeline():
    # 1. LOAD PLAYER 1 (Night Vision)
    print("🔌 Loading EnlightenGAN...")
    # NOTE: The warning about 'CUDAExecutionProvider' is normal on some laptops. 
    # It just means it will run on CPU, which is fine for this test.
    enhancer = EnlightenOnnxModel()

    # 2. LOAD PLAYER 2 (YOLO Brain)
    model_path = os.path.abspath(os.path.join(current_dir, "..", "runs", "detect", "heavy_night_model2", "weights", "best.pt"))
    
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Could not find best.pt at: {model_path}")
        return

    print(f"🧠 Loading YOLO from: {os.path.basename(model_path)}")
    detector = YOLO(model_path)

    # 3. GET A DARK IMAGE
    dataset_path = os.path.abspath(os.path.join(current_dir, "..", "dataset", "test", "images"))
    from glob import glob
    images = glob(os.path.join(dataset_path, "*.jpg")) + glob(os.path.join(dataset_path, "*.png"))
    
    if not images:
        print("❌ No images found.")
        return
    
    # Process 3 images
    print(f"🚀 Processing 3 images...")
    
    for i in range(min(3, len(images))):
        img_path = images[i]
        filename = os.path.basename(img_path)
        print(f"   Processing: {filename}")

        # Step A: Read Dark Image
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"   ⚠️ Skipping {filename} (could not read)")
            continue

        # Step B: Brighten (Player 1)
        bright_img = enhancer.predict(original_img)

        # --- THE FIX IS HERE ---
        # We force the image memory to be 'contiguous' so YOLO doesn't crash
        bright_img = np.ascontiguousarray(bright_img)
        # -----------------------

        # Step C: Detect (Player 2)
        results = detector(bright_img, verbose=False)

        # Step D: Draw Boxes
        final_result = results[0].plot()

        # Step E: Save Comparison
        # Resize to make them match height if needed
        h, w = original_img.shape[:2]
        
        # Combine horizontally (Original | Bright | Detected)
        combined = np.hstack((original_img, bright_img, final_result))
        
        save_name = f"result_{filename}"
        cv2.imwrite(save_name, combined)
        print(f"   ✅ Saved: {save_name}")

    print("\n🎉 PIPELINE COMPLETE!")
    print("Check the 'result_*.jpg' files in your folder.")

if __name__ == "__main__":
    run_pipeline()