import sys
import os
import cv2
import numpy as np
from glob import glob

# 1. Setup paths
current_dir = os.getcwd()
# Add the repo we cloned to the system path so Python can find it
sys.path.append(os.path.join(current_dir, "EnlightenGAN-inference"))

try:
    from enlighten_inference import EnlightenOnnxModel
except ImportError:
    print("❌ ERROR: Could not import EnlightenGAN. Make sure you are in the 'enhancer' folder.")
    sys.exit()

def run_test():
    # 2. Load the Night Vision Brain
    print("🔌 Loading EnlightenGAN model...")
    model = EnlightenOnnxModel()

    # 3. Find a dark image from your dataset
    # We look in the main project dataset folder
    dataset_path = os.path.abspath(os.path.join(current_dir, "..", "dataset", "test", "images"))
    images = glob(os.path.join(dataset_path, "*.jpg")) + glob(os.path.join(dataset_path, "*.png"))

    if not images:
        print(f"❌ No images found in {dataset_path}")
        return

    # Pick the first image found
    test_image_path = images[0]
    print(f"📸 Testing on: {os.path.basename(test_image_path)}")

    # 4. Process the image
    img = cv2.imread(test_image_path)
    if img is None:
        print("❌ Failed to load image.")
        return

    # Run the "Enlighten" process
    processed = model.predict(img)

    # 5. Save the result
    output_filename = "test_result.jpg"
    # Stack images side-by-side (Left: Original, Right: Brightened)
    # Resize for display if needed
    h, w = img.shape[:2]
    combined = np.hstack((img, processed))
    
    cv2.imwrite(output_filename, combined)
    print(f"✅ SUCCESS! Check the file '{output_filename}' in your enhancer folder.")
    print("It shows the Before vs. After comparison.")

if __name__ == "__main__":
    run_test()