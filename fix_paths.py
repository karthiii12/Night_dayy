import os
from ultralytics import YOLO

def find_file(filename, search_path):
    print(f"Searching for {filename} inside {search_path}...")
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            full_path = os.path.join(root, filename)
            print(f"✅ FOUND IT: {full_path}")
            return full_path
    return None

def main():
    # 1. Auto-find data.yaml
    current_dir = os.getcwd()
    yaml_path = find_file("data.yaml", current_dir)

    # 2. Auto-find best.pt
    model_path = find_file("best.pt", current_dir)

    # 3. Run Validation (Only if both are found)
    if yaml_path and model_path:
        print("🚀 Starting Report Generation...")
        model = YOLO(model_path)
        # Running validation with workers=0 to be extra safe on Windows
        model.val(data=yaml_path, split='test', workers=0)
    else:
        print("❌ CRITICAL: Could not find files. Check if you are in the 'Final_Project_Heavy' folder.")

if __name__ == '__main__':
    main()