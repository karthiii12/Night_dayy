from ultralytics import YOLO
import os

# Force the path to be relative to the script
yaml_path = os.path.join(os.getcwd(), "data.yaml")
model_path = "runs/detect/heavy_night_model2/weights/best.pt"

print(f"Looking for data at: {yaml_path}")

# Run Validation
model = YOLO(model_path)
model.val(data=yaml_path, split='test')