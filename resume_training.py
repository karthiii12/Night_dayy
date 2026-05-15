from ultralytics import YOLO
import os

def resume_process():
    # UPDATED PATH: Pointing to 'heavy_night_model2'
    ckpt_path = "runs/detect/heavy_night_model2/weights/last.pt"

    # Safety Check
    if not os.path.exists(ckpt_path):
        print(f"ERROR: File not found at {ckpt_path}")
        return

    # Load and Resume
    print(f"Loading checkpoint from: {ckpt_path}")
    model = YOLO(ckpt_path)
    
    print("Resuming heavy model training...")
    results = model.train(resume=True)

if __name__ == '__main__':
    resume_process()