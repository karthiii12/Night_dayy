from ultralytics import YOLO

def start_training():
    # 1. Load the "Extra Large" Model (The most powerful one)
    print("Loading the Heavy Model (YOLOv8x)...")
    model = YOLO('yolov8x.pt') 

    # 2. Train it
    print("Starting Training on RTX 3050...")
    results = model.train(
        data='dataset/data.yaml', # We will create this file next
        epochs=50,                # Let's start with 50 to see results faster
        imgsz=640,                # High Quality Standard
        batch=2,                  # CRITICAL: Keep this at 2 for your 6GB VRAM
        nbs=64,                   # Accumulate gradients (Simulates a bigger GPU)
        name='heavy_night_model'
    )

if __name__ == '__main__':
    start_training()