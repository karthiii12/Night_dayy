# Create a file: install_requirements.py
import os

print("Installing the Power Tools...")
# Installs the ultralytics library for YOLO
os.system("pip install ultralytics") 
# Installs PyTorch with CUDA support (Essential for your RTX 3050)
os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("Done! You are ready.")