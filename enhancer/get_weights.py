import os
import requests

# The correct link for the ONNX version (ArsenyInfo repo)
url = "https://github.com/arsenyinfo/EnlightenGAN-inference/raw/main/enlighten_inference/enlighten.onnx"
save_path = os.path.join("EnlightenGAN-inference", "enlighten_inference", "enlighten.onnx")

print(f"Downloading Night Vision weights to: {save_path}...")

# Create folder if it doesn't exist
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Download
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✅ SUCCESS: 'enlighten.onnx' downloaded successfully!")
    print("You are ready for the test run.")
else:
    print(f"❌ ERROR: Failed to download. Status code: {response.status_code}")