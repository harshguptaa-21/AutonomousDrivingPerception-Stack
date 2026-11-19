
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from pathlib import Path

# 1. Choose device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# 2. Load pretrained MiDaS model
model_name = "Intel/dpt-hybrid-midas"

processor = DPTImageProcessor.from_pretrained(model_name)
model = DPTForDepthEstimation.from_pretrained(model_name)
model.to(device)
model.eval()

# 3. Read first frame of our traffic video
video = cv2.VideoCapture("data/traffic.mp4")

success, frame = video.read()
video.release()

if not success:
    raise RuntimeError("Could not read video frame")

# Convert OpenCV BGR image to RGB
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image = Image.fromarray(rgb)

# 4. Prepare image for the AI model
inputs = processor(images=image, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

# 5. Predict relative depth
with torch.no_grad():
    outputs = model(**inputs)
    depth = outputs.predicted_depth

# 6. Resize depth map to original image size
depth = torch.nn.functional.interpolate(
    depth.unsqueeze(1),
    size=frame.shape[:2],
    mode="bicubic",
    align_corners=False
).squeeze().cpu().numpy()

# 7. Convert depth predictions to visible colors
depth_normalized = cv2.normalize(
    depth, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

depth_colored = cv2.applyColorMap(
    depth_normalized,
    cv2.COLORMAP_INFERNO
)

# 8. Save results
Path("outputs").mkdir(exist_ok=True)

cv2.imwrite("outputs/depth_original.jpg", frame)
cv2.imwrite("outputs/depth_map.jpg", depth_colored)

print("Depth estimation completed!")
print("Saved to outputs/depth_map.jpg")
