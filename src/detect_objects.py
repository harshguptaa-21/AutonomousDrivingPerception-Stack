
from ultralytics import YOLO

# Load a small, pretrained YOLO model
model = YOLO("yolo11n.pt")

# Detect objects in a sample image
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    device="mps",
    save=True
)

print("Object detection completed!")
