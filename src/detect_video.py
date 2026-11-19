
from ultralytics import YOLO

# Load our pretrained object detection model
model = YOLO("yolo11n.pt")

# Run detection on a sample traffic video
results = model.predict(
    source="data/traffic.mp4",
    device="mps",
    save=True,
    stream=True
)

# Process the video frame by frame
for frame in results:
    print("Processed a frame!")

print("Video detection completed!")
