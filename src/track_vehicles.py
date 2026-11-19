
from ultralytics import YOLO

# Load the object detection model
model = YOLO("yolo11n.pt")

# Track vehicles across video frames
results = model.track(
    source="data/traffic.mp4",
    tracker="bytetrack.yaml",
    device="mps",
    classes=[2, 3, 5, 7],
    conf=0.25,
    save=True,
    stream=True,
    persist=True,
    project="outputs",
    name="tracking"
)

# Read the tracking IDs for each frame
for frame in results:
    if frame.boxes.id is not None:
        ids = frame.boxes.id.int().cpu().tolist()
        print("Vehicle IDs:", ids)

print("Vehicle tracking completed!")
