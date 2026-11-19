
import cv2
import numpy as np
from pathlib import Path

# Input files
original_path = "outputs/depth_original.jpg"
depth_path = "outputs/depth_map.jpg"
lane_video_path = "outputs/lane_detection.mp4"

# Find the tracking video
tracking_files = list(
    Path("runs/detect").glob("outputs/tracking*/traffic.mp4")
)

if not tracking_files:
    raise FileNotFoundError("Tracking video not found")

tracking_path = str(tracking_files[-1])

# Read images
original = cv2.imread(original_path)
depth = cv2.imread(depth_path)

if original is None or depth is None:
    raise RuntimeError("Could not read depth images")

# Read first frames of videos
lane_video = cv2.VideoCapture(lane_video_path)
tracking_video = cv2.VideoCapture(tracking_path)

lane_ok, lane_frame = lane_video.read()
tracking_ok, tracking_frame = tracking_video.read()

lane_video.release()
tracking_video.release()

if not lane_ok or not tracking_ok:
    raise RuntimeError("Could not read video frames")

# Make every image the same size
size = (640, 360)

original = cv2.resize(original, size)
depth = cv2.resize(depth, size)
lane_frame = cv2.resize(lane_frame, size)
tracking_frame = cv2.resize(tracking_frame, size)

# Add labels
panels = [
    ("Original Camera", original),
    ("YOLO + ByteTrack", tracking_frame),
    ("OpenCV Lane Detection", lane_frame),
    ("MiDaS Relative Depth", depth)
]

for label, image in panels:
    cv2.rectangle(image, (0, 0), (320, 40), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

# Arrange into a 2x2 dashboard
top = np.hstack([original, tracking_frame])
bottom = np.hstack([lane_frame, depth])
dashboard = np.vstack([top, bottom])

# Save the result
Path("outputs").mkdir(exist_ok=True)

cv2.imwrite("outputs/perception_dashboard.jpg", dashboard)

print("Combined perception dashboard created!")
print("Saved to outputs/perception_dashboard.jpg")
