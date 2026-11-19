
import cv2
import numpy as np
from pathlib import Path

# ----------------------------------
# 1. INPUT AND OUTPUT VIDEO PATHS
# ----------------------------------

input_path = "data/traffic.mp4"
output_path = "outputs/lane_detection.mp4"

Path("outputs").mkdir(exist_ok=True)

video = cv2.VideoCapture(input_path)

if not video.isOpened():
    raise RuntimeError("Could not open input video")

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    video.release()
    raise RuntimeError("Could not create output video")


# ----------------------------------
# 2. PROCESS VIDEO FRAME BY FRAME
# ----------------------------------

while True:
    success, frame = video.read()

    if not success:
        break

    # Convert image to HLS color space
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # Detect bright white markings
    white_mask = cv2.inRange(
        hls,
        np.array([0, 170, 0], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8)
    )

    # ----------------------------------
    # 3. SELECT THE ROAD REGION
    # ----------------------------------

    mask = np.zeros_like(white_mask)

    polygon = np.array([[
        (0, height - 1),
        (0, int(height * 0.48)),
        (width - 1, int(height * 0.48)),
        (width - 1, height - 1)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)

    road_mask = cv2.bitwise_and(
        white_mask,
        mask
    )

    # ----------------------------------
    # 4. DETECT EDGES
    # ----------------------------------

    blurred = cv2.GaussianBlur(
        road_mask,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    # ----------------------------------
    # 5. DETECT STRAIGHT LINES
    # ----------------------------------

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=35,
        maxLineGap=15
    )

    # ----------------------------------
    # 6. DRAW DETECTED LINES
    # ----------------------------------

    if lines is not None:
        for line in lines:

            x1, y1, x2, y2 = line.flatten()

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

    # Save the processed frame
    writer.write(frame)


# ----------------------------------
# 7. FINISH AND SAVE VIDEO
# ----------------------------------

video.release()
writer.release()

print("Lane detection completed!")
print("Saved to:", output_path)
