
from ultralytics import YOLO
import time

# Load exported ONNX model
model = YOLO("yolo11n.onnx", task="detect")

# Run detection on our traffic image
start = time.perf_counter()

results = model.predict(
    source="bus.jpg",
    device="cpu",
    save=True,
    project="outputs",
    name="onnx_test",
    exist_ok=True
)

elapsed = time.perf_counter() - start

print("ONNX inference completed!")
print("Objects detected:", len(results[0].boxes))
print(f"Total execution time: {elapsed:.3f} seconds")
