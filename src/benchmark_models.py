
import time
import csv
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

# Our two exported model formats
models = [
    ("PyTorch", "yolo11n.pt", "mps" if torch.backends.mps.is_available() else "cpu"),
    ("ONNX", "yolo11n.onnx", "cpu")
]

# Use the same image for both models
image_path = "bus.jpg"

# Create output folder
Path("outputs").mkdir(exist_ok=True)

results = []

for model_name, model_path, device in models:
    print(f"\nTesting {model_name} on {device}...")

    model = YOLO(model_path, task="detect")

    # Warm-up runs
    for _ in range(3):
        model.predict(
            image_path,
            device=device,
            verbose=False
        )

    # Measure repeated predictions
    times = []

    for _ in range(20):
        start = time.perf_counter()

        model.predict(
            image_path,
            device=device,
            verbose=False
        )

        # Wait for Apple GPU work to finish
        if device == "mps":
            torch.mps.synchronize()

        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    average_ms = float(np.mean(times))
    fps = 1000 / average_ms

    results.append({
        "model": model_name,
        "device": device,
        "average_ms": round(average_ms, 2),
        "fps": round(fps, 2)
    })

    print(f"Average latency: {average_ms:.2f} ms")
    print(f"Throughput estimate: {fps:.2f} FPS")

# Save results to CSV
with open("outputs/benchmark_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["model", "device", "average_ms", "fps"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nBenchmark completed!")
print("Saved to outputs/benchmark_results.csv")
