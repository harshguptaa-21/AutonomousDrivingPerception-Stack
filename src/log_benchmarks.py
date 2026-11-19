
import csv
import wandb

# Start a W&B experiment
run = wandb.init(
    project="autonomous-driving-perception",
    name="yolo-model-benchmark"
)

# Read our benchmark results
with open("outputs/benchmark_results.csv", "r") as file:
    results = list(csv.DictReader(file))

# Create a comparison table
table = wandb.Table(
    columns=["Model", "Device", "Latency (ms)", "Estimated FPS"]
)

for row in results:
    table.add_data(
        row["model"],
        row["device"],
        float(row["average_ms"]),
        float(row["fps"])
    )

wandb.log({"benchmark_comparison": table})

# Log individual measurements
for row in results:
    wandb.log({
        "model": row["model"],
        "device": row["device"],
        "average_latency_ms": float(row["average_ms"]),
        "estimated_fps": float(row["fps"])
    })

run.finish()

print("Benchmark results uploaded successfully!")
