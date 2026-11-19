# Nexvia Technologies AI/ML Internship Projects

**Submitted by:** Harsh Gupta

This repository contains two AI/ML internship projects.

## Project 1: Autonomous Driving Perception Stack

Object detection using YOLO11n, vehicle tracking using ByteTrack,
lane detection using OpenCV, and relative depth estimation using MiDaS.

Includes ONNX export, TensorRT FP16 optimization, benchmarking,
architecture documentation, and a recorded demonstration.

## Project 2: Movie Recommendation System

A two-stage recommendation system using TensorFlow, FAISS,
and a deep learning ranking model.

Includes a Streamlit interface, trained models, evaluation scripts,
and a recorded demonstration.

### Project Files

- [Autonomous Driving Source Code](src/)
- [Movie Recommendation System](Movie-Recommendation-System/)
- [Movie Recommendation Demo](Movie-Recommendation-System/movie_demo.mov)
- [Architecture and Presentation](docs/)

## Current Limitations

CARLA/ROS2 integration and some additional infrastructure
components remain future work.

# Autonomous Driving Perception Stack

An AI-based autonomous driving perception project integrating object detection, object tracking, lane detection, relative depth estimation, and NVIDIA TensorRT optimization.

## Project Overview

This project demonstrates several computer vision techniques used in autonomous driving systems. It processes traffic footage and sample KITTI imagery to detect road users, track objects, identify candidate lane markings, and estimate relative scene depth.

The project also explores model deployment and inference optimization using ONNX and NVIDIA TensorRT.

## Technologies Used

- Python
- YOLO11n (Ultralytics)
- ByteTrack
- OpenCV
- MiDaS
- PyTorch
- ONNX and ONNX Runtime
- NVIDIA TensorRT
- CUDA
- Weights & Biases
- KITTI sample imagery

## Implemented Features

### 1. Object Detection

Uses pretrained YOLO11n to detect vehicles, pedestrians, and other supported object classes in images and traffic footage.

### 2. Multi-Object Tracking

Integrates ByteTrack with YOLO to assign tracking IDs to detected objects across video frames.

### 3. Lane Detection

Uses OpenCV edge detection, a region of interest, and Hough line detection to identify candidate road markings.

This is a basic lane-detection demonstration and is not a production-ready lane segmentation model.

### 4. Relative Depth Estimation

Uses MiDaS to generate a relative depth map from camera imagery. The output represents relative scene depth rather than calibrated physical distances.

### 5. Perception Dashboard

Combines saved object-tracking, lane-detection, and depth-estimation outputs into a four-panel visualization.

The current dashboard is a static summary rather than a synchronized real-time pipeline.

### 6. ONNX Export

Exports the YOLO11n PyTorch model to ONNX for portable inference and deployment experiments.

### 7. TensorRT Optimization

Builds an FP16 TensorRT inference engine on an NVIDIA GeForce RTX 4060 Laptop GPU and compares it with PyTorch FP32 inference.

### 8. Experiment Tracking

Uses Weights & Biases to record benchmark metrics and experiment results.

## TensorRT Benchmark Results

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)

Model: YOLO11n

| Metric | PyTorch FP32 | TensorRT FP16 |
|---|---:|---:|
| Average latency | 25.85 ms | 13.06 ms |
| Estimated FPS | 38.68 | 76.59 |

**Observed speedup:** 1.98×  
**Latency reduction:** 49.5%

Benchmark methodology: 30 prediction calls per model after 5 warmup runs, including preprocessing and postprocessing. FPS values are estimated from average latency, not measured end-to-end video throughput.

These results describe the tested hardware and benchmark setup and may vary on other systems.

## Project Structure

```text
Autonomous-Driving-Project/
├── data/                 # Input videos and KITTI sample imagery
├── docs/                 # Documentation and TensorRT evidence
├── models/               # Model-related artifacts
├── outputs/              # Generated perception results
├── runs/                 # Ultralytics experiment outputs
├── src/
│   ├── detect_objects.py
│   ├── detect_video.py
│   ├── track_vehicles.py
│   ├── detect_lanes.py
│   ├── estimate_depth.py
│   ├── combine_perception.py
│   ├── export_onnx.py
│   ├── test_onnx.py
│   ├── benchmark_models.py
│   └── log_benchmarks.py
├── README.md
├── requirements.txt
└── .gitignore
```

The TensorRT engine, Windows benchmark script, and benchmark evidence are packaged separately in `docs/TensorRT_Submission.zip`.

## Installation

Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows, activate the virtual environment using:

```powershell
.\.venv\Scripts\Activate.ps1
```

NVIDIA TensorRT requires a compatible NVIDIA GPU and supporting CUDA/TensorRT environment. The TensorRT engine is hardware- and software-environment-dependent and may need to be rebuilt on another machine.

## Running the Project

From the project root, run the relevant scripts:

```bash
python src/detect_objects.py
python src/detect_video.py
python src/track_vehicles.py
python src/detect_lanes.py
python src/estimate_depth.py
python src/combine_perception.py
```

Some scripts use configured local input paths and require the corresponding input video or image to be available.

## Dataset

The project uses a sample image from a community KITTI demonstration repository and separately sourced traffic footage.

The KITTI sample demonstrates dataset-based object detection. It does not represent training or evaluation on the full official KITTI benchmark.

## Current Limitations and Remaining Work

- Lane detection is sensitive to camera viewpoint and background objects.
- MiDaS provides relative depth estimates, not exact distance measurements.
- The dashboard combines saved outputs rather than running all models simultaneously.
- CARLA simulator integration has not yet been implemented.
- ROS2 communication has not yet been implemented.
- Full KITTI benchmark evaluation has not been performed.

## Conclusion

This project demonstrates a modular computer vision perception workflow and compares standard PyTorch inference against NVIDIA TensorRT optimization.

The TensorRT experiment achieved approximately 1.98× lower-latency inference speedup under the measured benchmark conditions. Further work includes simulation, ROS2 integration, and more robust lane and depth perception.
## Project Demo

[Watch Demo Video](https://drive.google.com/file/d/1iy4lGO_Pm9oL_oFLBye8fFXuHc3Bp7_Y/view?usp=sharing)