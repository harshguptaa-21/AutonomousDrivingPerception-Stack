
from ultralytics import YOLO

# Load our pretrained YOLO model
model = YOLO("yolo11n.pt")

# Export it to ONNX format
model.export(format="onnx", imgsz=640, device="cpu")

print("YOLO successfully exported to ONNX!")
