from ultralytics import YOLO
import torch
import torch.nn as nn
import onnx

# Load trained model
model = YOLO("runs/detect/engine_bay_runs/yolo11_engine/weights/best.pt")

# Export with NMS
model.export(format="onnx", imgsz=640, opset=12, simplify=True, nms=True)