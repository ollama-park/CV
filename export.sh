yolo export \
  model=runs/detect/engine_bay_runs/yolo11_engine/weights/best.pt \
  format=onnx \
  imgsz=640 \
  opset=12 \
  simplify=True \
  nms=True \
  conf=0.23 \
  iou=0.6

  