yolo detect train \
  model=yolo11n.pt \
  data=Car-Engine-Bay-1/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=engine_bay_runs \
  name=yolo11_engine