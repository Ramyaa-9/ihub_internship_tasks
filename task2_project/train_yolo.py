from ultralytics import YOLO

model = YOLO("yolov8n.pt") 

results = model.train(data="coco128.yaml", epochs=5, imgsz=640, name="yolov8n_metrics_exp")
