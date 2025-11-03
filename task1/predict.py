from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-seg.pt")  # load an official model
model = YOLO("runs/segment/train/weights/best.pt")  # load a custom model

# Predict with the model
results = model("topgun.jpeg", save=True, project="my_outputs", name="bus_prediction")

# Access the results
for result in results:
    xy = result.masks.xy  # mask in polygon format
    xyn = result.masks.xyn  # normalized
    masks = result.masks.data  # mask in matrix format (num_objects x H x W)