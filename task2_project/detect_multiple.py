from ultralytics import YOLO
import os

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Directory containing input images
input_dir = "images"
output_dir = "runs/segment/predict"

# Find all image files in the folder
images = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not images:
    print("⚠️ No images found in the 'images' folder.")
else:
    print(f"Found {len(images)} images: {[os.path.basename(i) for i in images]}")

    # Run segmentation on all images
    results = model.predict(source=images, task="segment", save=True)

    print("✅ Detection and segmentation complete!")
    print(f"Check your output images here: {output_dir}")

    if os.path.exists(output_dir):
        print("Files in output folder:", os.listdir(output_dir))
