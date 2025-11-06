from ultralytics import YOLO
import os
import time

INPUT_VIDEO_PATH = os.path.join(os.path.dirname(__file__), "input_video", "traffic_footage_5s.mp4")
OUTPUT_VIDEO_NAME = "traffic_analysis_output.mp4" 

model = YOLO("yolov8n-seg.pt")

print(f"Starting video analysis on: {INPUT_VIDEO_PATH}")

results = model.predict(source=INPUT_VIDEO_PATH, 
                        save=True, 
                        conf=0.25,
                        show_labels=True,
                        show_conf=True,
                        show_boxes=True,
                        imgsz=640)

ultralytics_runs_base = os.path.join(os.getcwd(), "runs/segment/") 

actual_output_folder = None
latest_time = 0

if os.path.exists(ultralytics_runs_base):
    for folder_name in os.listdir(ultralytics_runs_base):
        full_path = os.path.join(ultralytics_runs_base, folder_name)
        if os.path.isdir(full_path) and "predict" in folder_name:
            mod_time = os.path.getmtime(full_path)
            if mod_time > latest_time:
                latest_time = mod_time
                actual_output_folder = full_path

if actual_output_folder and os.path.exists(actual_output_folder):
    print(f"Video analysis complete! Output saved to: {actual_output_folder}")
    print(f"Check for your output video ({OUTPUT_VIDEO_NAME}) in this folder.")
else:
    print("Output folder not found automatically. Please check 'runs/segment/' manually within your ihubFiles folder.")