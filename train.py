from ultralytics import YOLO
import os

# -------------------------------
# 1. DATASET PATH
# -------------------------------
dataset_path = "/home/agcl/Desktop/edgefleet trial/"  
# Example: "/home/agcl/Desktop/edgefleet trial/"  
# It must contain data.yaml

data_yaml = os.path.join(dataset_path, "data.yaml")
print("Using:", data_yaml)

# -------------------------------
# 2. LOAD YOLO11 MODEL
# -------------------------------
model = YOLO("yolo11n.pt")   # or yolo11s.pt / yolo11m.pt / yolo11l.pt

# -------------------------------
# 3. TRAIN THE MODEL
# -------------------------------
model.train(
    data=data_yaml,
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,           # GPU (use "cpu" if no GPU)
    name="yolo11_edgefleet"
)

