# YOLO11 Custom Object Detection Dataset & Training

This project contains a custom annotated dataset prepared for training a YOLO11 object detection model.  
All images were extracted from a video, annotated using Roboflow, and exported in YOLO format.

Dataset Overview

- Total extracted video frames: 373 for training
- All  frames manually annotated in Roboflow
- Roboflow automatically augmented/split the dataset into:

| Split        Images |
|----------------------|
| Training      969 |
| Validation    30 |
| Test          20 |


---

## Dataset Format

The dataset is exported in YOLO format.


Example `data.yaml
yaml
train: dataset/images/train
val: dataset/images/val
test: dataset/images/test

nc: 1
names: ["cricket ball"]  

requirments
pip install ultralytics

runs/detect/train/
│── weights/best.pt
│── results.png
│── confusion_matrix.png
│── precision_recall_curve.png

## Download Dataset & Weights

You can download the complete dataset, annotations, and trained YOLO11 model from Google Drive:

Google Drive Download: https://drive.google.com/drive/folders/1YYfcfAZo2vnRVzgAraPdJPr-ZXd6lP47?usp=sharing

After downloading, extract the folder and run the training script as described above.





