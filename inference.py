import cv2
import csv
import os
from ultralytics import YOLO

# -------------------------------
# CONFIG
# -------------------------------
model = YOLO("/home/agcl/Desktop/edgefleet trial/runs/detect/yolo11_edgefleet/weights/best.pt")
video_folder = "/home/agcl/Desktop/edgefleet/"
output_folder = "outputs_edgefleet"

os.makedirs(output_folder, exist_ok=True)

# Get list of videos
video_list = [v for v in os.listdir(video_folder) if v.endswith((".mov", ".mp4"))]

print("Found videos:", video_list)

# ---------------------------------------------
# PROCESS EACH VIDEO
# ---------------------------------------------
for video_name in video_list:

    video_path = os.path.join(video_folder, video_name)
    print(f"\nProcessing: {video_path}")

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Get video properties for saving output video
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Output video path
    out_video_path = os.path.join(output_folder, f"{video_name}_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    # CSV file
    csv_path = os.path.join(output_folder, f"{video_name}_centroids.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_index", "x_centroid", "y_centroid", "visibility_flag"])

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        # YOLO inference
        results = model(frame, verbose=False)
        detections = results[0].boxes

        visibility_flag = 0
        cx, cy = -1, -1

        if len(detections) > 0:
            box = detections[0].xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = box

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            visibility_flag = 1

            # Draw centroid
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

        # Save to CSV
        csv_writer.writerow([frame_index, cx, cy, visibility_flag])

        # Annotated video frame
        annotated = results[0].plot()

        # Put frame index on screen
        cv2.putText(
            annotated,
            f"Frame: {frame_index}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 255),
            2
        )

        # Save frame to output video
        out_video.write(annotated)

    cap.release()
    out_video.release()
    csv_file.close()

    print(f"Saved => {csv_path}")
    print(f"Saved => {out_video_path}")

print("\nAll videos processed successfully!")

