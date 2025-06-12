from ultralytics import YOLO
import cv2
import sys
import os

# Usage: python video.py path/to/video.mp4
video_path = 'path/to/default/video.mp4'      ## Run video directly
weights_path = 'weights/yolov8_best.pt'
output_path = 'output/annotated_video.mp4'
os.makedirs('output', exist_ok=True)

model = YOLO(weights_path)
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, conf=0.75, save=False)
    annotated = results[0].plot()
    out.write(annotated)

    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"🖼️ Processed frame {frame_idx}")

cap.release()
out.release()
print(f"✅ Video saved to: {output_path}")
cv2.destroyAllWindows()
print(f"✅ Video saved to: {output_path}")
