from ultralytics import YOLO
import cv2
import sys

# Usage: python video.py path/to/video.mp4
video_path = sys.argv[1] if len(sys.argv) > 1 else 'path/to/default/video.mp4'      ## Run image from terminal or directly
weights_path = 'weights/yolov8_best.pt'

model = YOLO(weights_path)
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, conf=0.7, save=False)
    annotated = results[0].plot()
    cv2.imshow('Prediction', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
