from ultralytics import YOLO
import sys
from pathlib import Path
import cv2

# Usage: python predict.py path/to/image.jpg
image_path = image_path = sys.argv[1] if len(sys.argv) > 1 else 'path/to/default/image.jpg'            ## Run image from terminal or directly
weights_path = 'weights/yolov8_best.pt' 

model = YOLO(weights_path)
results = model.predict(source=image_path, save=True)
print(f"Prediction complete. Saved to: {results[0].save_dir}")
