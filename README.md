# Surgical Tool Detection with YOLOv11
This repository contains code and configuration for a computer vision project using YOLOv11 to detect surgical tools in operating room videos.

## Environment Setup
Use the following command to install required packages:
```bash
pip install -r requirements.txt
```

## Run Prediction on Image
```bash
python predict.py /path/to/best.pt /path/to/image.jpg
```

## Run Prediction on Video
```bash
python video.py /path/to/best.pt /path/to/video.mp4 /path/to/output_directory
```

## Final Model Weights
The final model weights can be downloaded from the following link:
[Download best.pt](https://drive.google.com/file/d/your_model_link_here/view?usp=sharing)
