# Furniture Detection with YOLOv5

This repository contains the implementation of a furniture detection system using the YOLOv5 model. The system is designed to identify mirrors and sofas in images. The project covers data preparation, model training, evaluation, and inference stages.

## Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Inference](#inference)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

This project aims to create an efficient object detection system for identifying specific furniture items, such as mirrors and sofas, using the YOLOv5 architecture. The model is trained on a custom dataset and evaluated using metrics like Intersection over Union (IoU), precision, recall, and F1-score.

## Installation

To replicate this project, follow these steps:

1. Clone the YOLOv5 repository and install the necessary dependencies:
   ```bash
   git clone https://github.com/ultralytics/yolov5
   cd yolov5
   pip install -r requirements.txt
   pip install roboflow
   ```

2. Download the dataset from Roboflow:
   ```python
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_API_KEY")
   project = rf.workspace("workspace-name").project("project-name")
   dataset = project.version(1).download("yolov5")
   ```

## Dataset

The dataset contains images annotated with bounding boxes for mirrors and sofas. It is divided into training, validation, and test sets to ensure comprehensive model evaluation.

## Model Training

Train the YOLOv5 model using the following command:
```bash
python train.py --img 416 --batch 16 --epochs 10 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache
```

This command trains the model with images resized to 416x416 pixels, a batch size of 16, and for 10 epochs. The weights are initialized from the pre-trained `yolov5s.pt` model.

## Model Evaluation

Evaluate the model using TensorBoard to visualize the metrics:
```bash
tensorboard --logdir runs
```

For detailed evaluation, inspect the model's predictions directly:
```python
import torch
from utils.general import non_max_suppression

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')

# Perform inference
results = model('/content/my-datasets/test/images')

# Extract predictions
predictions = results.pred[0]
```

## Inference

Run inference on new images with the following command:
```bash
python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.1 --source {dataset.location}/test/images
```

## Results

The model detects mirrors and sofas in the test images, displaying bounding boxes and confidence scores. Example outputs include:
```plaintext
image 1/11: 2 mirrors, 218.6ms
image 2/11: 3 mirrors, 2 sofas, 194.8ms
...
```

## Deployment
medium: https://medium.com/@hamidantiprita/application-of-yolov5-algorithm-in-furniture-detection-e1b751170031 
Dataset Roboflow: https://app.roboflow.com/prita-hamidanti-dpbne/dataset-uas/1 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
