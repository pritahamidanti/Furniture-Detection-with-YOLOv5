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

## Deployment
medium: https://medium.com/@hamidantiprita/application-of-yolov5-algorithm-in-furniture-detection-e1b751170031 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
