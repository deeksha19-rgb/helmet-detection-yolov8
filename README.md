# Helmet Detection System using YOLOv8

An AI-based object detection project that identifies motorcycle and bicycle riders wearing helmets and those without helmets using YOLOv8.

## Project Overview

This project implements an object detection pipeline using YOLOv8n to detect two classes:

- With Helmet
- Without Helmet

The project focuses on dataset preprocessing, annotation conversion, model training, performance evaluation, and analysis of class imbalance.

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- Google Colab
- Kaggle Dataset

## Dataset

- **Source:** Helmet Detection dataset by Andrew MVD on Kaggle
- **Total images:** 764
- **Annotation format:** PASCAL VOC XML
- **Classes:** With Helmet, Without Helmet
- **Data split:** 70% Training, 20% Validation, 10% Testing

The dataset contains an imbalance between the two classes, with fewer images representing riders without helmets.

## Project Workflow

### 1. Dataset Preprocessing
Collected and organized the helmet detection dataset and examined the distribution of the two classes.

### 2. Annotation Conversion
Developed `xml_to_yolo.py` to convert PASCAL VOC XML bounding-box annotations into YOLO-compatible normalized text labels.

### 3. Dataset Splitting
Developed `split_dataset.py` to divide the dataset into training, validation, and testing sets and generate the `data.yaml` configuration file.

### 4. Model Training
Trained the YOLOv8n model using Google Colab with the following configuration:

| Parameter | Value |
|---|---|
| Model | YOLOv8n |
| Epochs configured | 50 |
| Training epochs completed | 47 |
| Image size | 640 × 640 |
| Batch size | 16 |
| Early stopping patience | 15 |

### 5. Model Evaluation
Evaluated model performance using precision, recall, mAP@50, and mAP@50–95.

## Model Performance

The following are the best validation metrics recorded in `results.csv` during the 47-epoch training run.

| Evaluation Metric | Result |
|---|---:|
| Precision | 77.98% |
| Recall | 77.93% |
| mAP@50 | 77.26% |
| mAP@50–95 | 46.37% |

**Best validation mAP@50 was recorded at epoch 40.**

## Results and Visualizations

The `helmet-results/` directory contains:

- Training and validation performance plots
- Precision, recall, F1, and PR curves
- Confusion matrix
- Normalized confusion matrix
- Dataset label distribution
- Validation ground-truth and prediction samples

These visualizations help analyze model performance and identify classification errors.

## Class Imbalance Analysis

The dataset contains fewer examples of riders without helmets than riders wearing helmets.

This imbalance affects the model's ability to detect the minority class consistently. The results highlight the importance of balanced training
