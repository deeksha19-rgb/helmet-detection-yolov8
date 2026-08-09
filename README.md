# Helmet Detection — YOLOv8

Object detection pipeline that identifies motorcycle/bicycle riders wearing helmets vs. not, built on YOLOv8n.

## Pipeline

1. **Dataset**: [Helmet Detection (andrewmvd)](https://www.kaggle.com/datasets/andrewmvd/helmet-detection) from Kaggle — 764 images, PASCAL VOC XML annotations, 2 classes (`With Helmet`, `Without Helmet`)
2. **`xml_to_yolo.py`** — converts PASCAL VOC XML bounding boxes into YOLO's normalized `.txt` format, auto-discovering class names from the annotations
3. **`split_dataset.py`** — splits the converted dataset into train/val/test (70/20/10), producing the folder structure and `data.yaml` that Ultralytics expects
4. **Training** — YOLOv8n (nano), 50 epochs configured with early stopping (`patience=15`), trained on a Colab T4 GPU

## Results

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| All | 0.706 | 0.757 | 0.747 | 0.471 |
| With Helmet | 0.866 | 0.813 | 0.886 | 0.578 |
| Without Helmet | 0.545 | 0.700 | 0.607 | 0.364 |

Training stopped early at epoch 32 (best checkpoint), with no further improvement over the next 15 epochs.

## Known limitation

The dataset has a **2.5:1 class imbalance** (111 "With Helmet" vs. 43 "Without Helmet" images in the training split), which shows up directly in the per-class mAP50 gap above. This is a data volume issue, not a training or architecture issue — further epochs plateaued without closing the gap. Addressing it would require sourcing additional "Without Helmet" images rather than extending training.

## Reproducing

```bash
# 1. Download the dataset from Kaggle and place images/ and annotations/ folders
#    at the paths configured in xml_to_yolo.py

python xml_to_yolo.py      # converts XML -> YOLO labels
python split_dataset.py    # creates train/val/test splits + data.yaml

# 3. Train (requires: pip install ultralytics)
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="dataset_split/data.yaml", epochs=50, imgsz=640, batch=16, patience=15)
```
