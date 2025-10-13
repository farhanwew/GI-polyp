# Gastrointestinal Polyp Object Detection 
---

## Dataset

### Dataset Selection

This project uses the **Gastroenteroscopic Polyp** dataset ([Science Data Bank link](https://www.scidb.cn/en/detail?dataSetId=0ba79c744d05419d99c7af26618ee402&version=V2)), which contains **5460 images** with bounding box annotations for polyps (single-class). The dataset is characterized by:
- High diversity (multi-source, multi-hospital)
- Various perspectives, lighting, and resolutions (300×243 to 2559×1432)
- High-quality annotations
- Direct clinical relevance for early colorectal cancer screening

### Dataset Format

- **YOLO**: `.txt` files per image with `class, x_center, y_center, width, height` (normalized)
- **RF-DETR**: COCO JSON (`images`, `categories`, `annotations`)

---

## Object Detection Methods

### Methods Used

Three object detection models are implemented:

- **YOLOv8**: Single-stage, anchor-free, improved backbone/head for better accuracy-speed trade-off.
- **YOLOv11**: Newer YOLO version, lighter, improved feature extraction and attention.
- **RF-DETR**: Transformer-based, single-stage, real-time, end-to-end without NMS, efficient query selection.

### Architecture Brief

- **YOLOv8**: CSPDarknet backbone, PAN neck, decoupled anchor-free head.
- **YOLOv11**: Enhanced CSPDarknet, improved PAN with spatial channel attention, optimized head.
- **RF-DETR**: ViT/DINOv2 backbone, C2f projector, lightweight DETR-style decoder, hybrid query.

---

## Evaluation Metrics

- **IoU (Intersection over Union)**: Overlap between prediction and ground truth.
- **mAP@50**: Mean Average Precision at IoU 0.5.
- **mAP@50-95**: Mean Average Precision at IoU 0.5 to 0.95 (interval 0.05).
- **Precision & Recall Box**: Bounding box prediction accuracy and completeness.

---

## Implementation & Experiments

### Experimental Setup

- **Frameworks**: Ultralytics (YOLOv8/v11), RF-DETR Python package
- **Hardware**: Google Colab GPU L4 (24GB VRAM)
- **Data split**: 80% training, 20% validation
- **Monitoring**: [Weights & Biases](https://wandb.ai), [MLflow](https://dagshub.com/farhanwew/polyp-object-detection.mlflow)
- **Data cleaning**: Image and annotation integrity validation

### Experiment Configurations

| Experiment | Model            | Epochs | Batch Size | Image Size |
|:----------:|:----------------|:------:|:----------:|:----------:|
| 1          | YOLOv8-Nano     | 50     | 64         | 640×640    |
| 2          | YOLOv8-Nano     | 100    | 64         | 640×640    |
| 3          | YOLOv8-Medium   | 100    | 64         | 640×640    |
| 4          | YOLOv8-Medium   | 100    | 32         | 640×640    |
| 5          | YOLOv11-Nano    | 100    | 64         | 640×640    |
| 6          | RF-DETR-Medium  | 30     | 12         | 640×640    |

- Exp 1-2: Effect of epochs
- Exp 3-4: Effect of batch size
- Exp 5: YOLOv11 vs YOLOv8
- Exp 6: RF-DETR (transformer-based)

### Experiment Notebooks

| Experiment | Model             | Notebook                                                                                                                                            |
|:----------:|:------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|
| 1          | YOLOv8-Nano 50ep  | [YOLOv8/1.50_epoch_Notebook_Gastrointestinal_endoscopy_polyp.ipynb](https://github.com/farhanwew/GI-polyp/blob/main/Yolo8/1.50_epoch_Notebook_Gastrointestinal_endoscopy_polyp.ipynb)   |
| 2          | YOLOv8-Nano 100ep | [YOLOv8/3.Notebook_Gastrointestinal_endoscopy_polyp.ipynb](https://github.com/farhanwew/GI-polyp/blob/main/Yolo8/3.Notebook_Gastrointestinal_endoscopy_polyp.ipynb)                    |
| 3          | YOLOv8-Medium 64bs| [YOLOv8/4. yolom - batch size 64 Notebook_Gastrointestinal_endoscopy_polyp (1).ipynb](https://github.com/farhanwew/GI-polyp/blob/main/Yolo8/4.%20yolom%20-%20batch%20size%2064%20Notebook_Gastrointestinal_endoscopy_polyp%20(1).ipynb)   |
| 4          | YOLOv8-Medium 32bs| [YOLOv8/4.Notebook_Gastrointestinal_endoscopy_polyp.ipynb](https://github.com/farhanwew/GI-polyp/blob/main/Yolo8/4.Notebook_Gastrointestinal_endoscopy_polyp.ipynb)                    |
| 5          | YOLOv11-Nano      | [yolo11/1.Notebook_Gastrointestinal_endoscopy_polyp.ipynb](https://github.com/farhanwew/GI-polyp/blob/main/yolo11/1.Notebook_Gastrointestinal_endoscopy_polyp.ipynb)                   |
| 6          | RF-DETR-Medium    | [RT-DETR/1.Finetune_rf_detr_on_detection_dataset.ipynb](https://github.com/farhanwew/GI-polyp/blob/main/RT-DETR/1.Finetune_rf_detr_on_detection_dataset.ipynb)                         |

### MLflow Experiment Tracking

| Experiment | Model             | MLflow Run Link                                                                                                                                      |
|:----------:|:------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------|
| 1          | YOLOv8-Nano 50ep  | [MLflow Run 1](https://dagshub.com/farhanwew/polyp-object-detection.mlflow/#/experiments/0/runs/3f65a1764b344cb1b1e88bc4b45278f4)                   |
| 2          | YOLOv8-Nano 100ep | [MLflow Run 2](https://dagshub.com/farhanwew/polyp-object-detection.mlflow/#/experiments/0/runs/4d8b0da495ec4179b57e0fe69fe6b493)                   |
| 3          | YOLOv8-Medium 64bs| [MLflow Run 3](https://dagshub.com/farhanwew/polyp-object-detection.mlflow/#/experiments/0/runs/987e3cbb83824df1bf75ca4932006c2a)                  |
| 4          | YOLOv8-Medium 32bs| [MLflow Run 4](https://dagshub.com/farhanwew/polyp-object-detection.mlflow/#/experiments/0/runs/dea92e0808ea42c8b158aa5ff2163e50)                   |
| 5          | YOLOv11-Nano      | [MLflow Run 5](https://dagshub.com/farhanwew/polyp-object-detection.mlflow/#/experiments/2/runs/9512421ab7bb40d5955ba827093b1682)                   |
| 6          | RF-DETR-Medium    | [MLflow Run 6](https://dagshub.com/farhanwew/polyp-object-detection.mlflow/#/experiments/3/runs/728a52d3b5044555bf23fcb95a6e87aa)                   |

### Dataset Format for Each Model

- **YOLOv8/YOLOv11**: YOLO format (`.txt`) with normalized bounding box coordinates
- **RF-DETR**: COCO JSON format with absolute bounding box coordinates

### Tracking and Monitoring

- **Weights & Biases (W&B)**: Real-time monitoring for loss, metrics, and visualization
- **MLflow**: Model versioning and artifact management

### Validation and Preprocessing

- Image file integrity checks (remove corrupt images)
- Consistency check between annotation and images

---

## Results & Analysis

### Model Performance

| Experiment | Model              | Precision | Recall  | mAP@50 | mAP@50-95 |
|:----------:|:------------------|:---------:|:-------:|:------:|:---------:|
| 1          | YOLOv8n (50 ep)   | 0.9089    | 0.8556  | 0.9284 | 0.6797    |
| 2          | YOLOv8n (100 ep)  | 0.8754    | 0.8739  | 0.9160 | 0.6702    |
| 3          | YOLOv8m (64 bs)   | 0.9022    | 0.8281  | 0.9062 | 0.6596    |
| 4          | YOLOv8m (32 bs)   | 0.9015    | 0.8287  | 0.9086 | 0.6737    |
| 5          | YOLOv11n          | 0.8736    | 0.8596  | 0.9180 | 0.6803    |
| 6          | RF-DETR-Medium    | **0.9402**| **0.8800**| **0.9471**| **0.7130**|

### Analysis

- **YOLOv8-Nano**: More epochs increase recall but may decrease precision (overfitting).
- **YOLOv8-Medium**: Smaller batch size (32) slightly improves generalization.
- **YOLOv11-Nano**: Outperforms YOLOv8 on mAP@50-95.
- **RF-DETR-Medium**: Best performance across all metrics; excels in mAP@50-95.

### Visualization

Sample detection results (bounding box & confidence score) are available in the [`results/`](https://github.com/farhanwew/GI-polyp/tree/main/results) folder.

#### Example RF-DETR Detection

![Example RT-DETR Result](https://files.catbox.moe/dluo4d.png)

*Left: ground truth annotation. Right: RF-DETR model prediction (confidence 0.75, high IoU with ground truth).*

---

## Conclusion

- **RF-DETR-Medium**: Best model (Precision 0.9402, Recall 0.8800, mAP@50 0.9471, mAP@50-95 0.7130).
- **YOLOv11-Nano**: Best YOLO variant (mAP@50-95 0.6803).
- **YOLOv8-Nano 50 ep**: Highest YOLO precision (0.9089).
- Increasing epochs on YOLOv8-Nano may cause overfitting.
- Transformer-based models (RF-DETR) excel on diverse polyp datasets, with potential for further improvement with more training.

---

## Additional Resources

| Resource                             | URL                                                                                                              |
|-------------------------------------- |------------------------------------------------------------------------------------------------------------------|
| **Source Code & Notebooks**          | [GitHub Repository (Jupyter Notebooks Folder)](https://github.com/farhanwew/GI-polyp/tree/main)                  |
| **Experiment Notebooks**             | See above table ([Yolo8/](https://github.com/farhanwew/GI-polyp/tree/main/Yolo8), [yolo11/](https://github.com/farhanwew/GI-polyp/tree/main/yolo11), [RT-DETR/](https://github.com/farhanwew/GI-polyp/tree/main/RT-DETR)) |
| **Experiment Tracking (MLflow)**     | [MLflow Dashboard on DagsHub](https://dagshub.com/farhanwew/polyp-object-detection.mlflow)                       |
| **Dataset (Science Data Bank)**      | [Dataset DOI](https://www.scidb.cn/en/detail?dataSetId=0ba79c744d05419d99c7af26618ee402&version=V2)              |

---

## References

- Science Data Bank — Gastroenteroscopic Polyp Object Detection Dataset (V2). [link](https://www.scidb.cn/en/detail?dataSetId=0ba79c744d05419d99c7af26618ee402&version=V2)
- Jocher, G., et al. (2023). Ultralytics YOLO Documentation.
- Khanam, T. (2024). YOLOv11 Overview & Key Architectural Advances.
- Gupta, et al. (2025). YOLO-LAN: Precise Polyp Detection via Optimized Loss, Augmentations and Negatives (Kvasir-SEG).
- RF-DETR Official Documentation.
