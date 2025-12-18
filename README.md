# Coxit Task — Cabinet Classification

This repository contains inference and evaluation code for cabinet image classification using multiple deep learning models.  
The task focuses on comparing different model families on a unified test set.

---

## Objective

Classify cabinet images into the following categories:

- **Base Cabinet – Open** (`lc:bcabo`)
- **Wall Cabinet – Open** (`lc:wcabo`)
- **Miscellaneous Cabinet – Insulated** (`lc:muscabinso`)
- **Wall Cabinet – Open Cubbie** (`lc:wcabcub`)
- **Base Cabinet – Open Cubbie** (`lc:bcabocub`)

---

## Models Used

The following classification models were trained and evaluated:

- **YOLOv8 / YOLOv11 (classification heads)**
- **EfficientNet (B0 / B2)**
- **ConvNeXt (Tiny)**

All models are evaluated using the same test dataset and comparable metrics.

---

## Data & Weights

Model weights and test dataset are stored externally due to size constraints.

🔗 **Google Drive:**  
https://drive.google.com/drive/folders/1U6UVj4NxoxD7vxPspGEWzoa8rwbasje5?usp=sharing

After downloading, place files as follows:
```
weights/ # model weights (.pt / .pth)
data/cabinet_dataset/test/ # test dataset
```

Expected dataset structure:
```
data/cabinet_dataset/
test/
<class_name>/*.png
```

---

## Repository Structure

- `inference_yolo.py` — inference and metrics for YOLO classification models  
- `inference_efficientnet.py` — inference and metrics for EfficientNet models  
- `inference_convnext.py` — inference and metrics for ConvNeXt models  
- `plot_model_comparison.py` — accuracy vs latency comparison plot  
- `data2dataset.py` — dataset preparation script (used offline)  
- `oversample.py` — class balancing utility (training-time)  
- `inference_results/` — saved metrics, confusion matrices, visualizations  
- `weights/` — empty folder tracked in Git (weights downloaded separately)

---

## How to Run Inference

All inference scripts:
- run on **test split only**
- compute accuracy, precision, recall, F1-score
- generate normalized confusion matrices
- save predictions and visual examples
- store results in a subfolder named after the weights file

---

### YOLO Classification

```bash
python inference_yolo.py \
  --model weights/yolov8n_cls_cabinet_224_v1.pt \
  --data_root data/cabinet_dataset \
  --split test \
  --outdir inference_results \
  --device 0
```

### EfficientNet

```bash
python inference_efficientnet.py \
  --ckpt weights/efficientnetb2_cabinet_224_v1.pt \
  --model tf_efficientnet_b2 \
  --data_root data/cabinet_dataset \
  --split test \
  --outdir inference_results
```

### ConvNeXt

```bash
python inference_convnext.py \
  --ckpt weights/convnext_tiny_cabinet_224_v1.pth \
  --model convnext_tiny \
  --data_root data/cabinet_dataset \
  --split test \
  --outdir inference_results
```

---

## Outputs

For each model, results are saved to:
```
inference_results/<weights_name>/
```

Including:
- metrics summary (*.txt)
- prediction CSV
- confusion matrix (raw + normalized)
- example prediction grids

---

## Notes
- Training artifacts and full datasets are intentionally excluded from GitHub
- Evaluation is fully reproducible using provided scripts
- All reported metrics are computed on a held-out test set
