# NN-Project: PlantVillage Classification

This project builds a machine learning pipeline for plant disease classification using the PlantVillage dataset, with a focus on **data leakage prevention**, **leaf-level splitting**, and **robustness testing (color vs segmented images)**.

**Python version used:** 3.12  
**Dependencies:** See `requirements.txt`

Install dependencies:
```
pip install -r requirements.txt
```

---

## Project Structure
```
NN-Project/
│
├── raw_data/
│   └── PlantVillage-Dataset/
│       (Download from: https://github.com/spMohanty/PlantVillage-Dataset/tree/master)
│        └── raw/
│        └── leaf-map.json
│
├── data/
│   ├── metadata.csv
│   ├── metadata_clean.csv
│   └── split.csv
│
├── scripts/
│   ├── build_metadata.py
│   ├── create_split.py
│   ├── dataset.py
│   ├── baseline_logreg.py
│   └── train.py
│
├── results/
│
└── README.md
```
---

## Pipeline Overview (Running Order)

Run the scripts in the following order:

### 1. build_metadata.py
- Scans image folders
- Extracts:
  - image_path
  - label
  - crop
  - disease
  - leaf_id
  - width / height
- Outputs:
  - `data/metadata.csv`
  - `data/metadata_clean.csv` (fallback + unmatched segmented removed)

### 2. create_split.py
- Performs **leaf-level train/val/test split**
- Prevents data leakage (same leaf cannot appear in multiple splits)
- Uses **color images as reference** and applies same split to segmented images
- Outputs:
  - `data/split.csv`

### 3. dataset.py
- PyTorch Dataset class
- Reads `split.csv`
- Loads images and labels
- Used by both logistic regression and CNN training

### 4. baseline_logreg.py
- Logistic Regression baseline
- Uses resized + flattened pixel features
- Evaluates on:
  - Test Color
  - Test Segmented

### 5. train.py
- CNN model (ResNet18)
- Trains on:
  - Train Color
- Validates on:
  - Val Color
- Tests on:
  - Test Color
  - Test Segmented

---

## Key Experiment Design

| Train Data | Test Data | Purpose |
|-------------|-----------|---------|
| Color | Color | Standard performance |
| Color | Segmented | Background removal test |
| Segmented | Segmented | Leaf-only training |
| Color + Segmented | Color | Data augmentation |

This allows us to study **background dependence and robustness**.

---

## Results

All model outputs, logs, and evaluation metrics should be saved in:
