# Robust Plant Disease Classification Under Real-World Distribution Shifts

This project investigates the robustness of convolutional neural networks for plant disease classification under real-world distribution shifts.

Using the PlantVillage dataset, we evaluate how models trained on clean color images generalize to segmented and shifted image distributions. The project focuses on:

- Data leakage prevention through leaf-level splitting
- CNN vs Logistic Regression comparison
- Distribution shift evaluation
- Robustness testing under noise, blur, and brightness changes
- Real-world generalization analysis

Developed as part of ECE 525: Neural Networks at NC State University.

**Python version used:** 3.12  
**Dependencies:** See `requirements.txt`

Install dependencies:
```
pip install -r requirements.txt
```
## Key Results

| Model | Test Set | Accuracy | Macro-F1 |
|---|---|---|---|
| CNN | Color Images | 96% | 0.94 |
| CNN | Segmented Images | 47% | 0.30 |
| Logistic Regression | Color Images | 63.65% | - |

Key finding:
Models trained on clean benchmark data experience severe performance degradation under distribution shift.

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
- Custom CNN architecture
- 4 convolutional blocks
- BatchNorm + ReLU + MaxPooling
- Dropout regularization
- Adam optimizer

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

## Robustness Evaluation

The trained models were evaluated under several real-world perturbations:

- Brightness changes
- Contrast changes
- Blur corruption
- Noise corruption
- Segmented leaf distributions

Results show that CNN performance remains relatively stable under brightness and contrast variation, but degrades significantly under blur, noise, and distribution shift conditions.
## Research Poster

Full poster available here:

- [Poster PDF](results/cnn_poster.pdf)

Preview:

![Research Poster](results/poster_preview.png)

## Authors

- Luisa Chavez
- Chase Wrenn
- Joyce Zhou

## Acknowledgements

Dataset:
PlantVillage Dataset by Mohanty et al.

Course:
ECE 525 — Neural Networks
North Carolina State University
