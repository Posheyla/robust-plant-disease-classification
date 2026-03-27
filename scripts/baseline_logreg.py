import os
import cv2
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import ConvergenceWarning
import warnings


SPLIT_CSV = "data/split.csv"
BASE_DIR = "raw_data/PlantVillage-Dataset"

IMAGE_SIZE = 32   # baseline resize
MAX_ITER = 1000
RANDOM_STATE = 42

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_split_dataframe(split_csv, split, data_type="color"):
    df = pd.read_csv(split_csv)
    df = df[(df["split"] == split) & (df["data_type"] == data_type)].copy()
    df = df.reset_index(drop=True)
    return df


def build_label_mapping(split_csv):
    df = pd.read_csv(split_csv)
    labels = sorted(df["label"].unique())
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    return label_to_idx, idx_to_label


def load_image_as_feature(image_path, image_size=32):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32) / 255.0

    # flatten RGB pixels
    feat = img.reshape(-1)
    return feat


def dataframe_to_xy(df, label_to_idx, base_dir, image_size=32):
    X = []
    y = []

    for _, row in df.iterrows():
        full_path = os.path.join(base_dir, row["image_path"])
        feat = load_image_as_feature(full_path, image_size=image_size)
        X.append(feat)
        y.append(label_to_idx[row["label"]])

    X = np.array(X)
    y = np.array(y)
    return X, y


def evaluate_model(model, X, y, split_name, idx_to_label):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)

    print(f"\n{split_name} Accuracy: {acc:.4f}")
    print(f"{split_name} Classification Report:")
    print(classification_report(y, y_pred, target_names=[idx_to_label[i] for i in sorted(idx_to_label.keys())], zero_division=0))

    return acc, y_pred


def main():
    label_to_idx, idx_to_label = build_label_mapping(SPLIT_CSV)

    train_df = load_split_dataframe(SPLIT_CSV, split="train", data_type="color")
    val_df = load_split_dataframe(SPLIT_CSV, split="val", data_type="color")
    test_color_df = load_split_dataframe(SPLIT_CSV, split="test", data_type="color")
    test_seg_df = load_split_dataframe(SPLIT_CSV, split="test", data_type="segmented")

    print("Train images:", len(train_df))
    print("Val images:", len(val_df))
    print("Test color images:", len(test_color_df))
    print("Test segmented images:", len(test_seg_df))

    X_train, y_train = dataframe_to_xy(train_df, label_to_idx, BASE_DIR, image_size=IMAGE_SIZE)
    X_val, y_val = dataframe_to_xy(val_df, label_to_idx, BASE_DIR, image_size=IMAGE_SIZE)
    X_test_color, y_test_color = dataframe_to_xy(test_color_df, label_to_idx, BASE_DIR, image_size=IMAGE_SIZE)
    X_test_seg, y_test_seg = dataframe_to_xy(test_seg_df, label_to_idx, BASE_DIR, image_size=IMAGE_SIZE)

    print("\nFeature shape:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test_color:", X_test_color.shape)
    print("X_test_seg:", X_test_seg.shape)

    # multinomial logistic regression baseline
    model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        solver="lbfgs", 
        max_iter=1000,
        n_jobs=-1
    ))
])

    print("\nTraining logistic regression baseline...")
    model.fit(X_train, y_train)

    evaluate_model(model, X_val, y_val, "Validation", idx_to_label)
    evaluate_model(model, X_test_color, y_test_color, "Test Color", idx_to_label)
    evaluate_model(model, X_test_seg, y_test_seg, "Test Segmented", idx_to_label)


if __name__ == "__main__":
    main()