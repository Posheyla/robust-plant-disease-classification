import os
from typing import Dict, Optional

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


class PlantVillageDataset(Dataset):
    """
    PyTorch Dataset for PlantVillage using split.csv.

    Expected columns in split.csv:
        - image_path
        - label
        - data_type
        - split
        - image_identifier (optional but useful for debugging)
        - leaf_id (optional but useful for debugging)

    Parameters
    ----------
    split_csv : str
        Path to split.csv
    split : str
        One of {"train", "val", "test"}
    data_type : str
        One of {"color", "segmented"}
    transform : callable, optional
        Image transform pipeline
    base_dir : str
        Root directory that contains the raw/ folder

    Notes
    -----
    This dataset reads image paths from split.csv instead of scanning folders.
    That ensures the training / validation / test split strictly follows the
    leaf-level split and avoids data leakage.
    """

    def __init__(
        self,
        split_csv: str,
        split: str = "train",
        data_type: str = "color",
        transform=None,
        base_dir: str = "raw_data/PlantVillage-Dataset",
        label_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.split_csv = split_csv
        self.split = split
        self.data_type = data_type
        self.transform = transform
        self.base_dir = base_dir

        df = pd.read_csv(split_csv)

        required_cols = {"image_path", "label", "data_type", "split"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"split.csv is missing required columns: {missing_cols}")

        self.df = df[
            (df["split"] == split) &
            (df["data_type"] == data_type)
        ].copy().reset_index(drop=True)

        if self.df.empty:
            raise ValueError(
                f"No rows found for split='{split}' and data_type='{data_type}'."
            )

        # Build a consistent label mapping.
        # Best practice: build it once from the full CSV and share it across datasets.
        if label_to_idx is None:
            all_labels = sorted(df["label"].unique())
            self.label_to_idx = {label: i for i, label in enumerate(all_labels)}
        else:
            self.label_to_idx = label_to_idx

        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

        print(
            f"[PlantVillageDataset] split={split}, data_type={data_type}, "
            f"n_images={len(self.df)}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.base_dir, row["image_path"])
        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")

        # OpenCV reads BGR; convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_str = row["label"]
        if label_str not in self.label_to_idx:
            raise ValueError(f"Label '{label_str}' not found in label_to_idx.")

        label = self.label_to_idx[label_str]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    @staticmethod
    def build_label_mapping(split_csv: str) -> Dict[str, int]:
        """
        Build a consistent label mapping from the full split.csv.
        Use this once and pass the mapping into all dataset objects.
        """
        df = pd.read_csv(split_csv)
        all_labels = sorted(df["label"].unique())
        return {label: i for i, label in enumerate(all_labels)}