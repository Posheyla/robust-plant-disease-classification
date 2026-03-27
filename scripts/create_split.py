"""
Purpose: Create train / validation / test splits at the LEAF level to avoid data leakage.
Splitting is performed using COLOR images as the reference, and the same split
assignment is applied to SEGMENTED images to create a paired dataset.

Input: data/metadata_clean.csv

Output: data/split.csv
This file contains all rows (color + segmented) with an additional column:
    split  ∈ {"train", "val", "test"}
Each leaf_id appears in ONLY ONE split to prevent data leakage.


=== Stratified Split =====
If stratified=True:
    The split is performed at the leaf level using class label distribution,
    so that train/val/test have similar class distributions.

If stratified=False:
    The split is performed using GroupShuffleSplit only by leaf_id,
    without enforcing class balance.

===== To use the split =====
split_df = pd.read_csv("data/split.csv")
train_color = split_df[(split_df["split"] == "train") & (split_df["data_type"] == "color")]
val_color   = split_df[(split_df["split"] == "val") & (split_df["data_type"] == "color")]
test_color  = split_df[(split_df["split"] == "test") & (split_df["data_type"] == "color")]
test_seg = split_df[(split_df["split"] == "test") & (split_df["data_type"] == "segmented")]
"""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

METADATA_PATH = Path("data/metadata_clean.csv")
OUTPUT_PATH = Path("data/split.csv")


def create_train_val_test_split(
    metadata_path=METADATA_PATH,
    output_path=OUTPUT_PATH,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    stratified=True,
):
    """
    Create train/val/test split at the leaf level using color images as the reference,
    then map the same split assignment to segmented images to create a paired split file.

    Output:
        One unified split.csv containing both color and segmented rows, with columns:
        [image_path, filename, image_identifier, data_type, label, crop, disease,
         is_healthy, leaf_id, width, height, split]
    """

    df = pd.read_csv(metadata_path)

    if df.empty:
        raise ValueError("Metadata file is empty.")

    if df["leaf_id"].isna().any():
        raise ValueError("Some rows have missing leaf_id.")

    if abs(train_size + val_size + test_size - 1.0) > 1e-8:
        raise ValueError("train_size + val_size + test_size must sum to 1.")

    # -------------------------
    # Use color images as split reference
    # -------------------------
    df_color = df[df["data_type"] == "color"].copy()
    df_seg = df[df["data_type"] == "segmented"].copy()

    if df_color.empty:
        raise ValueError("No color rows found in metadata_clean.csv.")

    # Check that each leaf_id maps to exactly one label in color data
    leaf_label_check = df_color.groupby("leaf_id")["label"].nunique()
    if (leaf_label_check > 1).any():
        bad_leafs = leaf_label_check[leaf_label_check > 1]
        raise ValueError(
            "Some leaf_id values map to multiple labels in color data. "
            f"Example:\n{bad_leafs.head()}"
        )

    # Leaf-level dataframe for splitting
    leaf_df = (
        df_color.groupby("leaf_id", as_index=False)
        .agg(
            label=("label", "first"),
            crop=("crop", "first"),
            disease=("disease", "first"),
        )
    )

    print("\nLeaf-level class counts before split:")
    print(leaf_df["label"].value_counts().sort_values().head(10))

    # -------------------------
    # First split: train vs temp
    # -------------------------
    if stratified:
        sss1 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=(1 - train_size),
            random_state=random_state,
        )
        train_leaf_idx, temp_leaf_idx = next(
            sss1.split(leaf_df, leaf_df["label"])
        )
    else:
        gss1 = GroupShuffleSplit(
            n_splits=1,
            test_size=(1 - train_size),
            random_state=random_state,
        )
        train_leaf_idx, temp_leaf_idx = next(
            gss1.split(leaf_df, groups=leaf_df["leaf_id"])
        )

    train_leaf_df = leaf_df.iloc[train_leaf_idx].copy()
    temp_leaf_df = leaf_df.iloc[temp_leaf_idx].copy()

    # -------------------------
    # Second split: val vs test
    # -------------------------
    val_ratio_within_temp = val_size / (val_size + test_size)

    if stratified:
        sss2 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=(1 - val_ratio_within_temp),
            random_state=random_state,
        )
        val_leaf_idx, test_leaf_idx = next(
            sss2.split(temp_leaf_df, temp_leaf_df["label"])
        )
    else:
        gss2 = GroupShuffleSplit(
            n_splits=1,
            test_size=(1 - val_ratio_within_temp),
            random_state=random_state,
        )
        val_leaf_idx, test_leaf_idx = next(
            gss2.split(temp_leaf_df, groups=temp_leaf_df["leaf_id"])
        )

    val_leaf_df = temp_leaf_df.iloc[val_leaf_idx].copy()
    test_leaf_df = temp_leaf_df.iloc[test_leaf_idx].copy()

    # -------------------------
    # Build leaf_id -> split map
    # -------------------------
    split_map = {}

    for leaf_id in train_leaf_df["leaf_id"]:
        split_map[leaf_id] = "train"
    for leaf_id in val_leaf_df["leaf_id"]:
        split_map[leaf_id] = "val"
    for leaf_id in test_leaf_df["leaf_id"]:
        split_map[leaf_id] = "test"

    # Apply split to ALL rows (color + segmented)
    df_split = df[df["leaf_id"].isin(split_map.keys())].copy()
    df_split["split"] = df_split["leaf_id"].map(split_map)

    # -------------------------
    # Leakage check
    # -------------------------
    leakage_check = df_split.groupby("leaf_id")["split"].nunique()
    leakage_count = (leakage_check > 1).sum()

    # -------------------------
    # Paired consistency check
    # -------------------------
    # For image_identifiers appearing in both color and segmented, they should have same split
    pair_check = (
        df_split.groupby(["image_identifier", "data_type"])["split"]
        .first()
        .unstack()
    )

    inconsistent_pairs = 0
    if "color" in pair_check.columns and "segmented" in pair_check.columns:
        paired_rows = pair_check.dropna(subset=["color", "segmented"])
        inconsistent_pairs = (paired_rows["color"] != paired_rows["segmented"]).sum()

    # -------------------------
    # Print summaries
    # -------------------------
    print("\nSplit summary (image level):")
    print(df_split["split"].value_counts())

    print("\nSplit summary by data_type:")
    print(pd.crosstab(df_split["split"], df_split["data_type"]))

    print("\nSplit summary (leaf level, based on color reference):")
    print({
        "train": len(train_leaf_df),
        "val": len(val_leaf_df),
        "test": len(test_leaf_df),
    })

    print("\nClass distribution by split (color only):")
    color_split_view = df_split[df_split["data_type"] == "color"]
    print(pd.crosstab(color_split_view["split"], color_split_view["label"]))

    print("\nLeaf-group leakage check:")
    if leakage_count == 0:
        print("No leakage detected.")
    else:
        print(f"WARNING: {leakage_count} leaf groups appear in multiple splits.")

    print("\nPaired split consistency check:")
    if inconsistent_pairs == 0:
        print("All matched color/segmented pairs have consistent split assignments.")
    else:
        print(f"WARNING: {inconsistent_pairs} matched pairs have inconsistent splits.")

    # -------------------------
    # Save unified paired split file
    # -------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_split.to_csv(output_path, index=False)
    print(f"\nSaved unified paired split file to: {output_path}")

    return df_split


if __name__ == "__main__":
    create_train_val_test_split(stratified=True)