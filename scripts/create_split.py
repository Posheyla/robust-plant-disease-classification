from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

METADATA_PATH = Path("data/metadata.csv")
OUTPUT_PATH = Path("data/split.csv")


def create_train_val_test_split(
    metadata_path=METADATA_PATH,
    output_path=OUTPUT_PATH,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
    data_type="color"
):
    df = pd.read_csv(metadata_path)

    # only split one data source at a time
    df = df[df["data_type"] == data_type].copy()

    if df["leaf_id"].isna().any():
        raise ValueError("Some rows have missing leaf_id.")

    if abs(train_size + val_size + test_size - 1.0) > 1e-8:
        raise ValueError("train_size + val_size + test_size must sum to 1.")

    # first split: train vs temp
    gss1 = GroupShuffleSplit(
        n_splits=1,
        test_size=(1 - train_size),
        random_state=random_state
    )
    train_idx, temp_idx = next(gss1.split(df, groups=df["leaf_id"]))

    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    # second split: val vs test
    val_ratio_within_temp = val_size / (val_size + test_size)

    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=(1 - val_ratio_within_temp),
        random_state=random_state
    )
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["leaf_id"]))

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # leakage check
    leakage_check = split_df.groupby("leaf_id")["split"].nunique()
    leakage_count = (leakage_check > 1).sum()

    print("\nSplit summary:")
    print(split_df["split"].value_counts())

    print("\nLeaf-group leakage check:")
    if leakage_count == 0:
        print("No leakage detected.")
    else:
        print(f"WARNING: {leakage_count} leaf groups appear in multiple splits.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(output_path, index=False)
    print(f"\nSaved split file to: {output_path}")

    return split_df


if __name__ == "__main__":
    create_train_val_test_split()