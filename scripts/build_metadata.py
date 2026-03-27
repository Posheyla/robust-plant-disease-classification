"""
This script generates:
1. metadata.csv       -> raw metadata extracted from the dataset
2. metadata_clean.csv -> cleaned metadata used for EDA and splitting

Cleaning rules:
- remove rows with fallback leaf_id
- remove segmented images without matching color images
"""

from pathlib import Path
from collections import defaultdict
import pandas as pd
import json
import os
from PIL import Image


# Configuration
REPO_PATH = Path("raw_data/PlantVillage-Dataset")
LEAF_MAP_PATH = REPO_PATH / "leaf-map.json"
OUTPUT_PATH = Path("data/metadata.csv")
CLEAN_OUTPUT_PATH = Path("data/metadata_clean.csv")

DATA_TYPES = ["color", "segmented"]


class PlantVillageMetadataExtractor:
    """
    Extract metadata from local PlantVillage dataset.

    Main responsibilities:
    1. Scan local image files
    2. Parse label / crop / disease / filename / data_type
    3. Extract image_identifier from filename
    4. Assign leaf_id using leaf-map.json
    5. Record image size
    """

    def __init__(self, data_dir, leaf_map_path=None):
        self.data_dir = Path(data_dir)
        self.leaf_map = {}

        # Maps image IDs to physical leaf identifiers
        if leaf_map_path and os.path.exists(leaf_map_path):
            with open(leaf_map_path, "r") as f:
                self.leaf_map = json.load(f)
            print(f"Loaded leaf map with {len(self.leaf_map)} keys.")
        else:
            print("WARNING: No leaf_map provided. Data leakage may occur.")

    def extract_metadata(self, image_rel_path):
        """
        Extract all metadata from a single image path.

        Input: "raw/color/Apple___Apple_scab/1.jpg"
        Output: {
            "image_path": "raw/color/Apple___Apple_scab/1.jpg",
            "filename": "1.jpg",
            "image_identifier": "1",
            "data_type": "color",
            "label": "Apple___Apple_scab",
            "crop": "Apple",
            "disease": "Apple_scab",
            "leaf_id": "leaf_42_Apple___Apple_scab",
        }
        """
        parts = image_rel_path.split("/")

        if len(parts) < 4:
            return None

        data_type = parts[1]
        label = parts[2]
        file_name = parts[3]

        sub_parts = label.split("___")
        crop = sub_parts[0]
        disease = sub_parts[1] if len(sub_parts) > 1 else "healthy"
        is_healthy = int(disease.lower() == "healthy")

        image_identifier = self._get_image_identifier(file_name)
        leaf_id = self._get_leaf_id(image_identifier, label)

        abs_image_path = self.data_dir / image_rel_path
        try:
            with Image.open(abs_image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Failed to read image size for {abs_image_path}: {e}")
            width, height = None, None

        return {
            "image_path": image_rel_path,
            "filename": file_name,
            "image_identifier": image_identifier,
            "data_type": data_type,
            "label": label,
            "crop": crop,
            "disease": disease,
            "is_healthy": is_healthy,
            "leaf_id": leaf_id,
            "width": width,
            "height": height,
        }

    def _get_image_identifier(self, file_name):
        """
        Convert filename to a canonical image identifier used for leaf_map lookup.

        Examples:
        - '1.jpg' -> '1'
        - '1_final_masked.jpg' -> '1'
        - 'abc___1.jpg' -> '1'
        - '1copy.jpg' -> '1'
        - 'Com.G_TgS_FL 0004.JPG' -> 'com.g_fl 0004'
        """
        image_identifier = file_name.replace("_final_masked", "")

        if "___" in image_identifier:
            image_identifier = image_identifier.split("___")[-1]

        image_identifier = image_identifier.split("copy")[0]

        # remove extension more robustly
        image_identifier = image_identifier.rsplit(".", 1)[0]

        image_identifier = image_identifier.strip().lower()
        
        ## Fix inconsistency in leaf_map
        # Tomato target spot: "com.g_tgs_fl 0004" -> "com.g_fl 0004"
        image_identifier = image_identifier.replace("com.g_tgs_fl", "com.g_fl")
        


        return image_identifier

    def _get_leaf_id(self, image_identifier, label):
        """
        Assign physical leaf group ID from image_identifier + leaf_map.
        This is critical for leakage-free splitting.
        """
        suggestions = self.leaf_map.get(image_identifier)

        if not suggestions:
            return f"fallback_{image_identifier}"

        if len(suggestions) == 1:
            return suggestions[0]

        for suggestion in suggestions:
            if label in suggestion:
                return suggestion

        return f"fallback_{image_identifier}"

    def scan_dataset(self, data_type="color"):
        """
        Scan raw/{data_type}/ and return:
        - all_metadata: list of row dicts
        - leaf_groups: dict(leaf_id -> list of rows)
        """
        all_metadata = []
        leaf_groups = defaultdict(list)

        base_path = self.data_dir / "raw" / data_type

        if not base_path.exists():
            raise FileNotFoundError(f"Data path not found: {base_path}")

        for class_dir in sorted(base_path.iterdir()):
            if not class_dir.is_dir():
                continue

            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    rel_path = f"raw/{data_type}/{class_dir.name}/{img_file.name}"

                    metadata = self.extract_metadata(rel_path)
                    if metadata:
                        all_metadata.append(metadata)
                        leaf_groups[metadata["leaf_id"]].append(metadata)

        return all_metadata, leaf_groups


# -------------------------
# Step 1: Build raw metadata
# -------------------------
all_rows = []

extractor = PlantVillageMetadataExtractor(
    data_dir=REPO_PATH,
    leaf_map_path=LEAF_MAP_PATH
)

for data_type in DATA_TYPES:
    print(f"\nScanning dataset: {data_type}")
    rows, leaf_groups = extractor.scan_dataset(data_type=data_type)
    print(f"Found {len(rows)} images from {len(leaf_groups)} unique leaf groups.")
    all_rows.extend(rows)

metadata_df = pd.DataFrame(all_rows)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
metadata_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nSaved raw metadata to: {OUTPUT_PATH}")
print(metadata_df.shape)


# -------------------------
# Step 2: Clean metadata
# -------------------------
df_clean = metadata_df.copy()

# Remove fallback leaf_id
before_fallback = len(df_clean)
df_clean = df_clean[~df_clean["leaf_id"].str.startswith("fallback_")].copy()
after_fallback = len(df_clean)

print("\nRemoved fallback leaf_id rows:")
print(f"Before: {before_fallback}, After: {after_fallback}, Removed: {before_fallback - after_fallback}")

# Remove unmatched segmented images
df_color = df_clean[df_clean["data_type"] == "color"].copy()
df_seg = df_clean[df_clean["data_type"] == "segmented"].copy()

color_ids = set(df_color["image_identifier"])
seg_ids = set(df_seg["image_identifier"])

only_seg = seg_ids - color_ids
only_color = color_ids - seg_ids

print("\nColor/Segmented matching check after fallback removal:")
print(f"Matched image_identifier: {len(color_ids & seg_ids)}")
print(f"Only in color: {len(only_color)}")
print(f"Only in segmented: {len(only_seg)}")

if len(only_seg) > 0:
    print("Removing segmented-only image_identifiers:")
    print(sorted(list(only_seg))[:10])

    df_clean = df_clean[
        ~(
            (df_clean["data_type"] == "segmented") &
            (df_clean["image_identifier"].isin(only_seg))
        )
    ].copy()

# Optional final summary
print("\nFinal cleaned metadata summary:")
print(f"Total rows: {len(df_clean)}")
print(f"Color rows: {(df_clean['data_type'] == 'color').sum()}")
print(f"Segmented rows: {(df_clean['data_type'] == 'segmented').sum()}")
print(f"Unique leaf_id: {df_clean['leaf_id'].nunique()}")
print(f"Unique labels: {df_clean['label'].nunique()}")

df_clean.to_csv(CLEAN_OUTPUT_PATH, index=False)
print(f"\nSaved cleaned metadata to: {CLEAN_OUTPUT_PATH}")