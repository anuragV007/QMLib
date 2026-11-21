
from pathlib import Path

import numpy as np
import pandas as pd


# This script is located at:
# QMLib/experiments/nasa_smap_msl/inspect_dataset.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "NASA Data"

ARRAY_ROOT = DATA_ROOT / "data" / "data"
TRAIN_DIR = ARRAY_ROOT / "train"
TEST_DIR = ARRAY_ROOT / "test"
LABELS_FILE = DATA_ROOT / "labeled_anomalies.csv"


def inspect_arrays(directory, split_name, limit=5):
    print("\n" + "=" * 75)
    print(f"{split_name} arrays")
    print("=" * 75)

    if not directory.exists():
        print(f"Directory not found: {directory}")
        return

    files = sorted(directory.glob("*.npy"))

    print(f"Directory: {directory}")
    print(f"Number of .npy files: {len(files)}")

    for path in files[:limit]:
        array = np.load(path, allow_pickle=False)

        print(f"\nChannel: {path.stem}")
        print(f"Shape: {array.shape}")
        print(f"Data type: {array.dtype}")

        if array.size > 0 and np.issubdtype(
            array.dtype, np.number
        ):
            finite_mask = np.isfinite(array)

            print(f"Finite values: {finite_mask.sum()} / {array.size}")
            print(f"NaN count: {np.isnan(array).sum()}")
            print(f"Positive/negative infinity count: "
                  f"{np.isinf(array).sum()}")

            finite_values = array[finite_mask]

            if finite_values.size:
                print(f"Minimum: {finite_values.min()}")
                print(f"Maximum: {finite_values.max()}")
                print(f"First values: {array.reshape(-1)[:8]}")

    if len(files) > limit:
        print(f"\nShowing {limit} of {len(files)} files.")


def inspect_labels():
    print("\n" + "=" * 75)
    print("Anomaly metadata")
    print("=" * 75)

    if not LABELS_FILE.exists():
        print(f"Labels file not found: {LABELS_FILE}")
        return

    labels = pd.read_csv(LABELS_FILE)

    print(f"File: {LABELS_FILE}")
    print(f"Shape: {labels.shape}")
    print("\nColumns:")
    print(labels.dtypes.to_string())

    print("\nFirst 10 metadata rows:")
    print(labels.head(10).to_string(index=False))

    print("\nMissing values per column:")
    print(labels.isna().sum().to_string())

    for column in ["spacecraft", "chan_id", "class"]:
        if column in labels.columns:
            print(f"\nUnique values/counts for {column}:")
            print(labels[column].value_counts(dropna=False).head(20))


def main():
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset root: {DATA_ROOT}")

    inspect_arrays(TRAIN_DIR, "TRAIN")
    inspect_arrays(TEST_DIR, "TEST")
    inspect_labels()


if __name__ == "__main__":
    main()
