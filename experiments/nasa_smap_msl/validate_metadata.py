
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "NASA Data"
ARRAY_ROOT = DATA_ROOT / "data" / "data"

TRAIN_DIR = ARRAY_ROOT / "train"
TEST_DIR = ARRAY_ROOT / "test"
LABELS_FILE = DATA_ROOT / "labeled_anomalies.csv"

REPORT_DIR = Path(__file__).resolve().parent / "results"


def load_intervals(value):
    """Parse anomaly intervals from the CSV without evaluating code."""
    intervals = literal_eval(value)

    if not isinstance(intervals, list):
        raise ValueError(f"Expected a list of intervals, got {intervals!r}")

    parsed = []

    for interval in intervals:
        if (
            not isinstance(interval, (list, tuple))
            or len(interval) != 2
        ):
            raise ValueError(f"Invalid anomaly interval: {interval!r}")

        start, end = interval

        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"Non-integer anomaly interval: {interval!r}")

        if start < 0 or end <= start:
            raise ValueError(f"Invalid interval boundaries: {interval!r}")

        parsed.append((start, end))

    return parsed


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(LABELS_FILE)

    print("=" * 75)
    print("NASA SMAP/MSL DATA VALIDATION")
    print("=" * 75)

    print(f"Metadata rows: {len(metadata)}")
    print(f"Unique channel IDs: {metadata['chan_id'].nunique()}")
    print("\nMetadata rows per channel ID:")
    print(
        metadata["chan_id"]
        .value_counts()
        .loc[lambda counts: counts > 1]
        .to_string()
    )

    train_files = {p.stem: p for p in TRAIN_DIR.glob("*.npy")}
    test_files = {p.stem: p for p in TEST_DIR.glob("*.npy")}

    print(f"\nTraining arrays: {len(train_files)}")
    print(f"Test arrays: {len(test_files)}")

    metadata_ids = set(metadata["chan_id"].astype(str))
    train_ids = set(train_files)
    test_ids = set(test_files)

    print("\nFilename/metadata matching:")
    print("Metadata IDs without a test file:", sorted(metadata_ids - test_ids))
    print("Test files without metadata:", sorted(test_ids - metadata_ids))
    print("Metadata IDs without a train file:", sorted(metadata_ids - train_ids))
    print("Train files without metadata:", sorted(train_ids - metadata_ids))

    # Duplicate metadata IDs need investigation before using ID alone
    # to join metadata with array files.
    duplicate_ids = metadata.loc[
        metadata["chan_id"].duplicated(keep=False),
        "chan_id",
    ].unique()

    if len(duplicate_ids):
        print("\nFull rows for duplicate channel IDs:")
        print(
            metadata[metadata["chan_id"].isin(duplicate_ids)]
            .to_string(index=False)
        )

    report_rows = []
    issues = []

    for _, row in metadata.iterrows():
        channel = str(row["chan_id"])

        if channel not in test_files:
            issues.append(f"{channel}: no matching test array")
            continue

        if channel not in train_files:
            issues.append(f"{channel}: no matching training array")
            continue

        train = np.load(train_files[channel], allow_pickle=False)
        test = np.load(test_files[channel], allow_pickle=False)

        if train.ndim != 2 or test.ndim != 2:
            issues.append(
                f"{channel}: expected 2D arrays; "
                f"train={train.shape}, test={test.shape}"
            )
            continue

        if train.shape[1] != test.shape[1]:
            issues.append(f"{channel}: feature-count mismatch")

        if not np.isfinite(train).all():
            issues.append(f"{channel}: non-finite training values")

        if not np.isfinite(test).all():
            issues.append(f"{channel}: non-finite test values")

        if int(row["num_values"]) != len(test):
            issues.append(
                f"{channel}: metadata num_values={row['num_values']}, "
                f"test length={len(test)}"
            )

        try:
            intervals = load_intervals(row["anomaly_sequences"])
        except (ValueError, SyntaxError) as exc:
            issues.append(f"{channel}: invalid intervals: {exc}")
            continue

        for start, end in intervals:
            # Treat intervals as half-open [start, end) for this
            # bounds check only. We must verify endpoint conventions
            # before creating final sample labels.
            if end > len(test):
                issues.append(
                    f"{channel}: interval [{start}, {end}) exceeds "
                    f"test length {len(test)}"
                )

        report_rows.append({
            "chan_id": channel,
            "spacecraft": row["spacecraft"],
            "train_rows": len(train),
            "test_rows": len(test),
            "num_features": test.shape[1],
            "num_anomaly_intervals": len(intervals),
            "anomaly_types": row["class"],
        })

    report = pd.DataFrame(report_rows)
    report_path = REPORT_DIR / "channel_validation.csv"
    report.to_csv(report_path, index=False)

    print("\n" + "=" * 75)
    print(f"Channel records inspected: {len(report_rows)}")
    print(f"Validation issues found: {len(issues)}")
    print(f"Channel report saved to: {report_path}")

    if issues:
        print("\nIssues requiring review:")
        for issue in issues:
            print("-", issue)
    else:
        print("\nNo shape, value, or interval-boundary issues found.")

    print("\nSpacecraft counts in metadata:")
    print(metadata["spacecraft"].value_counts().to_string())


if __name__ == "__main__":
    main()
