
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "NASA Data"
TEST_DIR = DATA_ROOT / "data" / "data" / "test"
LABELS_FILE = DATA_ROOT / "labeled_anomalies.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_intervals(value):
    intervals = literal_eval(value)

    if not isinstance(intervals, list):
        raise ValueError(f"Expected interval list, got {value!r}")

    parsed = []

    for interval in intervals:
        if len(interval) != 2:
            raise ValueError(f"Invalid interval: {interval!r}")

        start, end = interval

        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(f"Non-integer interval: {interval!r}")

        if start < 0 or end < start:
            raise ValueError(f"Invalid interval bounds: {interval!r}")

        parsed.append((start, end))

    return parsed


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(LABELS_FILE)
    smap = metadata[metadata["spacecraft"] == "SMAP"].copy()

    summaries = []
    total_points = 0
    total_anomalies = 0

    for channel, rows in smap.groupby("chan_id", sort=True):
        test_path = TEST_DIR / f"{channel}.npy"

        if not test_path.exists():
            raise FileNotFoundError(
                f"Missing test array for labelled channel {channel}"
            )

        X = np.load(test_path, allow_pickle=False)

        if X.ndim != 2:
            raise ValueError(
                f"{channel}: expected 2D array, got {X.shape}"
            )

        if not np.isfinite(X).all():
            raise ValueError(f"{channel}: non-finite test values")

        # Combine all metadata rows for this channel.
        # This correctly preserves both P-2 anomaly events.
        intervals = []

        for _, row in rows.iterrows():
            intervals.extend(parse_intervals(row["anomaly_sequences"]))

            if int(row["num_values"]) != len(X):
                raise ValueError(
                    f"{channel}: metadata length does not match test array"
                )

        # Original Telemanom labels use inclusive start/end indices.
        y = np.zeros(len(X), dtype=np.int8)

        for start, end in intervals:
            if end >= len(y):
                raise ValueError(
                    f"{channel}: interval [{start}, {end}] exceeds "
                    f"test length {len(y)}"
                )

            y[start:end + 1] = 1

        anomaly_points = int(y.sum())
        normal_points = int(len(y) - anomaly_points)

        summaries.append({
            "chan_id": channel,
            "spacecraft": "SMAP",
            "test_rows": len(X),
            "num_features": X.shape[1],
            "num_metadata_rows": len(rows),
            "num_anomaly_intervals": len(intervals),
            "anomaly_points": anomaly_points,
            "normal_points": normal_points,
            "anomaly_fraction": anomaly_points / len(y),
        })

        total_points += len(y)
        total_anomalies += anomaly_points

    report = pd.DataFrame(summaries)
    output_path = RESULTS_DIR / "smap_label_summary.csv"
    report.to_csv(output_path, index=False)

    print("=" * 70)
    print("SMAP LABEL SUMMARY")
    print("=" * 70)
    print(f"Unique labelled SMAP channels: {len(report)}")
    print(f"Total test time steps: {total_points}")
    print(f"Anomalous time steps: {total_anomalies}")
    print(f"Normal time steps: {total_points - total_anomalies}")

    if total_points:
        print(f"Anomaly fraction: {total_anomalies / total_points:.4%}")

    print("\nPer-channel summary:")
    print(report.to_string(index=False))

    print(f"\nSaved report: {output_path}")


if __name__ == "__main__":
    main()
